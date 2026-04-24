"""
maya_v2.py
----------
Variant of maya.py that supports sending commands to the MFCs *while* the
Teensy is streaming sensor data.

Design
======
The original `maya.py` opens a new `serial.Serial(...)` context every time it
sends a message or starts streaming. That prevents mixing commands and data
on the same port. This module instead keeps ONE persistent serial connection
and runs the reader in a background thread, so:

  * `start_streaming()` sends `MAYA,1`, then a background thread reads
    PACKET_SIZE byte packets and writes them to CSV with a timestamp.
  * Commands (e.g. `MFCControl.set_flow(...)`) can be sent at any time from
    the main thread. A `threading.Lock` guards `ser.write` so bytes from two
    commands never interleave. (Sensor packets and command bytes still flow
    in opposite directions on the USB Serial, so they do not collide.)
  * `stop_streaming()` sends `MAYA,0` and joins the reader thread.

Compatible with the firmware in sens_basic_ser7/. Based on observation of the
firmware:
  - The main `loop()` checks `Serial.available()` each iteration and calls
    `parseInput()`. It keeps running while `readSensTimer` (IntervalTimer ISR)
    emits binary packets. So commands CAN be processed during streaming.
  - MFC command 1 (set setpoint) and 0 (blink) and 99 (setup) do not write
    anything back on the USB Serial, so they don't corrupt the binary stream.
  - MFC command 9 (read_holding_registers) writes "DATA:..." on USB Serial,
    which WOULD corrupt the binary packet stream. `read_registers()` is
    therefore refused while streaming is active.
  - Malformed commands trigger `Serial.println("Invalid ...")` on the firmware
    side, which would also corrupt the stream. Only well-formed commands are
    sent by this module.

Note: this module does not modify the original maya.py or the firmware.
"""

import csv
import os
import threading
import time
from datetime import datetime

import serial

PACKET_SIZE = 128


class SerialCommunication:
    """Persistent serial connection with a background reader thread.

    Parameters
    ----------
    port : str
        e.g. ``/dev/ttyACM0``.
    baudrate : int
        Must match the firmware (115200).
    read_timeout : float
        Serial read timeout in seconds (default 1).
    """

    def __init__(self, port, baudrate, read_timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate, timeout=read_timeout)
        # Allow the Teensy to finish any USB enumeration-induced reset.
        time.sleep(0.1)
        self.ser.reset_input_buffer()

        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread = None

        self._file = None
        self._writer = None
        self._save_csv = False
        self._filename = None
        self._streaming = False

        # Optional in-memory sink for the latest packets.
        self._packet_buffer = None
        self._packet_buffer_lock = threading.Lock()
        self._packet_buffer_max = 0

        # Stats
        self.packets_received = 0

    # ------------------------------------------------------------------ #
    # Low level
    # ------------------------------------------------------------------ #
    def send_message(self, message, verbose=True):
        """Thread-safe write of a newline-terminated command string."""
        if not message.endswith("\n"):
            message = message + "\n"
        with self._write_lock:
            self.ser.write(message.encode())
        if verbose:
            print(f"Sent to Teensy: {message.strip()}")

    @property
    def is_streaming(self):
        return self._streaming

    # ------------------------------------------------------------------ #
    # Streaming control
    # ------------------------------------------------------------------ #
    def start_streaming(self, filename=None, save_csv=False, buffer_size=0):
        """Start background packet capture.

        Parameters
        ----------
        filename : str or None
            CSV file to append packets to. Required if ``save_csv`` is True.
        save_csv : bool
            Whether to write each packet to the CSV file.
        buffer_size : int
            If > 0, keep the last ``buffer_size`` packets in memory,
            accessible via :meth:`get_latest_packets`.
        """
        if self._streaming:
            print("Already streaming.")
            return

        if save_csv:
            if filename is None:
                raise ValueError("filename is required when save_csv=True")
            file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
            self._file = open(filename, mode="a", newline="")
            self._writer = csv.writer(self._file)
            if not file_exists:
                self._writer.writerow(["Timestamp", "Data"])
                print(f"Created {filename}.")
            else:
                print(f"Appending to existing {filename}.")
        self._save_csv = save_csv
        self._filename = filename

        if buffer_size > 0:
            self._packet_buffer = []
            self._packet_buffer_max = buffer_size
        else:
            self._packet_buffer = None
            self._packet_buffer_max = 0

        self.packets_received = 0
        self.ser.reset_input_buffer()
        self.send_message("MAYA,1", verbose=False)

        self._stop_event.clear()
        self._streaming = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="maya-reader", daemon=True
        )
        self._reader_thread.start()
        print("Streaming started.")

    def stop_streaming(self, drain_seconds=0.2):
        """Stop background capture and tell the Teensy to stop emitting."""
        if not self._streaming:
            return

        # Ask the Teensy to stop the IntervalTimer first, so no further
        # binary packets are queued on the USB Serial.
        self.send_message("MAYA,0", verbose=False)

        # Give the reader a moment to finish in-flight packets, then stop it.
        time.sleep(drain_seconds)
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        # Drop any leftover bytes (partial packet, or trailing "MAYA,0"
        # response, if any) so the next streaming session starts clean.
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

        self._streaming = False
        print(f"Streaming stopped. Packets received: {self.packets_received}")

    def _reader_loop(self):
        ser = self.ser
        while not self._stop_event.is_set():
            try:
                if ser.in_waiting >= PACKET_SIZE:
                    data = ser.read(PACKET_SIZE)
                    if len(data) != PACKET_SIZE:
                        continue
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    row = [timestamp] + list(data)
                    if self._writer is not None:
                        self._writer.writerow(row)
                    if self._packet_buffer is not None:
                        with self._packet_buffer_lock:
                            self._packet_buffer.append(row)
                            overflow = len(self._packet_buffer) - self._packet_buffer_max
                            if overflow > 0:
                                del self._packet_buffer[:overflow]
                    self.packets_received += 1
                else:
                    # Short sleep avoids hammering the CPU between packets.
                    time.sleep(0.001)
            except serial.SerialException as e:
                print(f"Serial error in reader thread: {e}")
                break
            except Exception as e:
                print(f"Unexpected error in reader thread: {e}")
                break

    def get_latest_packets(self, n=None):
        """Return a snapshot of the in-memory packet buffer (if enabled)."""
        if self._packet_buffer is None:
            return []
        with self._packet_buffer_lock:
            if n is None:
                return list(self._packet_buffer)
            return list(self._packet_buffer[-n:])

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        if self._streaming:
            self.stop_streaming()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------- #
# PCF8575
# ---------------------------------------------------------------------- #
class PCF8575Control_:
    """Same semantics as maya.PCF8575Control_, but uses the persistent
    SerialCommunication from this module."""

    def __init__(self, serial_comm, flags=None, frequencies=None):
        self.serial_comm = serial_comm
        if flags is None:
            flags = [True] * 16
        if frequencies is None:
            frequencies = ["0"] * 16
        self.checkboxes = list(flags)
        self.frequencies = list(frequencies)
        for i in range(6, 10):
            self.checkboxes[i] = True

    def set_flags(self, flags):
        self.checkboxes = list(flags)
        for i in range(6, 10):
            self.checkboxes[i] = True

    def set_frequencies(self, frequencies):
        self.frequencies = list(frequencies)

    def send_state(self):
        state = sum(
            (self.checkboxes[i] << i)
            for i in range(16)
            if i not in range(6, 10)
        )
        frequencies = [
            self.frequencies[i] if i not in range(6, 10) else "0"
            for i in range(16)
        ]
        frequency_string = ",".join(str(f) for f in frequencies)
        self.serial_comm.send_message(f"PCF,{state},{frequency_string}")
        print(
            f"PCF8575 state and frequencies updated:\n"
            f"State: {bin(state)}\nFrequencies: {frequency_string}"
        )

    def reset(self):
        self.checkboxes = [True] * 16
        self.frequencies = ["0"] * 16
        self.send_state()


# ---------------------------------------------------------------------- #
# MFC
# ---------------------------------------------------------------------- #
class MFCControl:
    """Mass Flow Controller commands.

    ``set_flow`` and ``blink`` are safe to call while streaming. ``setup`` is
    also safe (no USB Serial response), but is normally only called before a
    streaming session. ``read_registers`` is NOT safe during streaming
    because the firmware writes ``DATA:...`` onto USB Serial.

    Valid MFC addresses are 1, 2, 3.
    """

    SETPOINT_FULL_SCALE = 32000  # matches MFC_control.set_flow in the original

    def __init__(self, serial_comm):
        self.serial_comm = serial_comm
        self.flow_rates = [0.0, 0.0, 0.0]

    def _require_not_streaming(self, op):
        if self.serial_comm.is_streaming:
            raise RuntimeError(
                f"{op} cannot be called while streaming: it would write "
                f"text onto the USB Serial stream and corrupt packets."
            )

    def blink(self, addr):
        self.serial_comm.send_message(f"MFC,{addr},0")

    def set_flow(self, addr, rate):
        """Set the flow rate for MFC ``addr`` (1-3) to ``rate`` in [0, 1].

        Safe to call while streaming.
        """
        if addr not in (1, 2, 3):
            raise ValueError(f"MFC address must be 1, 2, or 3 (got {addr})")
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1] (got {rate})")
        rate_int = int(rate * self.SETPOINT_FULL_SCALE)
        self.serial_comm.send_message(f"MFC,{addr},1,{rate_int}")
        self.flow_rates[addr - 1] = rate
        print(f"MFC {addr} flow rate updated: {rate}")

    def setup(self):
        for i in range(1, 4):
            self.serial_comm.send_message(f"MFC,{i},99")
            print(f"MFC {i} setup command sent.")
            # Small gap so the Teensy finishes each Modbus transaction before
            # the next one is queued. (Firmware has its own 10 ms delays.)
            time.sleep(0.1)

    def read_registers(self, addr, register, length=1, timeout=2.0):
        """Read Modbus holding registers. MUST NOT be called while streaming."""
        self._require_not_streaming("read_registers")
        if isinstance(register, str) and register.startswith("0x"):
            register = int(register, 16)

        self.serial_comm.send_message(f"MFC,{addr},9,{register},{length}", verbose=False)
        prev_timeout = self.serial_comm.ser.timeout
        self.serial_comm.ser.timeout = timeout
        try:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                raw = self.serial_comm.ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if line.startswith("DATA:"):
                    values = list(map(int, line[5:].split(",")))
                    if len(values) == length:
                        return values
                    raise ValueError(
                        f"Expected {length} values, got {len(values)}: {values}"
                    )
                elif line:
                    print(f"Ignoring unexpected serial line: {line}")
            raise TimeoutError(f"No DATA response from MFC {addr}")
        finally:
            self.serial_comm.ser.timeout = prev_timeout

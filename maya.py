import serial
import threading
import csv
import time
from datetime import datetime

# Serial port configuration
PORT = "/dev/ttyACM0"  # Replace with your Teensy's serial port
BAUDRATE = 115200  # Match the baud rate used in the Teensy program
PACKET_SIZE = 128  # Size of each data packet (bytes)
DURATION = 300  # 5 minutes in seconds
WAIT_TIME = 3600  # 1 hour in seconds
TOTAL_HOURS = 5  # Number of hours to record data
FREQUENCY = 100  # Packets per second

class SerialCommunication:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate

    def send_message(self, message):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                ser.write(message.encode())
                print(f"Sent to Teensy: {message.strip()}")
        except serial.SerialException as e:
            print("Could not send data to Teensy: {e}")

    def record_data(self, filename, duration):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser, open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Data"])

                start_time = time.time()
                while time.time() - start_time < duration:
                    if ser.in_waiting >= PACKET_SIZE:
                        data = ser.read(PACKET_SIZE)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        writer.writerow([timestamp] + list(data))

                print(f"Data saved successfully to {filename}.")
        except Exception as e:
            print(f"Error during data recording: {e}")

    def stream_data(self, duration, save_csv=False, filename="test.csv"):
        try:
            command = f"MAYA,1\n"
            self.send_message(command)
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser, open(filename, mode="w", newline="") as file:
                if save_csv:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Data"])
                start_time = time.time()
                print('Streaming data...')
                while time.time() - start_time < duration:
                    if ser.in_waiting >= PACKET_SIZE:
                        data = ser.read(PACKET_SIZE)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        if save_csv:
                            writer.writerow([timestamp] + list(data))
                        yield [timestamp] + list(data)
                if save_csv:
                    print(f"Data saved successfully to {filename}.")
            command = f"MAYA,0\n"
            self.send_message(command)
            print('Done')
        except Exception as e:
            print(f"Error during data recording: {e}")


# # NEW VERSION
# class SerialCommunication:
#     def __init__(self, port, baudrate):
#         self.port = port
#         self.baudrate = baudrate
#
#         try:
#             self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
#             print(f"Connected to {self.port} at {self.baudrate} baud.")
#         except serial.SerialException as e:
#             raise RuntimeError(f"Could not open serial port: {e}")
#
#     def send_message(self, message):
#         try:
#             self.ser.write(message.encode())
#             print(f"Sent to Teensy: {message.strip()}")
#         except serial.SerialException as e:
#             print(f"Failed to send message: {e}")
#
#     def read_line(self):
#         try:
#             return self.ser.readline().decode('utf-8').strip()
#         except serial.SerialException as e:
#             print(f"Serial read error: {e}")
#             return ""
#
#     def record_data(self, filename, duration):
#         try:
#             with open(filename, mode="w", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow(["Timestamp", "Data"])
#
#                 start_time = time.time()
#                 while time.time() - start_time < duration:
#                     if self.ser.in_waiting >= PACKET_SIZE:
#                         data = self.ser.read(PACKET_SIZE)
#                         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#                         writer.writerow([timestamp] + list(data))
#
#                 print(f"Data saved successfully to {filename}.")
#         except Exception as e:
#             print(f"Error during data recording: {e}")
#
#     def stream_data(self, duration, save_csv=False, filename="test.csv"):
#         try:
#             self.send_message("MAYA,1\n")
#             time.sleep(0.01)
#             self.ser.reset_input_buffer()
#
#             with open(filename, mode="w", newline="") as file:
#                 if save_csv:
#                     writer = csv.writer(file)
#                     writer.writerow(["Timestamp", "Data"])
#                 start_time = time.time()
#                 print('Streaming data...')
#                 while time.time() - start_time < duration:
#                     if self.ser.in_waiting >= PACKET_SIZE:
#                         data = self.ser.read(PACKET_SIZE)
#                         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
#                         if save_csv:
#                             writer.writerow([timestamp] + list(data))
#                         yield [timestamp] + list(data)
#                 if save_csv:
#                     print(f"Data saved successfully to {filename}.")
#             self.send_message("MAYA,0\n")
#             print('Done')
#         except Exception as e:
#             print(f"Error during streaming: {e}")
#
#     def close(self):
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#             print("Serial port closed.")


class PCF8575Control_:
    def __init__(self, serial_comm, flags=[True for i in range(16)], frequencies=["0" for i in range(16)]):
        self.serial_comm = serial_comm
        self.checkboxes = flags
        self.frequencies = frequencies
        # Set P6, P7, P8, P9 to always be ON
        for i in range(6, 10):
            self.checkboxes[i] = True
            
    def set_flags(self, flags):
        self.checkboxes = flags

        # Set P6, P7, P8, P9 to always be ON
        for i in range(6, 10):
            self.checkboxes[i] = True
        
    def set_frequencies(self, frequencies):
        self.frequencies = frequencies

    def send_state(self):
        try:
            # Collect state and frequency information
            state = sum((self.checkboxes[i] << i) for i in range(16) if i not in range(6, 10))
            frequencies = [self.frequencies[i] if i not in range(6, 10) else "0" for i in range(16)]
            frequency_string = ",".join(frequencies)

            # Format and send the command
            command = f"PCF,{state},{frequency_string}\n"
            self.serial_comm.send_message(command)
            print(f"PCF8575 state and frequencies updated:\nState: {bin(state)}\nFrequencies: {frequency_string}")
        except Exception as e:
            print(f"Failed to send PCF8575 state and frequencies: {e}")

    def reset(self):
        self.checkboxes = [True for i in range(16)]
        self.frequencies = ["0" for i in range(16)]
        self.send_state()


class MFCControl:
    def __init__(self, serial_comm):
        self.serial_comm = serial_comm
        self.flow_rates = [0 for _ in range(3)]

    def blink(self, addr):
        try:
            command = f'MFC,{addr},0\n'
            self.serial_comm.send_message(command)
        except Exception as e:
            print(f"Failed to blink MFC LED: {e}")

    def set_flow(self, addr, rate):
        try:
            rate_int = int(rate * 32000)
            command = f"MFC,{addr},1,{rate_int}\n"
            self.serial_comm.send_message(command)
            print(f"MFC {addr} flow rate updated: {rate}")
            self.flow_rates[addr - 1] = rate
        except Exception as e:
            print(f"Failed to send MFC flow rates: {e}")

    def setup(self):
        for i in range(1, 4):
            try:
                command = f"MFC,{i},99\n"
                self.serial_comm.send_message(command)
                print(f"MFC {i} setup command sent.")
            except Exception as e:
                print(f"Failed to setup MFC {i}: {e}")

    def read_registers(self, addr, register, length=1, timeout=2.0):
        try:
            # Ensure register is converted to integer if it's in hex string format
            if isinstance(register, str) and register.startswith("0x"):
                register = int(register, 16)

            command = f"MFC,{addr},9,{register},{length}\n"
            self.serial_comm.send_message(command)

            # Temporarily set the serial timeout for reading response
            prev_timeout = self.serial_comm.ser.timeout
            self.serial_comm.ser.timeout = timeout

            timer = time.time()
            while (time.time() - timer < 10.):
                line = self.serial_comm.readline()
                if line.startswith("DATA:"):
                    data_str = line[5:]
                    values = list(map(int, data_str.split(',')))
                    if len(values) == length:
                        self.serial_comm.ser.timeout = prev_timeout  # Restore timeout
                        return values
                    else:
                        raise ValueError(f"Expected {length} values, got {len(values)}: {values}")
                elif line:
                    print(f"Ignoring unexpected serial line: {line}")

        except Exception as e:
            print(f"Failed to read registers from MFC {addr}: {e}")
            return None




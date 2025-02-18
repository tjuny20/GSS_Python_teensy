import tkinter as tk
from tkinter import messagebox
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
        self.is_running = False

    def send_message(self, message):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                ser.write(message.encode())
                print(f"Sent to Teensy: {message.strip()}")
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Could not send data to Teensy: {e}")

    def record_data(self, filename, duration):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser, open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Data"])

                start_time = time.time()
                while time.time() - start_time < duration and self.is_running:
                    if ser.in_waiting >= PACKET_SIZE:
                        data = ser.read(PACKET_SIZE)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        writer.writerow([timestamp] + list(data))

                print(f"Data saved successfully to {filename}.")
        except Exception as e:
            print(f"Error during data recording: {e}")

    def record_live(self, duration, save_csv=False, filename="test.csv"):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser, open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Data"])

                start_time = time.time()
                while time.time() - start_time < duration and self.is_running:
                    if ser.in_waiting >= PACKET_SIZE:
                        data = ser.read(PACKET_SIZE)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        if save_csv:
                            writer.writerow([timestamp] + list(data))
                        yield timestamp, list(data)
                if save_csv:
                    print(f"Data saved successfully to {filename}.")
        except Exception as e:
            print(f"Error during data recording: {e}")

class PCF8575Control:
    def __init__(self, parent):
        self.parent = parent
        # Create BooleanVars for the checkboxes and frequency entries for each bit
        self.checkboxes = [tk.BooleanVar(self.parent) for _ in range(16)]
        self.frequencies = [tk.StringVar(self.parent, value="0") for _ in range(16)]  # Default frequency is 0 Hz

        # Set P6, P7, P8, P9 to always be 0
        for i in range(6, 10):
            self.checkboxes[i].set(0)
            self.checkboxes[i].trace_add("write", lambda *args, idx=i: self.checkboxes[idx].set(0))

    def send_state(self):
        try:
            # Collect state and frequency information
            state = sum((self.checkboxes[i].get() << i) for i in range(16) if i not in range(6, 10))
            frequencies = [self.frequencies[i].get() if i not in range(6, 10) else "0" for i in range(16)]
            frequency_string = ",".join(frequencies)

            # Format and send the command
            command = f"PCF,{state},{frequency_string}\n"
            serial_comm.send_message(command)
            messagebox.showinfo("Success", f"PCF8575 state and frequencies updated:\nState: {bin(state)}\nFrequencies: {frequency_string}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send PCF8575 state and frequencies: {e}")

# Add the PCF8575 control frame to the GUI
def create_pcf8575_gui(root, pcf_control):
    pcf_frame = tk.LabelFrame(root, text="PCF8575 Control", padx=10, pady=10, font=("Arial", 12))
    pcf_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

    # Place checkboxes and frequency fields in two vertical columns
    for i in range(16):
        if 6 <= i <= 9:  # Skip P6, P7, P8, P9
            continue
        ii = i
        if i > 9:
            ii = ii + 2
        col = 0 if i < 6 else 1  # First column for P0-P5, second for P10-P15
        row = i if i < 6 else i - 10

        checkbox = tk.Checkbutton(pcf_frame, text=f"P{ii}", variable=pcf_control.checkboxes[i])
        checkbox.grid(row=row, column=col * 3, padx=5, pady=5, sticky="w")

        tk.Label(pcf_frame, text="Freq (Hz):", font=("Arial", 10)).grid(row=row, column=col * 3 + 1, sticky="e", padx=5)
        frequency_entry = tk.Entry(pcf_frame, textvariable=pcf_control.frequencies[i], width=8)
        frequency_entry.grid(row=row, column=col * 3 + 2, padx=5, pady=5)

    # Button to send data to Teensy
    sendpcf_button = tk.Button(root, text="Send to PCF8575", command=pcf_control.send_state, bg="cyan", fg="black", font=("Arial", 12))
    sendpcf_button.grid(row=9, column=0, columnspan=2, pady=10)

# Update the main GUI function
def create_gui():
    root = tk.Tk()
    root.title("Sensor data collector")

    # Input fields and labels for DAC control
    def add_labeled_entry(parent, label_text, default_value, row):
        tk.Label(parent, text=label_text, font=("Arial", 12)).grid(row=row, column=0, padx=10, pady=10)
        entry = tk.Entry(parent, font=("Arial", 12))
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, padx=10, pady=10)
        return entry

    dac1_entry = add_labeled_entry(root, "U16-TGS-3830 (mV):", "800", 0)
    dac2_entry = add_labeled_entry(root, "U25-TGS-3870 (mV):", "200", 1)
    frequency_entry = add_labeled_entry(root, "Packages / sec:", str(FREQUENCY), 2)

    def send_to_teensy():
        dac1_value, dac2_value, freq_value = dac1_entry.get(), dac2_entry.get(), frequency_entry.get()
        if not all(x.isdigit() for x in [dac1_value, dac2_value, freq_value]):
            messagebox.showerror("Input Error", "Please enter valid numeric values for DACs and frequency.")
            return

        message = f"{dac1_value},{dac2_value},{freq_value}\n"
        serial_comm.send_message(message)
        messagebox.showinfo("Success", f"Values sent to Teensy:\nDAC1: {dac1_value} mV\nDAC2: {dac2_value} mV\nFrequency: {freq_value} Hz")

    send_button = tk.Button(root, text="Send", command=send_to_teensy, bg="green", fg="white", font=("Arial", 12))
    send_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Parameter input fields
    duration_entry = add_labeled_entry(root, "Duration (s):", str(DURATION), 4)
    wait_time_entry = add_labeled_entry(root, "Wait Time (s):", str(WAIT_TIME), 5)
    total_hours_entry = add_labeled_entry(root, "Total Hours:", str(TOTAL_HOURS), 6)

    def update_parameters():
        try:
            global DURATION, WAIT_TIME, TOTAL_HOURS, FREQUENCY
            DURATION = int(duration_entry.get().strip())
            WAIT_TIME = int(wait_time_entry.get().strip())
            TOTAL_HOURS = int(total_hours_entry.get().strip())
            FREQUENCY = int(frequency_entry.get().strip())
            messagebox.showinfo("Success", "Parameters updated successfully.")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")

    update_button = tk.Button(root, text="Update Parameters", command=update_parameters, bg="orange", fg="black", font=("Arial", 12))
    update_button.grid(row=7, column=0, columnspan=2, pady=10)

    # PCF8575 control GUI
    pcf_control = PCF8575Control(root)
    create_pcf8575_gui(root, pcf_control)

    # Start and Stop buttons for data recording
    def start_recording():
        if serial_comm.is_running:
            messagebox.showinfo("Already Running", "Data recording is already running.")
            return
        serial_comm.is_running = True
        threading.Thread(target=record_data_thread, daemon=True).start()

    def stop_recording():
        serial_comm.is_running = False

    start_button = tk.Button(root, text="Start Recording", command=start_recording, bg="blue", fg="white", font=("Arial", 12))
    start_button.grid(row=10, column=0, columnspan=2, pady=10)

    stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, bg="red", fg="white", font=("Arial", 12))
    stop_button.grid(row=11, column=0, columnspan=2, pady=10)

    # Exit button
    exit_button = tk.Button(root, text="Exit", command=root.destroy, bg="gray", fg="white", font=("Arial", 12))
    exit_button.grid(row=12, column=0, columnspan=2, pady=10)

    root.mainloop()

def record_data_thread():
    hours_elapsed = 0
    while serial_comm.is_running and hours_elapsed < TOTAL_HOURS:
        file_name = datetime.now().strftime("teensy_data_%Y%m%d_%H%M%S.csv")
        print(f"Starting to save data to {file_name}...")
        serial_comm.record_data(file_name, DURATION)

        print("Waiting for the next hour to start (55 minutes)...")
        time.sleep(WAIT_TIME - DURATION)
        hours_elapsed += 1

if __name__ == "__main__":
    serial_comm = SerialCommunication(PORT, BAUDRATE)
    create_gui()

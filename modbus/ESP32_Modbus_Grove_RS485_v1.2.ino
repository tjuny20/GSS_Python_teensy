/*
  ESP32-Modbus-Grove_RS485

  Basic Modbus interfacing functions

  Initial release:    3 Juli 2024
  Software version:   1.2
  by Vincent Jassies

*/

/* --------- Modbus intitialization --------- */
#include <ModbusRTUMaster.h>
ModbusRTUMaster modbus(Serial1);      // serial port used for modbus transceiver
#define MB_BAUD_RATE    19200         // Modbus baud rate
#define RX_PIN          0            // Pin used for the Modbus transceiver RX
#define TX_PIN          1            // Pin used for the Modbus transceiver TX
uint16_t read_buffer    [2];  

/* --------- Flexiflow Registers --------- */
// found at: https://www.bronkhorst.com/getmedia/bb0e02ab-2429-4638-b751-7186bd7178fb/917035-Manual-Modbus-slave-interface.pdf
#define REG_WINK          0x0         // W  - Modbus register - PDU ADDRESS 
#define REG_FMEASURE      0xA100      // R  - Modbus register - PDU ADDRESS 
#define REG_COUNTER_VAL   0xE808      // RW - Modbus register - PDU ADDRESS 
#define REG_FSETPOINT     0xA118      // RW - Modbus register - PDU ADDRESS 
#define REG_TEMPERATURE   0xA138      // R  - Modbus register - PDU ADDRESS 

/* --------- General use variables --------- */
volatile int prev_millis =    0;      // Save the previous time 
volatile int interval_time =  3000;   // Time in milliseconds


// Used to construct a floating point value from 2 unsigned integers
float f_2uint_float(uint16_t uint1, uint16_t uint2) {    
  union f_2uint {
      float f;
      uint16_t i[2];
  } f_number;

  f_number.i[0] = uint2;
  f_number.i[1] = uint1;

  return f_number.f;
}


// Main init
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize Serial communication at X baud rate 
  Serial.begin(115200);   
  Serial.println("Serial started!");
  
  // Initialize Modbus communication at X baud rate with 8 data bits, even parity, and 1 stop bit, using specified RX and TX pins
  // modbus.begin(MB_BAUD_RATE, SERIAL_8E1, RX_PIN, TX_PIN);
  modbus.begin(MB_BAUD_RATE, SERIAL_8E1);
  Serial.println("ModbusRTU started!");
}


// Main loop
void loop() {
  // Check if the time since the last action exceeds the specified interval
  if(millis() - prev_millis >= interval_time){
    prev_millis = millis();

    // Toggle onboard LED to indicate running mode
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));

    // Write a single holding register to the Modbus slave device
    // SLAVE_ADDRESS - The address of the Modbus slave device to communicate with
    // REG_WINK      - The starting address of the register to write to
    // SET_VALUE     - The value to store in the address
    modbus.writeSingleHoldingRegister(1, REG_WINK, 0x3100);

    // Send a new Setpoint to modbus slave
    // modbus_setSetpoint(1, 100);

    // Get Fmeasure from modbus slave
    modbus_getFmeasure(1);

    // Get Temperature from modbus slave
    modbus_getTemperature(1);
  }
}

// void modbus_setSetpoint(1, 100){
  
// }



// Get Fmeasure from modbus server
float modbus_getFmeasure(uint address){
    uint16_t temp_buffer[2];
    modbus.readHoldingRegisters(address, REG_FMEASURE, temp_buffer, 2);
    float reading = f_2uint_float(temp_buffer[0], temp_buffer[1]);
    
    Serial.print("Address: "); Serial.print(address);
    Serial.print(" - Fmeasure: "); Serial.println(reading);
    
    return reading;
}

// Get temperature from modbus server
float modbus_getTemperature(uint address){
    uint16_t temp_buffer[2];
    modbus.readHoldingRegisters(address, REG_TEMPERATURE, temp_buffer, 2);
    float reading = f_2uint_float(temp_buffer[0], temp_buffer[1]);
    
    Serial.print("Address: "); Serial.print(address);
    Serial.print(" - Temperature: "); Serial.println(reading);
    
    return reading;
}
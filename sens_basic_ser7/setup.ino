void setup() {
  
  Serial.begin(115200);  // USB Serial communication

  memset(massflow1.buffer, 0, PACKET_SIZE);         //clear massflow1.buffer
  memset(massflow2.buffer, 0, PACKET_SIZE);         //clear massflow2.buffer
  memset(massflow3.buffer, 0, PACKET_SIZE);         //clear massflow3.buffer
  memset(ctrl.buffer, 0, PACKET_SIZE);              //clear ctrl.buffer
  memset(c.buffer, 0, PACKET_SIZE);                 //clear c.buffer
  memset(sens.buffer, 0, PACKET_SIZE);              //clear sens.buffer

  //add ID to the HID arrays
  massflow1.buffer[PACKET_SIZE-1] = 1;
  massflow2.buffer[PACKET_SIZE-1] = 2;
  massflow2.buffer[PACKET_SIZE-1] = 3;
  ctrl.buffer[PACKET_SIZE-1] = 4;
  sens.buffer[PACKET_SIZE-1] = 5;

//   readSensTimer.begin(readSensData, 1000000 / DATA_RATE_HZ);  // run every 10000 microseconds
  Wire.begin();

  // Set I2C clock to 400kHz for fast communication. CHECK IF POSSIBLE TO USE 3.4MHz
  Wire.setClock(400000);

    // Read the WHO_AM_I register of the BME680 this is a good test of communication
  byte f = readByte(BME680_ADDRESS, BME680_ID);  // Read WHO_AM_I register for BME680
  
  delay(1000); 

  if(f == 0x61) {
    
    bme = true;
    
    writeByte(BME680_ADDRESS, BME680_RESET, 0xB6); // reset BME680 before initialization
    delay(100);

    BME680TPHInit(); // Initialize BME680 Temperature, Pressure, Humidity sensors

    // Configure the gas sensor
    gasWait[0] = GWaitMult | 0x59;  // define gas wait time for heat set point 0x59 == 100 ms
    resHeat[0] = BME680_TT(200);    // define temperature set point in degrees Celsius for resistance set point 0
    BME680GasInit();

  }else {
    bme = false;
    Serial.println(" BME680 not functioning!");
  }
  
  delay(1000);  

  writePCF8575(0xffff);  // Make all pins LOW
    writeMCP4725(MCP4725_1_ADDR, (u16_mV + interpolate(u16_mV))*4095/10000);
    writeMCP4725(MCP4725_2_ADDR, (u25_mV + interpolate(u25_mV))*4095/10000);

  // Initialize Modbus communication at X baud rate with 8 data bits, even parity, and 1 stop bit, using specified RX and TX pins
  modbus.begin(MB_BAUD_RATE, SERIAL_8E1);
  Serial.println("ModbusRTU started!");

}

void setup() {
  
  Serial.begin(115200);  // USB Serial communication

  delay(100);
  if (!BME680.begin(I2C_STANDARD_MODE)){bme = false;} else {bme = true;
  
  BME680.setOversampling(TemperatureSensor, Oversample16);  // Use enumerated type values
  BME680.setOversampling(HumiditySensor, Oversample16);     // Use enumerated type values
  BME680.setOversampling(PressureSensor, Oversample16);     // Use enumerated type values
  BME680.setIIRFilter(IIR4);                                // Use enumerated type values
  BME680.setGas(320, 150);                                  // 320c for 150 milliseconds
  }

  memset(massflow1.buffer, 0, PACKET_SIZE); //clear massflow1.buffer
  memset(massflow2.buffer, 0, PACKET_SIZE); //clear massflow2.buffer
  memset(massflow3.buffer, 0, PACKET_SIZE); //clear massflow3.buffer
  memset(ctrl.buffer, 0, PACKET_SIZE);             //clear ctrl.buffer
  memset(c.buffer, 0, PACKET_SIZE);                //clear c.buffer
  memset(sens.buffer, 0, PACKET_SIZE);             //clear sens.buffer

  //add ID to the HID arrays
  massflow1.buffer[PACKET_SIZE-1] = 1;
  massflow2.buffer[PACKET_SIZE-1] = 2;
  massflow2.buffer[PACKET_SIZE-1] = 3;
  ctrl.buffer[PACKET_SIZE-1] = 4;
  sens.buffer[PACKET_SIZE-1] = 5;

  readSensTimer.begin(readSensData, 1000000 / DATA_RATE_HZ);  // run every 10000 microseconds
  Wire.begin();

  // Set I2C clock to 1 MHz for high-speed communication
  Wire.setClock(400000);


  writePCF8575(0x0000);  // Make all pins LOW
  writeMCP4725(MCP4725_1_ADDR, 522+u16_mV/1.55);
  writeMCP4725(MCP4725_2_ADDR, 522+u25_mV/1.55);
}

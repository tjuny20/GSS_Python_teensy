void readSensData(){
  for (byte i = 0; i <= 16; i++) {
    if (sensor_id[i][1]<99){
      splitu2.result = readADS1115(sensor_id[i][0], sensor_id[i][1]);
      sens.bufint[i] = splitu2.result;
    } else {
      if (bme == true){
        BME680.getSensorData(temp, humidity, pressure, gas);  // Get BME680 readings
      } else {
        temp = 10;
        humidity = 20;
        pressure = 30;
        gas = 40;
      }
      sens.bufint32[8] = temp;
      sens.bufint32[9] = humidity;
      sens.bufint32[10] = pressure;
      sens.bufint32[11] = gas;
    }
  } 

  Serial.write(sens.buffer, PACKET_SIZE);
}


float readADS1115(int ad, byte ch){
  
  const byte chan[4]={0xC2, 0xD2, 0xE2, 0xF2}; //11000010 11010010 11100010 11110010 ads1115 channels 0,1,2,3

  unsigned int data[2];

  // Start I2C Transmission
  Wire.beginTransmission(ad);
  // Select configuration register
  Wire.write(0x01);
  // AINP = AIN+ and AINN = GND, +/- 2.048V
  Wire.write(chan[ch]);
  // Continuous conversion mode, 128 SPS
  Wire.write(0x83);
  // Stop I2C Transmission
  Wire.endTransmission();
  //delay(10);

  // Start I2C Transmission
  Wire.beginTransmission(ad);
  // Select data register
  Wire.write(0x00);
  // Stop I2C Transmission
  Wire.endTransmission();

  // Request 2 bytes of data
  Wire.requestFrom(ad, 2);

  // Read 2 bytes of data
  // raw_adc msb, raw_adc lsb
  if (Wire.available() == 2)
  {
    data[0] = Wire.read();
    data[1] = Wire.read();
  } else { data[0]=0; data[1]=0; }

  // Convert the data
  int raw_adc = (data[0] * 256.0) + data[1];
  if (raw_adc > 32767)
  {
    raw_adc -= 65535;
  }

  return raw_adc;  
}


void parseInput(String input) {
  input.trim();
  if (input.length() == 0) return;

  if (input.startsWith("PCF")) {
    // Skip "PCF," prefix and split the input into state and frequency strings
    int firstComma = input.indexOf(',');
    if (firstComma == -1) {
      Serial.println("Invalid PCF format. Expected: PCF,<state>,<frequencies>");
      return;
    }

    // Extract frequencies (optional, if provided)
    String frequenciesStr = input.substring(firstComma + 1);
    int currentIndex = 0;
    for (int i = 0; i < 17 && currentIndex >= 0; i++) {
      int nextComma = frequenciesStr.indexOf(',', currentIndex);
      String freqStr = (nextComma == -1) ? frequenciesStr.substring(currentIndex) : frequenciesStr.substring(currentIndex, nextComma);
      parse[i] = freqStr.toInt(); // Store frequencies in PCFwindow[16] to PCFwindow[17]
      currentIndex = (nextComma == -1) ? -1 : nextComma + 1;
    }

    //Serial.println();
    for (int i = 0; i < 16; i++) {
      if (parse[i+1] > 0) { PCFwindow[i] = 1000/parse[i+1]/2; } else { PCFwindow[i] = 0;}
      //Serial.print(parse[i+1]); Serial.print(" Hz: "); Serial.print(PCFwindow[i]); Serial.println(" ms: ");
    }

    String stateStr = input.substring(4, firstComma); // Extract state
    newState = parse[0];

    // Update PCFon[0] to PCFon[15] based on newState bits
    for (int i = 0; i < 16; i++) {
      PCFon[i] = (newState >> i) & 0x01; // Extract each bit from the 16-bit state
      //Serial.print("PCF on: "); Serial.println(PCFon[i]);
    }

    writePCF8575(newState);
    //Serial.println(newState, BIN);
  } else {
    // Parse the input into DAC1, DAC2, and frequency values
    int firstComma = input.indexOf(',');
    int secondComma = input.indexOf(',', firstComma + 1);

    if (firstComma == -1 || secondComma == -1) {
      Serial.println("Invalid input format. Expected: DAC1,DAC2,FREQ");
      return;
    }

    String dac1ValueStr = input.substring(0, firstComma);
    String dac2ValueStr = input.substring(firstComma + 1, secondComma);
    String frequencyStr = input.substring(secondComma + 1);

    u16_mV = dac1ValueStr.toInt();
    u25_mV = dac2ValueStr.toInt();
    DATA_RATE_HZ = frequencyStr.toInt();

    // Update the interval based on the new frequency
    if (DATA_RATE_HZ > 0) {
      readSensTimer.end(); 
      readSensTimer.begin(readSensData, 1000000 / DATA_RATE_HZ);  // microseconds
    }

    // Set DAC outputs
    writeMCP4725(MCP4725_1_ADDR, 522 + u16_mV / 1.55);
    writeMCP4725(MCP4725_2_ADDR, 522 + u25_mV / 1.55);
  }
}



void writeMCP4725(byte ad, uint16_t value){
  if (value > 4095) value = 4095;  // Limit value to 12-bit max (4095)

  Wire.beginTransmission(ad);

  // Command to write data to the DAC (normal mode)
  Wire.write(0x40);  // Fast mode command, upper 4 bits of configuration (0100xxxx)

  // Send the upper 8 bits (MSB)
  Wire.write(value >> 4);  // Shift the 12-bit value right by 4 bits to get the upper 8 bits

  // Send the lower 4 bits (LSB), shifted into the upper nibble of a byte
  Wire.write((value & 0x0F) << 4);  // Mask the lower 4 bits and shift them left into upper nibble

  Wire.endTransmission();  // End the transmission  
}


// Function to write data to the PCF8575
void writePCF8575(uint16_t data) {
  Wire.beginTransmission(PCF8575_ADDRESS);
  Wire.write(lowByte(data));  // Send the lower 8 bits
  Wire.write(highByte(data)); // Send the upper 8 bits
  Wire.endTransmission();
}


// Function to read data from the PCF8575
uint16_t readPCF8575() {
  Wire.requestFrom(PCF8575_ADDRESS, 2);  // Request 2 bytes from the PCF8575
  
  if (Wire.available()) {
    uint8_t lowByteData = Wire.read();   // Read the lower 8 bits
    uint8_t highByteData = Wire.read();  // Read the upper 8 bits
    
    // Combine the two bytes into a single 16-bit value
    return ((uint16_t)highByteData << 8) | lowByteData;
  }
  
  return 0xFFFF;  // If no data is available, return all high (default idle state)
}

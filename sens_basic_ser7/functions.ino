void readSensData(){
  for (byte i = 0; i <= 16; i++) {
    if (sensor_id[i][1]<99){
      splitu2.result = readADS1115(sensor_id[i][0], sensor_id[i][1]);
      sens.bufint[i] = splitu2.result;
    } else {
      if (bme == true){
        if(cnt==0){
          writeByte(BME680_ADDRESS, BME680_CTRL_MEAS, Tosr << 5 | Posr << 2 | Mode);
          rawTemp =   readBME680Temperature();
          temperature_C = (float) BME680_compensate_T(rawTemp)/100.;
          rawPress =  readBME680Pressure();
          pressure = (float) BME680_compensate_P(rawPress)/100.; // Pressure in mbar
          rawHumidity =   readBME680Humidity();
          humidity = (float) BME680_compensate_H(rawHumidity)/1024.;
          rawGasResistance = readBME680GasResistance();
          resistance = (float) BME680_compensate_Gas(rawGasResistance);
        }
      } else {
        temperature_C = 10;
        humidity = 20;
        pressure = 30;
        resistance = 40;
      }
      cnt++;if(cnt>10)cnt=0;
      sens.bufint32[8] = (uint32_t)  temperature_C*100;
      //Serial.print(temperature_C);Serial.print(" C; ");
      sens.bufint32[9] = (uint32_t)  humidity*100;
      //Serial.print(humidity);Serial.print(" %RH; ");
      sens.bufint32[10] = (uint32_t)  pressure*100;
      //Serial.print(pressure);Serial.print(" mbar; ");
      sens.bufint32[11] = (uint32_t)  resistance*100;
      //Serial.print(resistance);Serial.println(" Ohm; ");
    }
  } 

  Serial.write(sens.buffer, PACKET_SIZE);
}


int readADS1115(int ad, byte ch){
//float readADS1115(int ad, byte ch){
  
  const byte chan[4]={0x40, 0x50, 0x60, 0x70}; //01000000 01010000 01100000 01110000 ads1115 channels 0,1,2,3

  unsigned int data[2];

  // Start I2C Transmission
  Wire.beginTransmission(ad);
  // Select configuration register
  Wire.write(0x01);
  // AINP = AIN+ and AINN = GND, +/- 2.048V
  Wire.write(chan[ch]);
  // Continuous conversion mode, 128 SPS, 10000011
  Wire.write(0x83);
  // Stop I2C Transmission
  Wire.endTransmission();
  delay(20);

  // Start I2C Transmission
  Wire.beginTransmission(ad);
  // Select data register
  Wire.write(0x00);
  // Stop I2C Transmission
  Wire.endTransmission();
  delay(20);

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
//   if (raw_adc > 32767)
//   {
//     raw_adc -= 65535;
//   }

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

  } else if (input.startsWith("MFC")) {
      // Skip "MFC," prefix and check the order given
      int firstComma = input.indexOf(',');
      if (firstComma == -1) {
        Serial.println("Invalid MFC format. Expected: MFC,<order>,*args");
        return;
      }
      int order = input.substring(firstComma, firstComma + 1).toInt();
//       digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
//       modbus.writeSingleHoldingRegister(1, REG_WINK, 0x3100);

      if (order == 0) {
      // Toggle onboard LED to indicate running mode
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));

      // Write a single holding register to the Modbus slave device
      // SLAVE_ADDRESS - The address of the Modbus slave device to communicate with
      // REG_WINK      - The starting address of the register to write to
      // SET_VALUE     - The value to store in the address
      modbus.writeSingleHoldingRegister(1, REG_WINK, 0x3100);

      } else if (order==1) {
        // Skip "MFC,<order>," prefix and split the input into state and frequency strings
        int secondComma = input.indexOf(',', firstComma + 1);
        if (secondComma == -1) {
          Serial.println("Invalid MFC format. Expected: MFC,<order>,<flow rate>");
          return;
        }

        // Extract flow rate
        int flowRate = input.substring(secondComma).toInt();
        // Write the flow rate to the Modbus slave device
        modbus.writeSingleHoldingRegister(1, REG_FSETPOINT, flowRate);

      } else {
        Serial.println("Invalid MFC order. Expected: 0 (blink) or 1 (set flow rate)");
        return;
      }



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


// } else if (input.startsWith("MFC")) {
//     // Skip "MFC," prefix and check the order given
//     int firstComma = input.indexOf(',');
//     if (firstComma == -1) {
//       Serial.println("Invalid MFC format. Expected: MFC,<order>,*args");
//       return;
//     }
//     int order = input.substring(firstComma, firstComma + 1).toInt();
//     digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
//
// //     if (order == 0) {
// //     // Toggle onboard LED to indicate running mode
// //     digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
// //
// //     // Write a single holding register to the Modbus slave device
// //     // SLAVE_ADDRESS - The address of the Modbus slave device to communicate with
// //     // REG_WINK      - The starting address of the register to write to
// //     // SET_VALUE     - The value to store in the address
// //     modbus.writeSingleHoldingRegister(1, REG_WINK, 0x3100);
// //
// //     } else {
// //       Serial.println("Invalid MFC order. Expected: 0 (blink) or 1 (set flow rate)");
// //       return;
// //     }


// Function for linear interpolation
float interpolate(float x) {
    // Handle edge cases
    if (x <= xValues[0]) return yValues[0];  // Below range
    if (x >= xValues[numPoints - 1]) return yValues[numPoints - 1];  // Above range

    // Find the two nearest points for interpolation
    for (int i = 0; i < numPoints - 1; i++) {
        if (x >= xValues[i] && x <= xValues[i + 1]) {
            // Perform linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            float x1 = xValues[i], x2 = xValues[i + 1];
            float y1 = yValues[i], y2 = yValues[i + 1];
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
        }
    }
    return 0; // Should never reach here
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
  data = 65535 - data;
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

//===================================================================================================================
//====== Set of useful function to access acceleration. gyroscope, magnetometer, and temperature data
//===================================================================================================================

 uint32_t readBME680Temperature()
{
  uint8_t rawData[3];  // 20-bit pressure register data stored here
  readBytes(BME680_ADDRESS, BME680_FIELD_0_TEMP_MSB, 3, &rawData[0]);  
  return (uint32_t) (((uint32_t) rawData[0] << 16 | (uint32_t) rawData[1] << 8 | rawData[2]) >> 4);
}

uint32_t readBME680Pressure()
{
  uint8_t rawData[3];  // 20-bit pressure register data stored here
  readBytes(BME680_ADDRESS, BME680_FIELD_0_PRESS_MSB, 3, &rawData[0]);  
  return (uint32_t) (((uint32_t) rawData[0] << 16 | (uint32_t) rawData[1] << 8 | rawData[2]) >> 4);
}

uint16_t readBME680Humidity()
{
  uint8_t rawData[3];  // 20-bit pressure register data stored here
  readBytes(BME680_ADDRESS, BME680_FIELD_0_HUM_MSB, 2, &rawData[0]);  
  return (uint16_t) (((uint16_t) rawData[0] << 8 | rawData[1]) );
}

uint16_t readBME680GasResistance()
{
  uint8_t rawData[2];  // 10-bit gas resistance register data stored here
  readBytes(BME680_ADDRESS, BME680_FIELD_0_GAS_RL_MSB, 2, &rawData[0]);  
  //if(rawData[1] & 0x20) Serial.println("Field 0 gas data valid"); 
  return (uint16_t) (((uint16_t) rawData[0] << 2 | (0xC0 & rawData[1]) >> 6) );

}


void BME680TPHInit()
{
  // Configure the BME680 Temperature, Pressure, Humidity sensors
  // Set H oversampling rate
  writeByte(BME680_ADDRESS, BME680_CTRL_HUM, 0x07 & Hosr);
  // Set T and P oversampling rates and sensor mode
  writeByte(BME680_ADDRESS, BME680_CTRL_MEAS, Tosr << 5 | Posr << 2 | Mode);
  // Set standby time interval in normal mode and bandwidth
  writeByte(BME680_ADDRESS, BME680_CONFIG, SBy << 5 | IIRFilter << 2);
 
  // Read and store calibration data
  uint8_t calib[41] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  readBytes(BME680_ADDRESS, BME680_CALIB_ADDR_1, 25, &calib[0]);
  readBytes(BME680_ADDRESS, BME680_CALIB_ADDR_2, 16, &calib[25]);
 // temperature compensation parameters
  dig_T1 = (uint16_t)(((uint16_t) calib[34] << 8) | calib[33]);
  dig_T2 = ( int16_t)((( int16_t) calib[2] << 8) | calib[1]);
  dig_T3 = (  int8_t)             (calib[3]);
 // pressure compensation parameters
  dig_P1 = (uint16_t)(((uint16_t) calib[6] << 8) | calib[5]);
  dig_P2 = ( int16_t)((( int16_t) calib[8] << 8) | calib[7]);
  dig_P3 =  ( int8_t)             (calib[9]);
  dig_P4 = ( int16_t)((( int16_t) calib[12] << 8) | calib[11]);
  dig_P5 = ( int16_t)((( int16_t) calib[14] << 8) | calib[13]);
  dig_P6 =  ( int8_t)             (calib[16]);
  dig_P7 =  ( int8_t)             (calib[15]);  
  dig_P8 = ( int16_t)((( int16_t) calib[20] << 8) | calib[19]);
  dig_P9 = ( int16_t)((( int16_t) calib[22] << 8) | calib[21]);
  dig_P10 = (uint8_t)             (calib[23]);
// humidity compensation parameters
  dig_H1 =  (uint16_t)(((uint16_t) calib[27] << 4) | (calib[26] & 0x0F));
  dig_H2 =  (uint16_t)(((uint16_t) calib[25] << 4) | (calib[26] >> 4));
  dig_H3 =  (int8_t) calib[28];
  dig_H4 =  (int8_t) calib[29];
  dig_H5 =  (int8_t) calib[30];
  dig_H6 = (uint8_t) calib[31];
  dig_H7 =  (int8_t) calib[32];
// gas sensor compensation parameters
  dig_GH1 =  (int8_t) calib[37];
  dig_GH2 = ( int16_t)((( int16_t) calib[36] << 8) | calib[35]);
  dig_GH3 =  (int8_t) calib[38];
}

void BME680GasInit()  // Initialize BME680 gas sensor
{
  // Configure the BME680 Gas Sensor
  writeByte(BME680_ADDRESS, BME680_CTRL_GAS_1, 0x10 | (numHeatPts - 1)); // write number of heater set points
  // Set gas sampling wait time and target heater resistance
  for(uint8_t ii = 0; ii < numHeatPts; ii++) 
  {
    writeByte(BME680_ADDRESS, (BME680_GAS_WAIT_X + ii), gasWait[ii]);
    writeByte(BME680_ADDRESS, (BME680_RES_HEAT_X + ii), resHeat[ii]);
  }
  //Serial.print("CTRL_GAS_1 = 0x"); Serial.println(readByte(BME680_ADDRESS, BME680_CTRL_GAS_1), HEX);
  //Serial.print("gas wait = 0x"); Serial.println(readByte(BME680_ADDRESS, BME680_GAS_WAIT_X), HEX);
  //Serial.print("res heat = 0x"); Serial.println(readByte(BME680_ADDRESS, BME680_RES_HEAT_X));
}

// Returns register code to be written to register BME680_RES_HEAT_CTRL for a user specified target temperature TT
// where TT is the target temperature in degrees Celsius
uint8_t BME680_TT(uint16_t TT) // TT is between 200 and 400  
{
  uint8_t res_heat_x = 0;
  double var1 = 0.0, var2 = 0.0, var3 = 0.0, var4 = 0.0, var5 = 0.0;
  uint16_t par_g1 = ((uint16_t) readByte(BME680_ADDRESS, 0xEC) << 8) | readByte(BME680_ADDRESS, 0xEB);
  uint8_t  par_g2 = readByte(BME680_ADDRESS, 0xED);
  uint8_t  par_g3 = readByte(BME680_ADDRESS, 0xEE);
  uint8_t  res_heat_range = (readByte(BME680_ADDRESS, 0x02) & 0x30) >> 4;
  uint8_t res_heat_val = readByte(BME680_ADDRESS, 0x00);
  var1 = ((double) par_g1/ 16.0) + 49.0;
  var2 = (((double)par_g2 / 32768.0) * 0.0005) + 0.00235;
  var3 = (double)par_g3 / 1024.0;
  var4 = var1 * (1.0 + (var2 * (double)TT));
  var5 = var4 + (var3 * 25.0); // use 25 C as ambient temperature
  res_heat_x = (uint8_t)(((var5 * (4.0/(4.0 * (double)res_heat_range))) - 25.0) * 3.4 / ((res_heat_val * 0.002) + 1));
  return res_heat_x;
  }

  
  // Compensate Raw Gas ADC values to obtain resistance
float BME680_compensate_Gas(uint16_t gas_adc)
{
  uint8_t gasRange = readByte(BME680_ADDRESS, BME680_FIELD_0_GAS_RL_LSB) & 0x0F;
  //Serial.print("gas range = "); Serial.println(gasRange);
  double var1 = 0, gas_switch_error = 1.0;
  var1 =  (1340.0 + 5.0 * gas_switch_error) * const_array1[gasRange];
  float gas_res = var1 * const_array2[gasRange] / (gas_adc - 512.0 + var1);
  return gas_res;
}

// Returns temperature in DegC, resolution is 0.01 DegC. Output value of
// “5123” equals 51.23 DegC.
int32_t BME680_compensate_T(uint32_t adc_T)
{
  int32_t var1 = 0, var2 = 0, var3 = 0, T = 0;
  var1 = ((int32_t) adc_T >> 3) - ((int32_t)dig_T1 << 1); 
  var2 = (var1 * (int32_t)dig_T2) >> 11;
  var3 = ((((var1 >> 1) * (var1 >> 1)) >> 12) * ((int32_t) dig_T3 << 4)) >> 14;
  t_fine = var2 + var3;
  T = (t_fine * 5 + 128) >> 8;
  return T;
}

// Returns the value in Pascal(Pa)
// Output value of "96386" equals 96386 Pa =
//  963.86 hPa = 963.86 millibar
int32_t BME680_compensate_P(uint32_t adc_P)
{
  int32_t var1 = 0, var2 = 0, var3 = 0, var4 = 0, P = 0;
  var1 = (((int32_t) t_fine) >> 1) - 64000;
  var2 = ((((var1 >> 2) * (var1 >> 2)) >> 11) * (int32_t) dig_P6) >> 2;
  var2 = var2 + ((var1 * (int32_t)dig_P5) << 1);
  var2 = (var2 >> 2) + ((int32_t) dig_P4 << 16);
  var1 = (((((var1 >> 2) * (var1 >> 2)) >> 13) * ((int32_t) dig_P3 << 5)) >> 3) + (((int32_t) dig_P2 * var1) >> 1);
  var1 = var1 >> 18;
  var1 = ((32768 + var1) * (int32_t) dig_P1) >> 15;
  P = 1048576 - adc_P;
  P = (int32_t)((P - (var2 >> 12)) * ((uint32_t)3125));
  var4 = (1 << 31);
  
  if(P >= var4)
    P = (( P / (uint32_t) var1) << 1);
  else
    P = ((P << 1) / (uint32_t) var1);
    
  var1 = ((int32_t) dig_P9 * (int32_t) (((P >> 3) * (P >> 3)) >> 13)) >> 12;
  var2 = ((int32_t)(P >> 2) * (int32_t) dig_P8) >> 13;
  var3 = ((int32_t)(P >> 8) * (int32_t)(P >> 8) * (int32_t)(P >> 8) * (int32_t)dig_P10) >> 17;
  P = (int32_t)(P) + ((var1 + var2 + var3 + ((int32_t)dig_P7 << 7)) >> 4);
  
  return P;
}

// Returns humidity in %RH as unsigned 32 bit integer in Q22.10 format (22integer and 10fractional bits).
// Output value of “47445”represents 47445/1024 = 46.333%RH
int32_t BME680_compensate_H(uint32_t adc_H)
{
  int32_t var1 = 0, var2 = 0, var3 = 0, var4 = 0, var5 = 0, var6 = 0, H = 0, T = 0;

  T = (((int32_t) t_fine * 5) + 128) >> 8;
  var1 = (int32_t) adc_H  - ((int32_t) ((int32_t)dig_H1 << 4)) - (((T * (int32_t) dig_H3) / ((int32_t)100)) >> 1);
  var2 = ((int32_t)dig_H2 * (((T * (int32_t)dig_H4) / 
         ((int32_t)100)) + (((T * ((T * (int32_t)dig_H5) / 
         ((int32_t)100))) >> 6) / ((int32_t)100)) + (int32_t)(1 << 14))) >> 10;
  var3 = var1 * var2;
  var4 = ((((int32_t)dig_H6) << 7) + ((T * (int32_t) dig_H7) / ((int32_t)100))) >> 4;
  var5 = ((var3 >> 14) * (var3 >> 14)) >> 10;
  var6 = (var4 * var5) >> 1;

  H = (var3 + var6) >> 12;

  if (H > 102400) H = 102400; // check for over- and under-flow
  else if(H < 0) H = 0;

  return H;
}


// I2C read/write functions for the BME680 sensors

        void writeByte(uint8_t address, uint8_t subAddress, uint8_t data)
{
  Wire.beginTransmission(address);  // Initialize the Tx buffer
  Wire.write(subAddress);           // Put slave register address in Tx buffer
  Wire.write(data);                 // Put data in Tx buffer
  Wire.endTransmission();           // Send the Tx buffer
}

        uint8_t readByte(uint8_t address, uint8_t subAddress)
{
  uint8_t data; // `data` will store the register data   
  Wire.beginTransmission(address);         // Initialize the Tx buffer
  Wire.write(subAddress);                  // Put slave register address in Tx buffer
  Wire.endTransmission();        // Send the Tx buffer, but send a restart to keep connection alive
//  Wire.endTransmission(false);             // Send the Tx buffer, but send a restart to keep connection alive
  Wire.requestFrom((uint8_t)address, (uint8_t)1);  // Read one byte from slave register address 
//  Wire.requestFrom(address, (size_t) 1);   // Read one byte from slave register address 
  data = Wire.read();                      // Fill Rx buffer with result
  return data;                             // Return data read from slave register
}

        void readBytes(uint8_t address, uint8_t subAddress, uint8_t count, uint8_t * dest)
{  
  Wire.beginTransmission(address);   // Initialize the Tx buffer
  Wire.write(subAddress);            // Put slave register address in Tx buffer
  Wire.endTransmission();  // Send the Tx buffer, but send a restart to keep connection alive
//  Wire.endTransmission(false);       // Send the Tx buffer, but send a restart to keep connection alive
  uint8_t i = 0;
//        Wire.requestFrom(address, count);  // Read bytes from slave register address 
        Wire.requestFrom(address, (size_t) count);  // Read bytes from slave register address 
  while (Wire.available()) {
        dest[i++] = Wire.read(); }         // Put read results in the Rx buffer
}


// MFC functions

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
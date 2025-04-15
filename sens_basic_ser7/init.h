#include <Wire.h>
#include <ModbusRTUMaster.h>

/* --------- Modbus intitialization --------- */
ModbusRTUMaster modbus(Serial1);      // serial port used for modbus transceiver
#define MB_BAUD_RATE    19200         // Modbus baud rate
#define RX_PIN          0            // Pin used for the Modbus transceiver RX
#define TX_PIN          1            // Pin used for the Modbus transceiver TX
uint16_t read_buffer    [2];

/* --------- Flexiflow Registers --------- */
// found at: https://www.bronkhorst.com/getmedia/bb0e02ab-2429-4638-b751-7186bd7178fb/917035-Manual-Modbus-slave-interface.pdf
#define REG_WINK          0x0         // W  - Modbus register - PDU ADDRESS
#define REG_MEASURE       0x0020      // R  - Modbus register - PDU ADDRESS
#define REG_FMEASURE      0xA100      // R  - Modbus register - PDU ADDRESS
#define REG_COUNTER_VAL   0xE808      // RW - Modbus register - PDU ADDRESS
#define REG_SETPOINT      0x0021      // RW  - Modbus register - PDU ADDRESS
#define REG_FSETPOINT     0xA118      // RW - Modbus register - PDU ADDRESS
#define REG_TEMPERATURE   0xA138      // R  - Modbus register - PDU ADDRESS
#define REG_INITRESET     0x000A      // RW  - Modbus register - PDU ADDRESS
#define REG_FBINTFACE     0x0FA7      // RW  - Modbus register - PDU ADDRESS
#define REG_FBSELECT      0x0FA8      // RW  - Modbus register - PDU ADDRESS

/* --------- General use variables --------- */
volatile int mb_prev_millis =    0;      // Save the previous time
volatile int mb_interval_time =  3000;   // Time in milliseconds

// BME680 registers
#define BME680_FIELD_0_MEAS_STATUS_0    0x1D
#define BME680_FIELD_0_PRESS_MSB        0x1F
#define BME680_FIELD_0_TEMP_MSB         0x22
#define BME680_FIELD_0_HUM_MSB          0x25

#define BME680_FIELD_0_GAS_RL_MSB       0x2A
#define BME680_FIELD_0_GAS_RL_LSB       0x2B

#define BME680_RES_HEAT_X               0x5A // 10 RES byte values  0x5A - 0x63
#define BME680_GAS_WAIT_X               0x64 // 10 WAIT byte values 0x64 - 0x6D
#define BME680_RES_HEAT_CTRL            0x6F  
#define BME680_CTRL_GAS_1               0x71  
#define BME680_CTRL_MEAS                0x74  
#define BME680_CONFIG                   0x75  

#define BME680_ID                       0xD0  //should return 0x61
#define BME680_RESET                    0xE0   

#define BME680_CTRL_HUM                 0xF2   
#define BME680_SPI_MEM_PAGE             0xF3   

#define BME680_CALIB_ADDR_1             0x89  // 25 bytes of calibration data for I2C
#define BME680_CALIB_ADDR_2             0xE1  // 16 bytes of calibration data for I2C

#define BME680_ADDRESS                  0x76   // Address of BME680 altimeter when ADO = 0 (default)


#define SerialDebug true  // set to true to get Serial output for debugging

enum Posr {
  P_OSR_00 = 0,  // no op
  P_OSR_01,
  P_OSR_02,
  P_OSR_04,
  P_OSR_08,
  P_OSR_16
};

enum Hosr {
  H_OSR_00 = 0,  // no op
  H_OSR_01,
  H_OSR_02,
  H_OSR_04,
  H_OSR_08,
  H_OSR_16
};

enum Tosr {
  T_OSR_00 = 0,  // no op
  T_OSR_01,
  T_OSR_02,
  T_OSR_04,
  T_OSR_08,
  T_OSR_16
};

enum IIRFilter {
  full = 0,  // bandwidth at full sample rate
  BW0_223ODR,
  BW0_092ODR,
  BW0_042ODR,
  BW0_021ODR // bandwidth at 0.021 x sample rate
};

enum Mode {
  BME680Sleep = 0,
  Forced,
  Parallel,
  Sequential
};

enum SBy {
  t_00_6ms = 0,
  t_62_5ms,
  t_125ms,
  t_250ms,
  t_500ms,
  t_1000ms,
  t_10ms,
  t_20ms,
};

enum GWaitMult {
  gw_1xmult = 0,
  gw_4xmult,
  gw_16xmult,
  gw_64xmult
};

// Specify BME680 configuration
//uint8_t Posr = P_OSR_16, Hosr = H_OSR_01, Tosr = T_OSR_02, Mode = Forced, IIRFilter = BW0_042ODR, SBy = t_10ms;     // set pressure amd temperature output data rate
uint8_t Posr = P_OSR_01, Hosr = H_OSR_01, Tosr = T_OSR_01, Mode = Forced, IIRFilter = BW0_223ODR, SBy = t_10ms;     // set pressure amd temperature output data rate
// Gas sensor configuration
uint8_t GWaitMult = gw_1xmult; // choose gas sensor wait time multiplier
uint8_t numHeatPts = 0x01; // one heat set point
// choose gas wait time in milliseconds x gas wait multiplier 0x00 | 0x59 == 100 ms gas wait time
uint8_t gasWait[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // must choose at least one non-zero wait time  
uint8_t resHeat[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // must choose at least one non-zero wait time  

// Data arrays for conversion of raw gas measurements into resistance
float const_array1[16] = {1, 1, 1, 1, 1, 0.99, 1, 0.992, 1, 1, 0.998, 0.995, 1, 0.99, 1, 1};
double const_array2[16] = {8000000.0, 4000000.0, 2000000.0, 1000000.0, 499500.4995, 248262.1648, 125000.0, 
63004.03226, 31281.28128, 15625.0, 7812.5, 3906.25, 1953.125, 976.5625, 488.28125, 244.140625};

// t_fine carries fine temperature as global value for BME680
int32_t t_fine;

float Temperature, Pressure, Humidity; // stores BME680 pressures sensor pressure and temperature
uint32_t rawPress, rawTemp;   // pressure and temperature raw count output for BME680
uint16_t rawHumidity, rawGasResistance;  // variables to hold raw BME680 humidity and gas resistance values

// BME680 compensation parameters
uint8_t  dig_P10, dig_H6;
uint16_t dig_T1, dig_P1, dig_H1, dig_H2;
int16_t  dig_T2, dig_P2, dig_P4, dig_P5, dig_P8, dig_P9, dig_GH2;
int8_t   dig_T3, dig_P3, dig_P6, dig_P7, dig_H3, dig_H4, dig_H5, dig_H7, dig_GH1, dig_GH3;
float   temperature_C, temperature_F, pressure, humidity, altitude, resistance; // Scaled output of the BME680

uint32_t delt_t = 0, count = 0, sumCount = 0, slpcnt = 0;  // used to control display output rate

uint8_t status0, status1, status2;

bool bme = false;
byte cnt=0;

#define PACKET_SIZE 128  // Size of each packet
uint16_t DATA_RATE_HZ = 1;      // Data transfer rate in Hz

uint8_t buffer[PACKET_SIZE];  // Temporary buffer to hold packet data

IntervalTimer readSensTimer;

unsigned long PCFtime[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
unsigned long PCFwindow[16] = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
unsigned long PCFnow[16];
bool PCFon[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

uint16_t parse[18];
uint16_t newState;

//static int32_t  temp, humidity, pressure, gas;  // BME readings

#define MCP4725_1_ADDR 0x60  // MCP4725 U16 default I2C address (change if needed)
#define MCP4725_2_ADDR 0x61  // MCP4725 U25 default I2C address (change if needed)
#define PCF8575_ADDRESS 0x20  // Base address of the PCF8575 (change based on your A0, A1, A2 connections)
uint16_t u16_mV = 800;
uint16_t u25_mV = 200;

// Lookup table MCP4725 data
const int numPoints = 5;  // Number of data points
const float xValues[numPoints] = {100, 200, 500, 1200, 2500}; // Input values
const float yValues[numPoints] = {864, 903, 948, 993, 1025};  // Correction factors


const byte ADS1115_ADDR[4] = {0x48, 0x49, 0x4A, 0x4B};
const char checkbox_labels[17][16] = {"U1-TGS2602", "U2-TGS2610-D00", "U3-SP3S-AQ2", "U4-GSBT11-DXX", "U8-TGS2600",
                                "U9-TGS2603", "U10-TGS2630", "U13-TGS2612-D00", "U14-TGS2620", "U15-MG-812",
                                "U16-TGS-3830", "U19-TGS1820", "U20-TGS2611-E00", "U21-TGS2616-C00", "U22-WSP2110", "U25-TGS-3870", "U7-BME680"};
uint8_t sensor_id[17][2] = {{0x49, 1},{0x49, 3},{0x4b, 2},{0x4b, 3},{0x49, 0},{0x49, 2},{0x4b, 0},{0x48, 1},
                               {0x48, 3},{0x4a, 1},{0x4a, 3},{0x4a, 2},{0x48, 0},{0x48, 2},{0x4a, 0},{0x4b, 1},  
                               {0x76, 99}};

union buf {
byte buffer[PACKET_SIZE];
uint16_t bufint[PACKET_SIZE/2];
uint32_t bufint32[PACKET_SIZE/4];
};

union buf sens;
union buf ctrl; 
union buf massflow1;
union buf massflow2;
union buf massflow3;
union buf c;

union split_u2bytes {
  uint16_t result;
  byte res_byte[2];
}splitu2;

int n;

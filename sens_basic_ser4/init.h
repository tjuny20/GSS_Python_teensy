//#include "/home/herman/Documents/thonny_test/sens_basic_sernew/Wire/Wire.h"
#include <Wire.h>

#include "Zanshin_BME680.h"  // Include the BME680 Sensor library

BME680_Class BME680;  ///< Create an instance of the BME680 class

bool bme = false;

#define PACKET_SIZE 128  // Size of each packet
uint16_t DATA_RATE_HZ = 100;      // Data transfer rate in Hz

uint8_t buffer[PACKET_SIZE];  // Temporary buffer to hold packet data

IntervalTimer readSensTimer;

unsigned long PCFtime[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
unsigned long PCFwindow[16] = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
unsigned long PCFnow[16];
bool PCFon[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

uint16_t parse[18];
uint16_t newState;

static int32_t  temp, humidity, pressure, gas;  // BME readings

#define MCP4725_1_ADDR 0x60  // MCP4725 U16 default I2C address (change if needed)
#define MCP4725_2_ADDR 0x61  // MCP4725 U25 default I2C address (change if needed)
#define PCF8575_ADDRESS 0x20  // Base address of the PCF8575 (change based on your A0, A1, A2 connections)
uint16_t u16_mV = 800;
uint16_t u25_mV = 200;

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

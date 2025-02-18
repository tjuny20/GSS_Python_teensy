
#include "init.h"

void loop() {

  // Check for incoming commands from the PC
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    //Serial.println(input);
    parseInput(input);
  }

  if ((PCFon[0] == true) & (PCFwindow[0] > 0)) {
    PCFnow[0] = millis();
    if (PCFnow[0] - PCFtime[0] > PCFwindow[0]) {
      PCFtime[0] = PCFnow[0];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000000001) == 0) writePCF8575(readPCF | 0b0000000000000001); else writePCF8575(readPCF & 0b1111111111111110);
    }
  }

  if ((PCFon[1] == true) & (PCFwindow[1] > 0)) {
    PCFnow[1] = millis();
    if (PCFnow[1] - PCFtime[1] > PCFwindow[1]) {
      PCFtime[1] = PCFnow[1];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000000010) == 0) writePCF8575(readPCF | 0b0000000000000010); else writePCF8575(readPCF & 0b1111111111111101);
    }
  }

  if ((PCFon[2] == true) & (PCFwindow[2] > 0)) {
    PCFnow[2] = millis();
    if (PCFnow[2] - PCFtime[2] > PCFwindow[2]) {
      PCFtime[2] = PCFnow[2];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000000100) == 0) writePCF8575(readPCF | 0b0000000000000100); else writePCF8575(readPCF & 0b11111111111111011);
    }
  }

  if ((PCFon[3] == true) & (PCFwindow[3] > 0)) {
    PCFnow[3] = millis();
    if (PCFnow[3] - PCFtime[3] > PCFwindow[3]) {
      PCFtime[3] = PCFnow[3];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000001000) == 0) writePCF8575(readPCF | 0b0000000000001000); else writePCF8575(readPCF & 0b1111111111110111);
    }
  }

  if ((PCFon[4] == true) & (PCFwindow[4] > 0)) {
    PCFnow[4] = millis();
    if (PCFnow[4] - PCFtime[4] > PCFwindow[4]) {
      PCFtime[4] = PCFnow[4];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000010000) == 0) writePCF8575(readPCF | 0b0000000000010000); else writePCF8575(readPCF & 0b1111111111101111);
    }
  }

  if ((PCFon[5] == true) & (PCFwindow[5] > 0)) {
    PCFnow[5] = millis();
    if (PCFnow[5] - PCFtime[5] > PCFwindow[5]) {
      PCFtime[5] = PCFnow[5];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000000000100000) == 0) writePCF8575(readPCF | 0b0000000000100000); else writePCF8575(readPCF & 0b1111111111011111);
    }
  }

  if ((PCFon[10] == true) & (PCFwindow[10] > 0)) {
    PCFnow[10] = millis();
    if (PCFnow[10] - PCFtime[10] > PCFwindow[10]) {
      PCFtime[10] = PCFnow[10];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000010000000000) == 0) writePCF8575(readPCF | 0b0000010000000000); else writePCF8575(readPCF & 0b1111101111111111);
    }
  }

  if ((PCFon[11] == true) & (PCFwindow[11] > 0)) {
    PCFnow[11] = millis();
    if (PCFnow[11] - PCFtime[11] > PCFwindow[11]) {
      PCFtime[11] = PCFnow[11];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0000100000000000) == 0) writePCF8575(readPCF | 0b0000100000000000); else writePCF8575(readPCF & 0b1111011111111111);
    }
  }

  if ((PCFon[12] == true) & (PCFwindow[12] > 0)) {
    PCFnow[12] = millis();
    if (PCFnow[12] - PCFtime[12] > PCFwindow[12]) {
      PCFtime[12] = PCFnow[12];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0001000000000000) == 0) writePCF8575(readPCF | 0b0001000000000000); else writePCF8575(readPCF & 0b1110111111111111);
    }
  }

  if ((PCFon[13] == true) & (PCFwindow[13] > 0)) {
    PCFnow[13] = millis();
    if (PCFnow[13] - PCFtime[13] > PCFwindow[13]) {
      PCFtime[13] = PCFnow[13];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0010000000000000) == 0) writePCF8575(readPCF | 0b0010000000000000); else writePCF8575(readPCF & 0b1101111111111111);
    }
  }

  if ((PCFon[14] == true) & (PCFwindow[14] > 0)) {
    PCFnow[14] = millis();
    if (PCFnow[14] - PCFtime[14] > PCFwindow[14]) {
      PCFtime[14] = PCFnow[14];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b0100000000000000) == 0) writePCF8575(readPCF | 0b0100000000000000); else writePCF8575(readPCF & 0b1011111111111111);
    }
  }

  if ((PCFon[15] == true) & (PCFwindow[15] > 0)) {
    PCFnow[15] = millis();
    if (PCFnow[15] - PCFtime[15] > PCFwindow[15]) {
      PCFtime[15] = PCFnow[15];
      uint16_t readPCF = readPCF8575();
      if ((readPCF & 0b1000000000000000) == 0) writePCF8575(readPCF | 0b1000000000000000); else writePCF8575(readPCF & 0b0111111111111111);
    }
  }


}

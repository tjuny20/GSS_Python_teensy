
#include "init.h"

void loop() {

  // Check for incoming commands from the PC
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    Serial.println(input);
    parseInput(input);
  }

  // Iterate through all PCF configurations
  for (int i = 0; i < 16; i++) {
    if (PCFon[i] && PCFwindow[i] > 0) {
      PCFnow[i] = millis();
      if (PCFnow[i] - PCFtime[i] > PCFwindow[i]) {
        PCFtime[i] = PCFnow[i];
        uint16_t readPCF = readPCF8575();
        uint16_t mask = 1 << i; // Create a mask for the current bit
        if ((readPCF & mask) == 0) {
          writePCF8575(readPCF | mask); // Set the bit
        } else {
          writePCF8575(readPCF & ~mask); // Clear the bit
        }
      }
    }
  }
}

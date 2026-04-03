#include <Arduino.h>
#include <Wire.h>
#include <MPU6050_light.h>

MPU6050 mpu(Wire);

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  byte status = mpu.begin();
  if (status != 0) {
    while (1) delay(10); 
  }

  delay(1000);
  mpu.calcOffsets(); 
}

void loop() {
  mpu.update();

  Serial.print(mpu.getAngleX());
  Serial.print(",");
  Serial.print(mpu.getAngleY());
  Serial.print(",");
  Serial.println(mpu.getAngleZ());

  delay(10); 
}

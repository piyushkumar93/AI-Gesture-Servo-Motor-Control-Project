#include <Servo.h>

Servo myServo;
String input = "";

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
  myServo.write(0);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      input.trim();
      if (input == "Thumbs Up") {
        myServo.write(90);
      } else if (input == "Stop") {
        myServo.write(0);
      }
      input = "";
    } else {
      input += c;
    }
  }
}

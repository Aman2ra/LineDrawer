// Make sure python isnt using comPort while you try to Upload

#include <AccelStepper.h>
#include <Servo.h>

#define DRAW 1
#define ERASE -1

#define CONSTANT 1
#define ACCEL 2

int interface = AccelStepper::HALF4WIRE;
int topPins[] = {2,4,3,5};
int centerPins[] = {6,8,7,9};
int bottomPins[] = {10,12,11,13};

int drawServoPin = 44;
int drawServoBasePos = 180;
int drawServoEngagePos = 149;
int drawServoPos = drawServoBasePos;

// Motor Setup
AccelStepper topMotor(interface, topPins[0], topPins[1], topPins[2], topPins[3]);
AccelStepper centerMotor(interface, centerPins[0], centerPins[1], centerPins[2], centerPins[3]);
AccelStepper bottomMotor(interface, bottomPins[0], bottomPins[1], bottomPins[2], bottomPins[3]);

Servo drawServo;

float stepsPerRev = 4096;
int secsPerMin = 60;
float maxMotSpeed = 32;
float maxAcceleration = 3000.0;
int motBaseSpeed = 15;
float acceleration = 1000.0;
float maxiSpeed = motBaseSpeed*stepsPerRev/secsPerMin; // in steps per second
float desiSpeed =  motBaseSpeed*stepsPerRev/secsPerMin; // in steps per second


long xDistCurr = 0;
long yDistCurr = 0;
long xDistPrev = 0;
long yDistPrev = 0;
int drawToolPrev = 0;
float speedX = 1;
float speedY = 1;
float dirX = 1;
float dirY = 1;
int drawToolCurr = 0;

int speedMode = CONSTANT;
//int speedMode = ACCEL;
int moveFinished = 1;
String dataFromPython;

void setup() {
  Serial.begin(9600);
  Serial.println("Enter Travel distance and drawPos separated by a comma: X,Y,D ");
  Serial.println("Enter Move Values Now: ");

  topMotor.setMaxSpeed(maxiSpeed);
  topMotor.setAcceleration(acceleration);
  topMotor.setSpeed(desiSpeed);
  
  centerMotor.setMaxSpeed(maxiSpeed);
  centerMotor.setAcceleration(acceleration);
  centerMotor.setSpeed(desiSpeed);
  
  bottomMotor.setMaxSpeed(maxiSpeed);
  bottomMotor.setAcceleration(acceleration);
  bottomMotor.setSpeed(desiSpeed);

  drawServo.attach(drawServoPin);
  drawServo.write(drawServoPos);
}
long f = 0;

void loop() {    
  while (Serial.available() > 0 && moveFinished == 1) {
    f = 0;
    xDistCurr = Serial.readStringUntil(',').toInt();
    yDistCurr = -1*Serial.readStringUntil(',').toInt();
    drawToolCurr = Serial.readStringUntil('\r').toInt();
    Serial.println("");
    Serial.print(millis());
    Serial.print(":[");
    Serial.print(f);
    Serial.print("] (");
    Serial.print(xDistCurr);
    Serial.print(", ");
    Serial.print(yDistCurr);
    Serial.print(", ");
    Serial.print(drawToolCurr);
    Serial.println(")");

    
    if (drawToolCurr == DRAW) {
      drawServoPos = drawServoEngagePos;
    } else if (drawToolCurr == ERASE) {
      drawServoPos = -drawServoEngagePos;
    } else {
      drawServoPos = drawServoBasePos;
    }
    
    if (abs(xDistCurr-xDistPrev) >= abs(yDistCurr-yDistPrev) && abs(xDistCurr-xDistPrev) != 0){
        speedX = 1;
        speedY = float(abs(yDistCurr-yDistPrev)) / abs(xDistCurr-xDistPrev);
    } else if (abs(xDistCurr-xDistPrev) < abs(yDistCurr-yDistPrev) && abs(yDistCurr-yDistPrev) != 0){
        speedY = 1;
        speedX = float(abs(xDistCurr-xDistPrev)) / abs(yDistCurr-yDistPrev);
    }

    if (speedMode == ACCEL) {
      topMotor.setMaxSpeed(speedX*motBaseSpeed*stepsPerRev/secsPerMin);
      centerMotor.setMaxSpeed(speedY*motBaseSpeed*stepsPerRev/secsPerMin);
      bottomMotor.setMaxSpeed(speedX*motBaseSpeed*stepsPerRev/secsPerMin);
    }
    
    topMotor.moveTo(xDistCurr);
    centerMotor.moveTo(yDistCurr);
    bottomMotor.moveTo(-xDistCurr);
    
    if (speedMode == CONSTANT) {
      if (xDistCurr > xDistPrev){
          dirX = 1;
      } else {
          dirX = -1;        
      } if (yDistCurr > yDistPrev){
          dirY = 1;
      } else {
          dirY = -1;        
      }
      topMotor.setSpeed(dirX*speedX*motBaseSpeed*stepsPerRev/secsPerMin);
      centerMotor.setSpeed(-dirY*speedY*motBaseSpeed*stepsPerRev/secsPerMin);
      bottomMotor.setSpeed(-dirX*speedX*motBaseSpeed*stepsPerRev/secsPerMin);
    }
    drawServo.write(drawServoPos);
    if (drawToolPrev != drawToolCurr) {
      delay(250);
    }
    moveFinished = 0;
    Serial.print("    ");
    Serial.print(millis());
    Serial.print(":[");
    Serial.print(f);
    Serial.println("] Move Start");
  }

  if ((topMotor.distanceToGo() != 0) || (centerMotor.distanceToGo() != 0) || (bottomMotor.distanceToGo() != 0)) {
    if (speedMode == CONSTANT){
      topMotor.runSpeedToPosition();
      centerMotor.runSpeedToPosition();
      bottomMotor.runSpeedToPosition();
    } else if (speedMode == ACCEL) {
      topMotor.run();
      centerMotor.run();
      bottomMotor.run();
    }
    f += 1;
  }   
  
  if ((moveFinished == 0) && (topMotor.distanceToGo() == 0) && (centerMotor.distanceToGo() == 0) && (bottomMotor.distanceToGo() == 0)) {
    xDistPrev = xDistCurr;
    yDistPrev = yDistCurr;
    drawToolPrev = drawToolCurr;
    Serial.print("    ");
    Serial.print(millis());
    Serial.print(":[");
    Serial.print(f);
    Serial.print("] (");
    Serial.print(xDistPrev);
    Serial.print(", ");
    Serial.print(yDistPrev);
    Serial.print(", ");
    Serial.print(drawToolPrev);
    Serial.println(")");
    Serial.print("    ");
    Serial.print(millis());
    Serial.print(":[");
    Serial.print(f);
    Serial.println("] COMPLETED");
    Serial.println("COMPLETED");
    moveFinished = 1;
    f += 1;
  }
//  delay(10);
}

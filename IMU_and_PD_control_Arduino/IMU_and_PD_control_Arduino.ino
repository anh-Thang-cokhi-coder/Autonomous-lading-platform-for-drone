
#include <Wire.h>
#include <ICM20948_WE.h>
#include <AccelStepper.h>

#define ICM20948_ADDR 0x69

ICM20948_WE myIMU(ICM20948_ADDR);


#define X_STEP_PIN 2
#define Y_STEP_PIN 3
#define Z_STEP_PIN 4

#define X_DIR_PIN 5
#define Y_DIR_PIN 6
#define Z_DIR_PIN 7

#define ENABLE_PIN 8

AccelStepper motorX(
  AccelStepper::DRIVER,
  X_STEP_PIN,
  X_DIR_PIN);

AccelStepper motorY(
  AccelStepper::DRIVER,
  Y_STEP_PIN,
  Y_DIR_PIN);

AccelStepper motorZ(
  AccelStepper::DRIVER,
  Z_STEP_PIN,
  Z_DIR_PIN);

// Camera target

float camRoll  = 0;
float camPitch = 0;
float camYaw   = 0;

unsigned long lastCameraUpdate = 0;

//PD control

const float KP = 3.0;
const float KD = 0.5;

const float DEAD_BAND = 4.0;

//Speed

const float MAX_SPEED_X = 60;
const float MAX_SPEED_Y = 100;
const float MAX_SPEED_Z = 100;


float prevErrRoll  = 0;
float prevErrPitch = 0;
float prevErrYaw   = 0;

class SimpleKalman
{
  public:

    float Q = 0.01;
    float R = 0.5;

    float P = 1.0;
    float X = 0.0;

    float update(float measurement)
    {
      P += Q;

      float K = P / (P + R);

      X = X + K * (measurement - X);

      P = (1.0 - K) * P;

      return X;
    }
};

SimpleKalman kalRoll;
SimpleKalman kalPitch;
SimpleKalman kalYaw;



float roll = 0;
float pitch = 0;
float yaw = 0;

unsigned long lastMicros = 0;
unsigned long lastSend = 0;



void setup()
{
  Serial.begin(115200);

  Wire.begin();

  if (!myIMU.init())
  {
    while (1);
  }

  myIMU.initMagnetometer();

  delay(1000);

  myIMU.setAccRange(ICM20948_ACC_RANGE_2G);
  myIMU.setAccDLPF(ICM20948_DLPF_6);

  lastMicros = micros();

  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(
    ENABLE_PIN,
    LOW);

  motorX.setMaxSpeed(MAX_SPEED_X);
  motorY.setMaxSpeed(MAX_SPEED_Y);
  motorZ.setMaxSpeed(MAX_SPEED_Z);
}

void updateOrientation()
{
  myIMU.readSensor();

  xyzFloat gyr;
  xyzFloat mag;

  myIMU.getGyrValues(&gyr);
  myIMU.getMagValues(&mag);

  unsigned long now = micros();

  float dt =
      (now - lastMicros)
      / 1000000.0f;

  lastMicros = now;

  float rollAccel =
      myIMU.getRoll();

  float pitchAccel =
      myIMU.getPitch();

  roll =
      0.98f *
      (roll + gyr.x * dt)
      +
      0.02f *
      rollAccel;

  pitch =
      0.98f *
      (pitch + gyr.y * dt)
      +
      0.02f *
      pitchAccel;

float yawRaw =
    atan2(-mag.y, mag.x)
    * 180.0f / PI;

  roll =
      kalRoll.update(roll);

  pitch =
      kalPitch.update(pitch);

  yaw =
      kalYaw.update(yawRaw);
}


void sendToPi()
{
  if (millis() - lastSend < 50)
  {
    return;
  }

  lastSend = millis();

  Serial.print("I,");

  Serial.print(roll, 2);
  Serial.print(",");

  Serial.print(pitch, 2);
  Serial.print(",");

  Serial.println(yaw, 2);
}


float wrapYawError(
  float target,
  float current)
{
  float error =
      target - current;

  while (error > 180)
    error -= 360;

  while (error < -180)
    error += 360;

  return error;
}

void receiveCamera()
{
  static String line = "";

  while (Serial.available())
  {
    char c =
        Serial.read();

    if (c == '\n')
    {
      if (line.startsWith("C,"))
      {
        
        int r,p,y;

        sscanf(
      line.c_str(),
      "C,%d,%d,%d",
      &r,&p,&y);

        camRoll = r;
        camPitch = p;
        camYaw = y; 

        lastCameraUpdate =millis();
      
      }

      line = "";
    }
    else
    {
      line += c;
    }
  }
}
void updatePD()
{
  // lost track from camera

  if (
      millis()
      -
      lastCameraUpdate
      >
      500)
  {
    motorX.setSpeed(0);
    motorY.setSpeed(0);
    motorZ.setSpeed(0);

    return;
  }

  float errRoll =
      camRoll - roll;

  float errPitch =
      camPitch - pitch;

  float errYaw =
      wrapYawError(
          camYaw,
          yaw);

  // deadband

  if (abs(errRoll) < DEAD_BAND)
    errRoll = 0;

  if (abs(errPitch) < DEAD_BAND)
    errPitch = 0;

  if (abs(errYaw) < DEAD_BAND)
    errYaw = 0;

  // PD

  float speedRoll =
      KP * errRoll
      +
      KD *
      (
          errRoll
          -
          prevErrRoll
      );

  float speedPitch =
      KP * errPitch
      +
      KD *
      (
          errPitch
          -
          prevErrPitch
      );

  float speedYaw =
      KP * errYaw
      +
      KD *
      (
          errYaw
          -
          prevErrYaw
      );

  prevErrRoll =
      errRoll;

  prevErrPitch =
      errPitch;

  prevErrYaw =
      errYaw;

  // limit

  speedYaw =
      constrain(
          speedYaw,
          -MAX_SPEED_X,
          MAX_SPEED_X);

  speedPitch =
      constrain(
          speedPitch,
          -MAX_SPEED_Y,
          MAX_SPEED_Y);

  speedRoll =
      constrain(
          speedRoll,
          -MAX_SPEED_Z,
          MAX_SPEED_Z);

  // X = yaw

  motorX.setSpeed(
      speedYaw);

  // Y = pitch

  motorY.setSpeed(
      speedPitch);

  // Z = roll

  motorZ.setSpeed(
      speedRoll);
}
void loop()
{
  receiveCamera();

  updateOrientation();

  updatePD();

  motorX.runSpeed();
  motorY.runSpeed();
  motorZ.runSpeed();

  sendToPi();
}


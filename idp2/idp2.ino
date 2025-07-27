#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// === Pins ===
#define CURRENT_PIN 34
#define VOLTAGE_PIN 35   // Replaces INA219, now using analogRead
#define ONE_WIRE_BUS 4
#define RELAY_PIN 26

// === Voltage Divider Constants ===
const float R1 = 4700.0;
const float R2 = 1000.0;
const float VREF = 3.3;
const float ADC_RESOLUTION = 4095.0;

// === Other Constants ===
const float ACS712_SENSITIVITY = 0.185;
const float TORQUE_CONSTANT = 0.79/18.0;

// === Sensor Objects ===
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

// === BLE UUIDs ===
#define SERVICE_UUID        "d5aaf934-4de5-11ee-be56-0242ac120002"
#define CHARACTERISTIC_UUID "d5aafc08-4de5-11ee-be56-0242ac120002"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// === BLE Callbacks ===
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) { deviceConnected = true; }
  void onDisconnect(BLEServer* pServer) { deviceConnected = false; }
};

// === Sensor Functions ===
float getVoltage() {
  int raw = analogRead(VOLTAGE_PIN);
  float vout = (raw / ADC_RESOLUTION) * VREF;
  float vin = vout * (R1 + R2) / R2;
  return vin;
}

float getCurrent() {
  int raw = analogRead(CURRENT_PIN);
  float voltage = (raw / ADC_RESOLUTION) * VREF;
  return (voltage - VREF / 2) / ACS712_SENSITIVITY;
}

float getTemperature() {
  tempSensor.requestTemperatures();
  return tempSensor.getTempCByIndex(0);
}

// === Setup ===
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(VOLTAGE_PIN, ADC_11db);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Active LOW relay
  tempSensor.begin();

  // BLE setup
  BLEDevice::init("ESP32-MotorMonitor");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  pServer->getAdvertising()->start();
}

// === Loop ===
void loop() {
  if (deviceConnected) {
    float voltage = getVoltage();
    float current = getCurrent();
    float temperature = getTemperature();
    float power = voltage * current;
    float torque = current * TORQUE_CONSTANT;
    float energy = power * 1; 
    float rpm = max(0.0, 12000 * (1.0 - (current - 1.2) / (18.0 - 1.2)));
    float efficiency = (2 * PI * rpm * torque)/(60 * voltage * current);

    // === Send BLE Data ===
    char buffer[128];
    snprintf(buffer, sizeof(buffer),
            "V:%.2f,I:%.2f,T:%.2f,P:%.2f,TQ:%.3f,E:%.2f,R:%.2f,Ef:%.2f",voltage, current, temperature, power, torque, energy, rpm, efficiency);

    pCharacteristic->setValue((uint8_t*)buffer, strlen(buffer));
    pCharacteristic->notify();
    Serial.println(buffer);

    delay(1000);  // 1-second update
  } 
  else {
    delay(500);
  }
}

#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include "boundary_controller.h"
#include "command_protocol.h"
#include "motor_driver.h"
#include "wifi_credentials.h"

// Измените только эти четыре константы, если разводка отличается.
// Electrical driver channels. The server swaps them to physical left/right
// so both current and already-installed firmware use the same protocol.
constexpr uint8_t LEFT_IN1_PIN = 27;
constexpr uint8_t LEFT_IN2_PIN = 12;
constexpr uint8_t RIGHT_IN1_PIN = 25;
constexpr uint8_t RIGHT_IN2_PIN = 26;
constexpr uint8_t ARM_SERVO_PIN = 32;
constexpr uint8_t BUCKET_SERVO_PIN = 33;
constexpr uint32_t SERIAL_BAUD = 115200;
// ESP32 joins the configured local network as a station (STA).
// WIFI_SSID and WIFI_PASSWORD live in the ignored wifi_credentials.h file.
constexpr char WIFI_HOSTNAME[] = "robot-pogruzchik1";
constexpr uint16_t WIFI_COMMAND_PORT = 3333;
constexpr uint8_t ARM_START_ANGLE = 90;
constexpr uint8_t BUCKET_START_ANGLE = 90;
constexpr uint16_t SERVO_MIN_PULSE_US = 500;
constexpr uint16_t SERVO_MAX_PULSE_US = 2400;
constexpr uint16_t SERVO_FREQUENCY_HZ = 50;
// Fail-safe: the server must refresh a movement command at least once a second.
constexpr uint32_t MOTOR_COMMAND_TIMEOUT_MS = 1000;
// High-rate movement refreshes must not produce a response for every packet:
// on an unreliable radio that response flood is worse than the lost packets.
constexpr uint32_t STATE_RESPONSE_INTERVAL_MS = 50;

MotorDriver motors(LEFT_IN1_PIN, LEFT_IN2_PIN, RIGHT_IN1_PIN, RIGHT_IN2_PIN);
BoundaryController boundary;
CommandProtocol serialProtocol;
CommandProtocol wifiProtocol;
WiFiServer wifiServer(WIFI_COMMAND_PORT);
WiFiClient wifiClient;
Print *response = &Serial;
Servo armServo;
Servo bucketServo;
uint8_t armAngle = ARM_START_ANGLE;
uint8_t bucketAngle = BUCKET_START_ANGLE;
bool armAttached = false;
bool bucketAttached = false;
uint32_t motionStopMs = 0;
uint32_t nextWifiReconnectMs = 0;
bool wifiWasConnected = false;
uint32_t lastStateMs = 0;
bool statePending = false;
bool emergencyStopLatched = false;

bool attachAndWriteServo(Servo &servo, uint8_t pin, uint8_t angle) {
    if (!servo.attached()) {
        servo.setPeriodHertz(SERVO_FREQUENCY_HZ);
        servo.attach(pin, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);
    }
    if (!servo.attached()) return false;
    servo.write(angle);
    return true;
}

void connectLocalWifi() {
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.setHostname(WIFI_HOSTNAME);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    nextWifiReconnectMs = millis() + 5000;
}

void sendState(bool force = true) {
    const uint32_t now = millis();
    if (!force && static_cast<uint32_t>(now - lastStateMs) < STATE_RESPONSE_INTERVAL_MS) {
        statePending = true;
        return;
    }
    response->printf("STATE motion=%s left_pwm=%d right_pwm=%d speed=%u emergency=%u arm=%u arm_active=%u bucket=%u bucket_active=%u boundary=%s x=%.2f y=%.2f heading=%.1f\n",
                  MotorDriver::motionName(motors.motion()), motors.leftOutput(), motors.rightOutput(), motors.speed(),
                  emergencyStopLatched ? 1U : 0U,
                  armAngle, armAttached, bucketAngle, bucketAttached,
                  BoundaryController::statusName(boundary.status(millis())),
                  boundary.x(), boundary.y(), boundary.heading());
    lastStateMs = now;
    statePending = false;
}

void reject(const char *reason) {
    motors.stop();
    response->printf("ERR %s\n", reason);
    sendState();
}

void requestMotion(Motion motion) {
    if (emergencyStopLatched) {
        motors.stop();
        motionStopMs = 0;
        sendState(false);
        return;
    }
    if (!boundary.isMotionAllowed(motion, millis())) {
        reject("MOTION_BLOCKED_BY_BOUNDARY");
        return;
    }
    motors.move(motion);
    motionStopMs = millis() + MOTOR_COMMAND_TIMEOUT_MS;
    sendState(false);
}

void handleCommand(const Command &command) {
    switch (command.type) {
        case CommandType::HALT:
            // A routine controller stop must never latch the red emergency
            // state. STOP is reserved exclusively for the emergency button.
            motors.stop();
            motionStopMs = 0;
            response->println("OK HALTED");
            sendState();
            break;
        case CommandType::STOP:
            motors.stop();
            motionStopMs = 0;
            emergencyStopLatched = true;
            response->println("OK STOP_LATCHED");
            sendState();
            break;
        case CommandType::RESUME:
            motors.stop();
            motionStopMs = 0;
            emergencyStopLatched = false;
            response->println("OK RESUMED");
            sendState();
            break;
        case CommandType::FORWARD: requestMotion(Motion::FORWARD); break;
        case CommandType::BACKWARD: requestMotion(Motion::BACKWARD); break;
        case CommandType::LEFT: requestMotion(Motion::LEFT); break;
        case CommandType::RIGHT: requestMotion(Motion::RIGHT); break;
        case CommandType::DRIVE:
            if (emergencyStopLatched) {
                motors.stop();
                motionStopMs = 0;
                sendState(false);
                break;
            }
            // Independent forward PWM for a smooth differential-drive arc.
            // Negative values are deliberately rejected: the controller must
            // never command a tank turn with the wheels in opposite directions.
            if (command.valueCount != 2 || command.values[0] < 0 || command.values[0] > 255 ||
                command.values[1] < 0 || command.values[1] > 255) {
                reject("DRIVE_REQUIRES_LEFT_RIGHT_0_TO_255");
                break;
            }
            if (!boundary.isMotionAllowed(Motion::FORWARD, millis())) {
                reject("MOTION_BLOCKED_BY_BOUNDARY");
                break;
            }
            motors.driveRaw(static_cast<int16_t>(command.values[0]), static_cast<int16_t>(command.values[1]));
            motionStopMs = millis() + MOTOR_COMMAND_TIMEOUT_MS;
            sendState(false);
            break;

        case CommandType::MANUAL_DRIVE:
            if (emergencyStopLatched) {
                motors.stop();
                motionStopMs = 0;
                sendState(false);
                break;
            }
            // Dead-man WASD control intentionally bypasses ArUco boundaries:
            // this mode is used when no camera pose exists. Signed PWM allows
            // forward, reverse and a controlled turn in either direction.
            if (command.valueCount != 2 || command.values[0] < -255 || command.values[0] > 255 ||
                command.values[1] < -255 || command.values[1] > 255) {
                reject("MANUAL_DRIVE_REQUIRES_LEFT_RIGHT_MINUS_255_TO_255");
                break;
            }
            motors.driveRaw(static_cast<int16_t>(command.values[0]), static_cast<int16_t>(command.values[1]));
            motionStopMs = millis() + MOTOR_COMMAND_TIMEOUT_MS;
            sendState(false);
            break;

        case CommandType::SET_MOTOR_INVERSION:
            if (command.valueCount != 2 || (command.values[0] != 0 && command.values[0] != 1) ||
                (command.values[1] != 0 && command.values[1] != 1)) {
                reject("SET_MOTOR_INVERSION_REQUIRES_LEFT_RIGHT_0_OR_1");
                break;
            }
            motors.stop();
            motors.setDirectionInverted(command.values[0] == 1, command.values[1] == 1);
            response->println("OK MOTOR_INVERSION");
            sendState();
            break;

        case CommandType::SET_SPEED:
            if (command.valueCount != 1 || command.values[0] < 0 || command.values[0] > 255) {
                reject("SET_SPEED_REQUIRES_0_TO_255");
                break;
            }
            motors.setSpeed(static_cast<uint8_t>(command.values[0]));
            // Применяем новую скорость к уже выполняемой команде.
            motors.move(motors.motion());
            response->println("OK");
            sendState();
            break;

        case CommandType::SET_BOUNDS: {
            if (command.valueCount != 4 && command.valueCount != 5) {
                reject("SET_BOUNDS_REQUIRES_MINX_MINY_MAXX_MAXY_OPTIONAL_MARGIN");
                break;
            }
            const float margin = command.valueCount == 5 ? command.values[4] : 10.0f;
            boundary.setBounds(command.values[0], command.values[1],
                               command.values[2], command.values[3], margin);
            motors.stop();
            if (!boundary.enabled()) {
                response->println("ERR INVALID_BOUNDS");
            } else {
                response->println("OK");
            }
            sendState();
            break;
        }

        case CommandType::POSITION:
            if (command.valueCount != 3 ||
                !boundary.setPosition(command.values[0], command.values[1], command.values[2], millis())) {
                reject("POSITION_REQUIRES_X_Y_HEADING");
                break;
            }
            if (boundary.needsEmergencyStop(millis())) {
                reject("ROBOT_OUTSIDE_BOUNDS");
            }
            break;

        case CommandType::SET_ARM:
            if (emergencyStopLatched) {
                response->println("ERR SERVO_BLOCKED_BY_EMERGENCY_STOP");
                sendState();
                break;
            }
            if (command.valueCount != 1 || command.values[0] < 0 || command.values[0] > 180) {
                reject("SET_ARM_REQUIRES_0_TO_180");
                break;
            }
            armAngle = static_cast<uint8_t>(command.values[0]);
            armAttached = attachAndWriteServo(armServo, ARM_SERVO_PIN, armAngle);
            if (!armAttached) {
                response->println("ERR ARM_SERVO_ATTACH_FAILED");
                sendState();
                break;
            }
            response->printf("OK SET_ARM angle=%u pulse_us=%d\n", armAngle, armServo.readMicroseconds());
            sendState();
            break;

        case CommandType::SET_BUCKET:
            if (emergencyStopLatched) {
                response->println("ERR SERVO_BLOCKED_BY_EMERGENCY_STOP");
                sendState();
                break;
            }
            if (command.valueCount != 1 || command.values[0] < 0 || command.values[0] > 180) {
                reject("SET_BUCKET_REQUIRES_0_TO_180");
                break;
            }
            bucketAngle = static_cast<uint8_t>(command.values[0]);
            bucketAttached = attachAndWriteServo(bucketServo, BUCKET_SERVO_PIN, bucketAngle);
            if (!bucketAttached) {
                response->println("ERR BUCKET_SERVO_ATTACH_FAILED");
                sendState();
                break;
            }
            response->printf("OK SET_BUCKET angle=%u pulse_us=%d\n", bucketAngle, bucketServo.readMicroseconds());
            sendState();
            break;

        case CommandType::TEST_LEFT:
        case CommandType::TEST_RIGHT: {
            if (emergencyStopLatched) {
                motors.stop();
                motionStopMs = 0;
                sendState(false);
                break;
            }
            if (command.valueCount != 1 || command.values[0] < -255 || command.values[0] > 255) {
                reject("MOTOR_TEST_REQUIRES_MINUS_255_TO_255");
                break;
            }
            const int16_t pwm = static_cast<int16_t>(command.values[0]);
            if (command.type == CommandType::TEST_LEFT) motors.driveRaw(pwm, 0);
            else motors.driveRaw(0, pwm);
            motionStopMs = millis() + MOTOR_COMMAND_TIMEOUT_MS;
            sendState(false);
            break;
        }

        case CommandType::STATUS:
            sendState();
            break;
        case CommandType::INVALID:
            reject(command.error.c_str());
            break;
        default:
            break;
    }
}

void setup() {
    // Reserve one hardware timer for both 50 Hz servo channels. Motor PWM uses
    // fixed channels 4..7 on other timers, so the two systems cannot collide.
    ESP32PWM::allocateTimer(0);
    // Моторы инициализируются до Serial: после сброса они гарантированно выключены.
    motors.begin();
    // Стандартные импульсы большинства аналоговых сервоприводов.
    Serial.begin(SERIAL_BAUD);
    Serial.setTimeout(10);
    delay(200);
    // Initialize both Servo-library outputs during boot instead of waiting for
    // the first network command. This keeps the 50 Hz signal alive at all times.
    armAttached = attachAndWriteServo(armServo, ARM_SERVO_PIN, armAngle);
    delay(50);
    bucketAttached = attachAndWriteServo(bucketServo, BUCKET_SERVO_PIN, bucketAngle);
    delay(50);
    connectLocalWifi();
    wifiServer.begin();
    wifiServer.setNoDelay(true);
    Serial.println("READY robot-controller-v3-servo-init");
    Serial.printf("SERVO arm_pin=%u arm_active=%u arm_angle=%u arm_us=%d bucket_pin=%u bucket_active=%u bucket_angle=%u bucket_us=%d range_us=%u..%u hz=%u\n",
                  ARM_SERVO_PIN, armAttached, armAngle, armServo.readMicroseconds(),
                  BUCKET_SERVO_PIN, bucketAttached, bucketAngle, bucketServo.readMicroseconds(),
                  SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US, SERVO_FREQUENCY_HZ);
    Serial.printf("WIFI STA connecting ssid=%s port=%u\n", WIFI_SSID, WIFI_COMMAND_PORT);
    sendState();
}

void loop() {
    Command command;
    response = &Serial;
    if (serialProtocol.poll(Serial, command)) handleCommand(command);

    const bool wifiConnected = WiFi.status() == WL_CONNECTED;
    if (!wifiConnected && static_cast<int32_t>(millis() - nextWifiReconnectMs) >= 0) {
        connectLocalWifi();
    }
    if (wifiConnected && !wifiWasConnected) {
        Serial.printf("WIFI STA connected ssid=%s ip=%s port=%u\n", WIFI_SSID,
                      WiFi.localIP().toString().c_str(), WIFI_COMMAND_PORT);
    }
    wifiWasConnected = wifiConnected;

    if (wifiClient && !wifiClient.connected()) {
        // Loss of the controller socket is an immediate stop condition.
        motors.stop();
        motionStopMs = 0;
        wifiClient.stop();
        wifiProtocol.reset();
    }
    if (!wifiClient) {
        WiFiClient candidate = wifiServer.available();
        if (candidate) {
            wifiClient = candidate;
            wifiProtocol.reset();
            wifiClient.setNoDelay(true);
            response = &wifiClient;
            response->println("READY robot-controller-v3-servo-init");
            sendState();
        }
    }
    if (wifiClient && wifiClient.connected()) {
        response = &wifiClient;
        if (wifiProtocol.poll(wifiClient, command)) handleCommand(command);
    }

    if (statePending && static_cast<uint32_t>(millis() - lastStateMs) >= STATE_RESPONSE_INTERVAL_MS) {
        sendState();
    }

    if (motionStopMs != 0 && static_cast<int32_t>(millis() - motionStopMs) >= 0) {
        motors.stop();
        motionStopMs = 0;
        sendState();
    }

    // Независимый watchdog: останавливает движение при пропаже координат.
    if (motors.motion() != Motion::STOPPED && boundary.needsEmergencyStop(millis())) {
        motors.stop();
        Serial.println("EVENT SAFETY_HALT POSITION_STALE_OR_OUTSIDE");
        response = &Serial;
        sendState();
        if (wifiClient && wifiClient.connected()) {
            response = &wifiClient;
            response->println("EVENT SAFETY_HALT POSITION_STALE_OR_OUTSIDE");
            sendState();
        }
    }
}

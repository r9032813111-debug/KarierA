#include "motor_driver.h"

namespace {
constexpr uint8_t LEFT_FORWARD_CHANNEL = 4;
constexpr uint8_t LEFT_REVERSE_CHANNEL = 5;
constexpr uint8_t RIGHT_FORWARD_CHANNEL = 6;
constexpr uint8_t RIGHT_REVERSE_CHANNEL = 7;
constexpr uint32_t MOTOR_PWM_FREQUENCY_HZ = 1000;
constexpr uint8_t MOTOR_PWM_RESOLUTION_BITS = 8;
}

MotorDriver::MotorDriver(uint8_t leftIn1, uint8_t leftIn2,
                         uint8_t rightIn1, uint8_t rightIn2)
    : leftIn1_(leftIn1), leftIn2_(leftIn2),
      rightIn1_(rightIn1), rightIn2_(rightIn2) {}

void MotorDriver::begin() {
    // Fixed channels 4..7 keep motor PWM away from the 50 Hz servo channels
    // allocated on timer 0 by ESP32Servo. Automatic analogWrite allocation can
    // otherwise reuse a servo channel and silently destroy its control pulse.
    ledcSetup(LEFT_FORWARD_CHANNEL, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_RESOLUTION_BITS);
    ledcSetup(LEFT_REVERSE_CHANNEL, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_RESOLUTION_BITS);
    ledcSetup(RIGHT_FORWARD_CHANNEL, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_RESOLUTION_BITS);
    ledcSetup(RIGHT_REVERSE_CHANNEL, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_RESOLUTION_BITS);
    ledcAttachPin(leftIn1_, LEFT_FORWARD_CHANNEL);
    ledcAttachPin(leftIn2_, LEFT_REVERSE_CHANNEL);
    ledcAttachPin(rightIn1_, RIGHT_FORWARD_CHANNEL);
    ledcAttachPin(rightIn2_, RIGHT_REVERSE_CHANNEL);
    stop();
}

void MotorDriver::setSpeed(uint8_t speed) { speed_ = speed; }
void MotorDriver::setDirectionInverted(bool left, bool right) {
    leftInverted_ = left;
    rightInverted_ = right;
}
uint8_t MotorDriver::speed() const { return speed_; }
int16_t MotorDriver::leftOutput() const { return leftOutput_; }
int16_t MotorDriver::rightOutput() const { return rightOutput_; }
Motion MotorDriver::motion() const { return motion_; }

int16_t MotorDriver::applyInversion(int16_t value, bool inverted) const {
    return inverted ? -value : value;
}

void MotorDriver::driveMotor(uint8_t channelForward, uint8_t channelReverse, int16_t value) {
    // Сначала выключаем оба входа. Это исключает короткий переходный импульс
    // при смене направления вращения.
    ledcWrite(channelForward, 0);
    ledcWrite(channelReverse, 0);
    if (value > 0) {
        ledcWrite(channelForward, static_cast<uint8_t>(value));
    } else if (value < 0) {
        ledcWrite(channelReverse, static_cast<uint8_t>(-value));
    }
}

void MotorDriver::move(Motion motion) {
    const int16_t pwm = speed_;
    switch (motion) {
        case Motion::FORWARD:
            driveRaw(pwm, pwm);
            break;
        case Motion::BACKWARD:
            driveRaw(-pwm, -pwm);
            break;
        case Motion::LEFT:
            driveRaw(-pwm, pwm);
            break;
        case Motion::RIGHT:
            driveRaw(pwm, -pwm);
            break;
        case Motion::STOPPED:
            stop();
            return;
    }
    motion_ = motion;
}

void MotorDriver::driveRaw(int16_t left, int16_t right) {
    left = constrain(left, -255, 255);
    right = constrain(right, -255, 255);
    leftOutput_ = applyInversion(left, leftInverted_);
    rightOutput_ = applyInversion(right, rightInverted_);
    driveMotor(LEFT_FORWARD_CHANNEL, LEFT_REVERSE_CHANNEL, leftOutput_);
    driveMotor(RIGHT_FORWARD_CHANNEL, RIGHT_REVERSE_CHANNEL, rightOutput_);
}

void MotorDriver::stop() {
    driveMotor(LEFT_FORWARD_CHANNEL, LEFT_REVERSE_CHANNEL, 0);
    driveMotor(RIGHT_FORWARD_CHANNEL, RIGHT_REVERSE_CHANNEL, 0);
    leftOutput_ = 0;
    rightOutput_ = 0;
    motion_ = Motion::STOPPED;
}

const char *MotorDriver::motionName(Motion motion) {
    switch (motion) {
        case Motion::FORWARD: return "FORWARD";
        case Motion::BACKWARD: return "BACKWARD";
        case Motion::LEFT: return "LEFT";
        case Motion::RIGHT: return "RIGHT";
        default: return "STOP";
    }
}

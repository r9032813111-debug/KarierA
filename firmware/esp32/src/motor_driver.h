#pragma once

#include <Arduino.h>

enum class Motion {
    STOPPED,
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class MotorDriver {
public:
    MotorDriver(uint8_t leftIn1, uint8_t leftIn2,
                uint8_t rightIn1, uint8_t rightIn2);

    void begin();
    void setSpeed(uint8_t speed);
    void setDirectionInverted(bool left, bool right);
    uint8_t speed() const;
    int16_t leftOutput() const;
    int16_t rightOutput() const;
    void move(Motion motion);
    // Diagnostic control: -255..255 for each motor, bypassing navigation only.
    void driveRaw(int16_t left, int16_t right);
    void stop();
    Motion motion() const;

    static const char *motionName(Motion motion);

private:
    void driveMotor(uint8_t channelForward, uint8_t channelReverse, int16_t value);
    int16_t applyInversion(int16_t value, bool inverted) const;

    uint8_t leftIn1_;
    uint8_t leftIn2_;
    uint8_t rightIn1_;
    uint8_t rightIn2_;
    uint8_t speed_ = 255;
    int16_t leftOutput_ = 0;
    int16_t rightOutput_ = 0;
    bool leftInverted_ = false;
    bool rightInverted_ = false;
    Motion motion_ = Motion::STOPPED;
};

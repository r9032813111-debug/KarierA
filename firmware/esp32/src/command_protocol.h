#pragma once

#include <Arduino.h>

enum class CommandType {
    NONE,
    HALT,
    STOP,
    RESUME,
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    DRIVE,
    MANUAL_DRIVE,
    SET_MOTOR_INVERSION,
    SET_SPEED,
    SET_BOUNDS,
    POSITION,
    SET_ARM,
    SET_BUCKET,
    TEST_LEFT,
    TEST_RIGHT,
    STATUS,
    INVALID
};

struct Command {
    CommandType type = CommandType::NONE;
    float values[5] = {0, 0, 0, 0, 0};
    uint8_t valueCount = 0;
    String error;
};

class CommandProtocol {
public:
    // Неблокирующее чтение одной строки из любого Arduino Stream.
    bool poll(Stream &stream, Command &command);
    void reset();

private:
    Command parseLine(char *line);
    char buffer_[128] = {};
    size_t length_ = 0;
};

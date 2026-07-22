#include "command_protocol.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

namespace {
bool parseFloatToken(char *token, float &value) {
    if (!token) return false;
    char *end = nullptr;
    value = strtof(token, &end);
    return end != token && *end == '\0';
}
}

bool CommandProtocol::poll(Stream &stream, Command &command) {
    while (stream.available()) {
        const char c = static_cast<char>(stream.read());
        if (c == '\r') continue;
        if (c == '\n') {
            if (length_ == 0) continue;
            buffer_[length_] = '\0';
            command = parseLine(buffer_);
            length_ = 0;
            return true;
        }
        if (length_ < sizeof(buffer_) - 1) {
            buffer_[length_++] = c;
        } else {
            length_ = 0;
            command.type = CommandType::INVALID;
            command.error = "LINE_TOO_LONG";
            return true;
        }
    }
    return false;
}

void CommandProtocol::reset() {
    length_ = 0;
}

Command CommandProtocol::parseLine(char *line) {
    Command result;
    char *save = nullptr;
    char *name = strtok_r(line, " \t", &save);
    if (!name) return result;
    for (char *p = name; *p; ++p) *p = static_cast<char>(toupper(*p));

    if (!strcmp(name, "HALT")) result.type = CommandType::HALT;
    else if (!strcmp(name, "STOP")) result.type = CommandType::STOP;
    else if (!strcmp(name, "RESUME")) result.type = CommandType::RESUME;
    else if (!strcmp(name, "FORWARD")) result.type = CommandType::FORWARD;
    else if (!strcmp(name, "BACKWARD")) result.type = CommandType::BACKWARD;
    else if (!strcmp(name, "LEFT")) result.type = CommandType::LEFT;
    else if (!strcmp(name, "RIGHT")) result.type = CommandType::RIGHT;
    else if (!strcmp(name, "DRIVE")) result.type = CommandType::DRIVE;
    else if (!strcmp(name, "MANUAL_DRIVE")) result.type = CommandType::MANUAL_DRIVE;
    else if (!strcmp(name, "SET_MOTOR_INVERSION")) result.type = CommandType::SET_MOTOR_INVERSION;
    else if (!strcmp(name, "SET_SPEED")) result.type = CommandType::SET_SPEED;
    else if (!strcmp(name, "SET_BOUNDS")) result.type = CommandType::SET_BOUNDS;
    else if (!strcmp(name, "POSITION") || !strcmp(name, "COMMAND_XY")) result.type = CommandType::POSITION;
    else if (!strcmp(name, "SET_ARM")) result.type = CommandType::SET_ARM;
    else if (!strcmp(name, "SET_BUCKET")) result.type = CommandType::SET_BUCKET;
    else if (!strcmp(name, "TEST_LEFT")) result.type = CommandType::TEST_LEFT;
    else if (!strcmp(name, "TEST_RIGHT")) result.type = CommandType::TEST_RIGHT;
    else if (!strcmp(name, "STATUS")) result.type = CommandType::STATUS;
    else {
        result.type = CommandType::INVALID;
        result.error = "UNKNOWN_COMMAND";
        return result;
    }

    char *token = nullptr;
    while ((token = strtok_r(nullptr, " \t", &save)) != nullptr) {
        if (result.valueCount >= 5 || !parseFloatToken(token, result.values[result.valueCount])) {
            result.type = CommandType::INVALID;
            result.error = "BAD_ARGUMENT";
            return result;
        }
        ++result.valueCount;
    }
    return result;
}

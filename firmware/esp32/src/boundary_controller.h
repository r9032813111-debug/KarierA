#pragma once

#include <Arduino.h>
#include "motor_driver.h"

enum class BoundaryStatus {
    NOT_CONFIGURED,
    POSITION_UNKNOWN,
    SAFE,
    NEAR_EDGE,
    OUTSIDE,
    STALE
};

class BoundaryController {
public:
    void setBounds(float minX, float minY, float maxX, float maxY, float margin = 10.0f);
    bool setPosition(float x, float y, float headingDegrees, uint32_t nowMs);
    bool isMotionAllowed(Motion motion, uint32_t nowMs);
    bool needsEmergencyStop(uint32_t nowMs) const;
    BoundaryStatus status(uint32_t nowMs) const;

    bool enabled() const { return enabled_; }
    bool hasPosition() const { return hasPosition_; }
    float x() const { return x_; }
    float y() const { return y_; }
    float heading() const { return headingDeg_; }
    static const char *statusName(BoundaryStatus status);

private:
    bool isOutside() const;
    bool positionIsFresh(uint32_t nowMs) const;

    float minX_ = 0;
    float minY_ = 0;
    float maxX_ = 0;
    float maxY_ = 0;
    float margin_ = 10;
    float x_ = 0;
    float y_ = 0;
    float headingDeg_ = 0;
    uint32_t lastPositionMs_ = 0;
    bool enabled_ = false;
    bool hasPosition_ = false;
    // Match the one-second marker hold used by the vision pipeline. A brief
    // missed ArUco frame may halt motion after this period, but is not an
    // operator emergency and must not latch the emergency state.
    static constexpr uint32_t POSITION_TIMEOUT_MS = 1000;
};

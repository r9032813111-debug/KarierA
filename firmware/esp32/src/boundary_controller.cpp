#include "boundary_controller.h"
#include <math.h>

void BoundaryController::setBounds(float minX, float minY, float maxX, float maxY, float margin) {
    minX_ = minX;
    minY_ = minY;
    maxX_ = maxX;
    maxY_ = maxY;
    margin_ = max(0.0f, margin);
    enabled_ = maxX_ > minX_ && maxY_ > minY_;
    hasPosition_ = false; // Требуем свежую позицию для новых границ.
}

bool BoundaryController::setPosition(float x, float y, float headingDegrees, uint32_t nowMs) {
    if (!isfinite(x) || !isfinite(y) || !isfinite(headingDegrees)) return false;
    x_ = x;
    y_ = y;
    headingDeg_ = headingDegrees;
    lastPositionMs_ = nowMs;
    hasPosition_ = true;
    return true;
}

bool BoundaryController::positionIsFresh(uint32_t nowMs) const {
    return hasPosition_ && (nowMs - lastPositionMs_ <= POSITION_TIMEOUT_MS);
}

bool BoundaryController::isOutside() const {
    return x_ < minX_ || x_ > maxX_ || y_ < minY_ || y_ > maxY_;
}

bool BoundaryController::isMotionAllowed(Motion motion, uint32_t nowMs) {
    if (motion == Motion::STOPPED || !enabled_) return true;
    if (!positionIsFresh(nowMs)) return false;

    // Turning in place is required to point back into the field, including
    // when vision reports that the center is already outside.
    if (motion == Motion::LEFT || motion == Motion::RIGHT) return true;

    // OpenCV обычно использует X вправо, Y вниз. heading=0 смотрит вправо,
    // heading=90 смотрит вниз. Для BACKWARD вектор инвертируется.
    const float radians = headingDeg_ * PI / 180.0f;
    float dx = cosf(radians);
    float dy = sinf(radians);
    if (motion == Motion::BACKWARD) {
        dx = -dx;
        dy = -dy;
    }

    // Outside recovery: permit translation only when its vector points back
    // through every violated side of the rectangle.
    if (isOutside()) {
        if (x_ < minX_ && dx <= 0) return false;
        if (x_ > maxX_ && dx >= 0) return false;
        if (y_ < minY_ && dy <= 0) return false;
        if (y_ > maxY_ && dy >= 0) return false;
        return true;
    }

    if (x_ <= minX_ + margin_ && dx < 0) return false;
    if (x_ >= maxX_ - margin_ && dx > 0) return false;
    if (y_ <= minY_ + margin_ && dy < 0) return false;
    if (y_ >= maxY_ - margin_ && dy > 0) return false;
    return true;
}

bool BoundaryController::needsEmergencyStop(uint32_t nowMs) const {
    // Direction filtering in isMotionAllowed() handles an outside position and
    // allows a controlled return. Loss of coordinates still means hard stop.
    return enabled_ && !positionIsFresh(nowMs);
}

BoundaryStatus BoundaryController::status(uint32_t nowMs) const {
    if (!enabled_) return BoundaryStatus::NOT_CONFIGURED;
    if (!hasPosition_) return BoundaryStatus::POSITION_UNKNOWN;
    if (!positionIsFresh(nowMs)) return BoundaryStatus::STALE;
    if (isOutside()) return BoundaryStatus::OUTSIDE;
    if (x_ <= minX_ + margin_ || x_ >= maxX_ - margin_ ||
        y_ <= minY_ + margin_ || y_ >= maxY_ - margin_) {
        return BoundaryStatus::NEAR_EDGE;
    }
    return BoundaryStatus::SAFE;
}

const char *BoundaryController::statusName(BoundaryStatus status) {
    switch (status) {
        case BoundaryStatus::NOT_CONFIGURED: return "DISABLED";
        case BoundaryStatus::POSITION_UNKNOWN: return "POSITION_UNKNOWN";
        case BoundaryStatus::SAFE: return "SAFE";
        case BoundaryStatus::NEAR_EDGE: return "NEAR_EDGE";
        case BoundaryStatus::OUTSIDE: return "OUTSIDE";
        case BoundaryStatus::STALE: return "STALE";
        default: return "UNKNOWN";
    }
}

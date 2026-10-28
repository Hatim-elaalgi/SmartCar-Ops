"""Our toy car as programmed in firmware/arduino_car/arduino_car.ino — keep the numbers below in sync with the sketch.

The dashboard uses them to drive the Prototype demo car like the real one, to hold the AVOIDING state for as long as
the real avoidance lasts, for the trip history, and on the Devices page.

What the sketch does:
  * line following with 2 IR sensors (1 = black tape): 00 / 11 -> forward, 01 -> turn right, 10 -> turn left;
    a turn stops one wheel, so on bends the car runs slower than on straight tape;
  * no "line lost" stop: with both sensors on the floor the car keeps going straight;
  * an obstacle closer than OBSTACLE_CM (HC-SR04) starts a timed detour, then the car drives back to the tape — it
    goes around obstacles, it never waits in front of them. The sketch turns left first; the dashboard draws the
    detour on the inside of the loop (on the right when driving it clockwise like the "Full loop", on the left
    anticlockwise like the "Reverse loop").
"""
import math

import config
from core import geo

# --- copied from the sketch ------------------------------------------------------------------------------------
MOTOR_PWM = 190            # MOTOR_SPEED: analogWrite() on IN2 / IN4 (ENA / ENB jumpers on), out of 255
OBSTACLE_CM = 13           # OBSTACLE_DISTANCE
AVOID_STEPS = [            # avoidObstacle(): (manoeuvre, duration in ms); then findLine() drives back to the tape
    ("rotate_left", 510),      # turn_1_time   — 90° left, right wheel only
    ("forward", 700),          # forward_1_time — step aside
    ("rotate_right", 350),     # turn_2_time   — 90° right, left wheel only
    ("forward", 600),          # forward_2_time — drive past the obstacle
    ("rotate_right", 330),     # turn_3_time   — 90° right, facing the tape again
]
# findLine() also pauses 200 ms (stopMotors) once a sensor sees the tape: left out here, the dashboard shows the car
# driving on without stopping.
PINS = {"IN1-IN4 (L298N)": "D8, D9, D10, D11", "IR left / right": "D2 / D3", "HC-SR04 Trig / Echo": "D12 / D13"}

# --- derived ---------------------------------------------------------------------------------------------------
SPEED_CM_S = config.PROTO_SPEED_CM_S       # straight tape at PWM 190 (estimate, see config.PROTO_SPEED_CM_S)
CURVE_FACTOR = 0.7                          # bends: forward / one-wheel turns alternate (a pivot moves at half speed)
DUTY = MOTOR_PWM / 255
WHEEL_TRACK_CM = 14.0      # distance between the two wheels (TT motor chassis, estimate)


def _pose(t, return_s):
    """Detour pose t seconds after it starts, in the tape's frame: (along, lateral to the left, heading in degrees
    with negative = turned left, speed). A sketch rotation stops one wheel, so the car pivots around that wheel: its
    centre moves on an arc of radius WHEEL_TRACK_CM / 2 at half speed (90° per rotation, as in the sketch)."""
    r = WHEEL_TRACK_CM / 2
    u = v = a = 0.0                 # a: heading in radians, 0 = along the tape, positive = turned left

    def left(angle):
        return -math.sin(angle), math.cos(angle)

    for kind, ms in AVOID_STEPS + [("find_line", return_s * 1000)]:
        d = ms / 1000
        k = min(1.0, max(0.0, t / d)) if d > 0 else 1.0
        if kind.startswith("rotate"):
            turn = math.radians(90) * k * (1 if kind == "rotate_left" else -1)
            side = 1 if kind == "rotate_left" else -1      # stopped wheel: left wheel for a left turn
            (lx0, ly0), (lx1, ly1) = left(a), left(a + turn)
            u, v, a = u + side * r * (lx0 - lx1), v + side * r * (ly0 - ly1), a + turn
        else:                                               # forward / find_line: straight at full speed
            u, v = u + SPEED_CM_S * d * k * math.cos(a), v + SPEED_CM_S * d * k * math.sin(a)
        if t < d:
            speed = SPEED_CM_S / 2 if kind.startswith("rotate") else SPEED_CM_S
            return u, v, -math.degrees(a), speed
        t -= d
    return u, v, 0.0, SPEED_CM_S


_, LATERAL_CM, _, _ = _pose(sum(ms for _, ms in AVOID_STEPS) / 1000, 0.0)   # off the tape before driving back
RETURN_S = LATERAL_CM / SPEED_CM_S                          # findLine(): straight back to the tape
AVOID_S = sum(ms for _, ms in AVOID_STEPS) / 1000 + RETURN_S
ALONG_CM = _pose(AVOID_S, RETURN_S)[0]                      # distance gained along the tape by the detour
# Mean speed during the detour (pivots move at half speed) and progress along the tape.
AVOID_MEAN_SPEED = SPEED_CM_S * (sum(ms for k, ms in AVOID_STEPS if k == "forward") / 1000 + RETURN_S
                                 + 0.5 * sum(ms for k, ms in AVOID_STEPS if k != "forward") / 1000) / AVOID_S
AVOID_PROGRESS_SPEED = ALONG_CM / AVOID_S


def avoid_pose(t):
    """Where the car is t seconds into the avoidance, relative to where it started: (distance gained along the
    tape, lateral offset to the left (the sketch's side), heading relative to the tape in degrees, current speed)."""
    if t >= AVOID_S:
        return ALONG_CM, 0.0, 0.0, SPEED_CM_S
    u, v, heading, speed = _pose(t, RETURN_S)
    return u, max(0.0, v), heading, speed


def track_point(points, cum, s, lateral=0.0):
    """(x, y, heading) at distance s along the tape, moved `lateral` cm to the left of the driving direction
    (negative = to the right)."""
    x, y, h = geo.point_along(config.PROTO, points, cum, s)
    r = math.radians(h)
    return x - lateral * math.cos(r), y + lateral * math.sin(r), h


def avoid_track(points, cum, s0, side, step=0.1):
    """Estimated path of a detour starting at distance s0 along the tape, on the given side (+1 left, -1 right):
    [(x, y), ...] from leaving the tape until back on it."""
    path, t = [], 0.0
    while t <= AVOID_S:
        along, lateral, _, _ = avoid_pose(t)
        x, y, _ = track_point(points, cum, min(cum[-1], s0 + along), lateral * side)
        path.append((round(x, 1), round(y, 1)))
        t += step
    along, _, _, _ = avoid_pose(AVOID_S)
    x, y, _ = track_point(points, cum, min(cum[-1], s0 + along))
    return path + [(round(x, 1), round(y, 1))]

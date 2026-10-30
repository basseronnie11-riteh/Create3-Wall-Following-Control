"""
Wall-Following Controller Benchmark - iRobot Create 3

Description:
    Experimental comparison of four wall-following controllers:
    PID, FLC, SMC, and Adaptive PID.

    The controllers are evaluated in three experimental scenarios:
    - Straight wall following
    - Obstacle avoidance
    - External disturbance rejection

    GTG/Bug mode switching is disabled during testing
    to enable direct comparison of wall-following performance.
"""

import math
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from irobot_create_msgs.msg import HazardDetection
from irobot_create_msgs.msg import HazardDetectionVector
from irobot_create_msgs.msg import IrIntensityVector
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data


def clamp(value, low, high):
    return max(low, min(high, value))


class PID:
    def __init__(self, kp, ki, kd, dt, integral_limit=80.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit
        self.prev_error = None
        self.integral = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / self.dt

        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.prev_error = None
        self.integral = 0.0


class AdaptivePID(PID):
    def __init__(self, kp, ki, kd, dt):
        super().__init__(kp, ki, kd, dt)
        self.base_kp = kp
        self.base_ki = ki
        self.base_kd = kd

    def compute(self, error):
        previous = 0.0 if self.prev_error is None else self.prev_error
        delta = error - previous

        if self.prev_error is not None and error * self.prev_error < 0.0:
            self.kp *= 0.96
            self.ki *= 0.98
            self.kd *= 1.03
        elif abs(error) > 24.0 and abs(delta) < 1.2:
            self.kp *= 1.006
            self.ki *= 1.002
            self.kd *= 0.998
        else:
            self.kp += 0.015 * (self.base_kp - self.kp)
            self.ki += 0.015 * (self.base_ki - self.ki)
            self.kd += 0.015 * (self.base_kd - self.kd)

        self.kp = clamp(self.kp, 0.004, 0.026)
        self.ki = clamp(self.ki, 0.0, 0.010)
        self.kd = clamp(self.kd, 0.0002, 0.006)
        return super().compute(error)

    def reset(self):
        super().reset()
        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd


class FuzzyController:
    def __init__(self, max_output):
        self.max_output = max_output
        self.prev_error = 0.0

    def triangular(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (c - x) / (c - b)

    def fuzzify(self, x):
        return {
            "NB": self.triangular(x, -1.25, -1.0, -0.42),
            "NS": self.triangular(x, -0.78, -0.34, 0.0),
            "ZE": self.triangular(x, -0.16, 0.0, 0.16),
            "PS": self.triangular(x, 0.0, 0.34, 0.78),
            "PB": self.triangular(x, 0.42, 1.0, 1.25),
        }

    def compute(self, error):
        error_delta = error - self.prev_error
        self.prev_error = error

        e = clamp(error / 55.0, -1.0, 1.0)
        de = clamp(error_delta / 26.0, -1.0, 1.0)

        e_sets = self.fuzzify(e)
        de_sets = self.fuzzify(de)
        outputs = {
            "NB": -1.0,
            "NS": -0.66,
            "ZE": 0.0,
            "PS": 0.66,
            "PB": 1.0,
        }

        numerator = 0.0
        denominator = 0.0
        for label, output in outputs.items():
            weight = e_sets[label] + 0.40 * de_sets[label]
            numerator += weight * output
            denominator += weight

        if denominator == 0.0:
            return 0.0
        return self.max_output * numerator / denominator

    def reset(self):
        self.prev_error = 0.0


class SMCController:
    def __init__(self, max_output, lambda_gain=0.72, boundary_layer=0.28):
        self.max_output = max_output
        self.lambda_gain = lambda_gain
        self.boundary_layer = boundary_layer
        self.prev_error = 0.0

    def compute(self, error):
        error_delta = error - self.prev_error
        self.prev_error = error

        e = clamp(error / 70.0, -1.0, 1.0)
        de = clamp(error_delta / 34.0, -1.0, 1.0)
        surface = self.lambda_gain * e + de
        return self.max_output * clamp(surface / self.boundary_layer, -1.0, 1.0)

    def reset(self):
        self.prev_error = 0.0


class WallFollowBenchmark(Node):
    DT = 0.02

    def __init__(self):
        super().__init__("create3_wall_follow_benchmark")

        qos_ir = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.publisher = self.create_publisher(Twist, "/create3/cmd_vel", 1)
        self.ir_sub = self.create_subscription(
            IrIntensityVector,
            "/create3/ir_intensity",
            self.ir_callback,
            qos_ir,
        )
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector,
            "/create3/hazard_detection",
            self.hazard_callback,
            qos_ir,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/create3/odom",
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(self.DT, self.control_loop)

        self.ir_raw = [0.0] * 7
        self.ir_filtered = [0.0] * 7
        self.ir_alpha = 0.35

        self.WALL_SETPOINT = 40.0 # Otprilike 15 cm
        self.WALL_DETECT = 13.0
        self.WALL_LOST = 7.0
        self.SIDE_CLOSE = 52.0
        self.SIDE_HARD = 76.0
        self.FRONT_WARN = 24.0
        self.FRONT_BLOCK = 38.0
        self.FRONT_PANIC = 76.0
        self.DEAD_END_FRONT = 34.0
        self.DEAD_END_SIDE = 38.0

        self.MAX_FORWARD_SPEED = 0.40 # Bilo je 30, a prije 23
        self.MAX_REVERSE_SPEED = 0.22
        self.MAX_ANGULAR_SPEED = 1.15

        self.MAX_WALL_SPEED = 0.40 # Bilo je 30
        self.MIN_WALL_SPEED = 0.045
        self.SEARCH_SPEED = 0.085
        self.OUTER_CORNER_SPEED = 0.085
        self.OUTER_CORNER_TURN = 0.50
        self.FRONT_AVOID_TURN = 0.88

        self.START_RAMP_TIME = 0.50 # Bilo je 0.75
        self.APPROACH_SLOWDOWN_DISTANCE = 0.30 # Bilo je 40 
        self.APPROACH_MIN_SPEED = 0.040
        self.ANGULAR_START_LIMIT = 0.35
        self.ANGULAR_APPROACH_LIMIT = 0.45

        self.OPEN_CORNER_LOST_TICKS = 5
        self.OUTER_CORNER_TIMEOUT = 2.2

        self.ESCAPE_BACKUP_SPEED = -0.20
        self.ESCAPE_BACKUP_DISTANCE = 0.30
        self.ESCAPE_BACKUP_TIMEOUT = 2.2
        self.ESCAPE_TURN_SPEED = 1.02
        self.ESCAPE_TURN_FORWARD = 0.045
        self.ESCAPE_TURN_ANGLE = 2.70
        self.ESCAPE_TURN_TIMEOUT = 3.2

        self.GOAL_TOLERANCE = 0.03
        self.PROGRESS_TOLERANCE = 0.02

        self.regulator_mode = "PID"
        self.wall_regulators = {
            "PID": {
                "LEFT": PID(0.010, 0.002, 0.0010, self.DT),
                "RIGHT": PID(0.010, 0.002, 0.0010, self.DT),
            },
            "Fuzzy": {
                "LEFT": FuzzyController(0.72),
                "RIGHT": FuzzyController(0.72),
            },
            "SMC": {
                "LEFT": SMCController(0.52),
                "RIGHT": SMCController(0.52),
            },
            "AdaptivePID": {
                "LEFT": AdaptivePID(0.010, 0.002, 0.0010, self.DT),
                "RIGHT": AdaptivePID(0.010, 0.002, 0.0010, self.DT),
            },
        }

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.test_length = 2.0
        self.test_start_time = 0.0

        self.is_running = False
        self.state = "IDLE"
        self.wall_side_choice = "AUTO"
        self.follow_side = "RIGHT"
        self.lost_wall_ticks = 0
        self.outer_corner_start_time = 0.0

        self.escape_phase = "NONE"
        self.escape_start_x = 0.0
        self.escape_start_y = 0.0
        self.escape_start_yaw = 0.0
        self.escape_turn_dir = 1
        self.escape_timeout = 0.0

        self.last_contact_hazard_time = 0.0
        self.last_safety_hazard_time = 0.0
        self.hazard_hold_time = 0.40

        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.max_linear_step = 0.010
        self.max_angular_step = 0.075

        self.debug_left = 0.0
        self.debug_front = 0.0
        self.debug_right = 0.0
        self.last_log_time = time.time()
        self.log_period = 0.70

    def ir_callback(self, msg):
        for i in range(7):
            value = float(msg.readings[i].value)
            self.ir_raw[i] = value
            self.ir_filtered[i] = (
                self.ir_alpha * value +
                (1.0 - self.ir_alpha) * self.ir_filtered[i]
            )

    def hazard_callback(self, msg):
        now = time.time()
        for detection in msg.detections:
            if detection.type in (HazardDetection.BUMP, HazardDetection.STALL):
                self.last_contact_hazard_time = now
            elif detection.type in (HazardDetection.CLIFF, HazardDetection.WHEEL_DROP):
                self.last_safety_hazard_time = now

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_yaw = self.euler_from_quaternion(qx, qy, qz, qw)

    def euler_from_quaternion(self, x, y, z, w):
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def set_regulator_mode(self, choice):
        lookup = {
            "1": "PID",
            "2": "Fuzzy",
            "3": "SMC",
            "4": "AdaptivePID",
            "pid": "PID",
            "fuzzy": "Fuzzy",
            "flc": "Fuzzy",
            "smc": "SMC",
            "adaptivepid": "AdaptivePID",
            "adaptivnipid": "AdaptivePID",
            "adaptivni pid": "AdaptivePID",
            "apid": "AdaptivePID",
        }
        key = choice.strip().lower()
        if key not in lookup:
            return False

        self.regulator_mode = lookup[key]
        self.reset_regulators()
        print(f"Odabrana regulacija: {self.regulator_mode}", flush=True)
        return True

    def set_wall_side(self, choice):
        lookup = {
            "1": "LEFT",
            "2": "RIGHT",
            "3": "AUTO",
            "l": "LEFT",
            "left": "LEFT",
            "lijevo": "LEFT",
            "d": "RIGHT",
            "r": "RIGHT",
            "right": "RIGHT",
            "desno": "RIGHT",
            "a": "AUTO",
            "auto": "AUTO",
        }
        key = choice.strip().lower()
        if key not in lookup:
            return False

        self.wall_side_choice = lookup[key]
        if self.wall_side_choice != "AUTO":
            self.follow_side = self.wall_side_choice
        print(f"Odabrana strana zida: {self.wall_side_choice}", flush=True)
        return True

    def start_test(self, length):
        self.test_length = max(0.20, float(length))
        self.start_x = self.current_x
        self.start_y = self.current_y
        self.start_yaw = self.current_yaw
        self.goal_x = self.start_x + self.test_length * math.cos(self.start_yaw)
        self.goal_y = self.start_y + self.test_length * math.sin(self.start_yaw)
        self.test_start_time = time.time()

        left, _, front, _, right = self.read_ir()
        if self.wall_side_choice == "AUTO":
            self.follow_side = self.auto_wall_side(left, right)

        self.is_running = True
        self.state = "SEARCH_WALL"
        self.escape_phase = "NONE"
        self.lost_wall_ticks = 0
        self.outer_corner_start_time = 0.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.reset_regulators()

        print(
            f"[START] length={self.test_length:.2f}m side={self.follow_side} "
            f"reg={self.regulator_mode} goal=({self.goal_x:.2f},{self.goal_y:.2f}) "
            f"frontIR={front:.0f}",
            flush=True,
        )

    def reset_regulators(self):
        for group in self.wall_regulators.values():
            for regulator in group.values():
                regulator.reset()

    def read_ir(self):
        left = self.ir_filtered[0]
        left_front = self.ir_filtered[1]
        front = max(self.ir_filtered[2], self.ir_filtered[3], self.ir_filtered[4])
        right_front = self.ir_filtered[5]
        right = self.ir_filtered[6]

        self.debug_left = left
        self.debug_front = front
        self.debug_right = right
        return left, left_front, front, right_front, right

    def wall_value(self, left, right):                                          # Strana praćenja
        return left if self.follow_side == "LEFT" else right

    def front_diag_value(self, left_front, right_front):
        return left_front if self.follow_side == "LEFT" else right_front

    def inner_corner_detected(self, left_front, front, right_front):            # Zatvoreni kut
        front_diag = self.front_diag_value(left_front, right_front)
        return front > self.FRONT_WARN and front_diag > self.SIDE_CLOSE

    def auto_wall_side(self, left, right):                                      # Auto strana zida
        if left > right + 8.0:
            return "LEFT"
        if right > left + 8.0:
            return "RIGHT"
        return "RIGHT"

    def contact_hazard_recent(self):                                            # Hazard kontakt 
        return time.time() - self.last_contact_hazard_time < self.hazard_hold_time

    def safety_hazard_recent(self):                                             # Hazard safety
        return time.time() - self.last_safety_hazard_time < self.hazard_hold_time

# --------------------------- Analitika puta --------------------------- #

    def progress_along_start_heading(self):
        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y
        return dx * math.cos(self.start_yaw) + dy * math.sin(self.start_yaw)

    def distance_to_goal(self):
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        return math.sqrt(dx * dx + dy * dy)

    def travelled_from(self, x, y):
        dx = self.current_x - x
        dy = self.current_y - y
        return math.sqrt(dx * dx + dy * dy)

    def angle_wrap(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    # Detekcija slijepe ulice
    def dead_end_detected(self, left, front, right):                            # Slijepa ulica Dead-end
        return (
            front > self.DEAD_END_FRONT and
            left > self.DEAD_END_SIDE and
            right > self.DEAD_END_SIDE
        )

    def test_finished(self):
        progress = self.progress_along_start_heading()
        if self.distance_to_goal() < self.GOAL_TOLERANCE:
            return True
        return progress >= self.test_length - self.PROGRESS_TOLERANCE

    def remaining_progress(self):
        return max(0.0, self.test_length - self.progress_along_start_heading())

# ------------------------------- Limit ------------------------------- #
    def publish_cmd(self, msg):
        msg.linear.x = clamp(msg.linear.x, -self.MAX_REVERSE_SPEED, self.MAX_FORWARD_SPEED)
        msg.angular.z = clamp(msg.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

        linear_delta = clamp(
            msg.linear.x - self.prev_linear,
            -self.max_linear_step,
            self.max_linear_step,
        )
        angular_delta = clamp(
            msg.angular.z - self.prev_angular,
            -self.max_angular_step,
            self.max_angular_step,
        )

        msg.linear.x = self.prev_linear + linear_delta
        msg.angular.z = self.prev_angular + angular_delta
        self.prev_linear = msg.linear.x
        self.prev_angular = msg.angular.z
        self.publisher.publish(msg)

    def forward_speed_limit(self):
        remaining = self.remaining_progress()
        elapsed = max(0.0, time.time() - self.test_start_time)

        start_factor = clamp(elapsed / self.START_RAMP_TIME, 0.0, 1.0)
        start_limit = self.APPROACH_MIN_SPEED + start_factor * (
            self.MAX_FORWARD_SPEED - self.APPROACH_MIN_SPEED
        )

        if remaining < self.APPROACH_SLOWDOWN_DISTANCE:
            approach_ratio = clamp(
                remaining / self.APPROACH_SLOWDOWN_DISTANCE,
                0.0,
                1.0,
            )
            approach_limit = self.APPROACH_MIN_SPEED + approach_ratio * (
                self.MAX_FORWARD_SPEED - self.APPROACH_MIN_SPEED
            )
        else:
            approach_limit = self.MAX_FORWARD_SPEED

        return min(start_limit, approach_limit)

    def angular_speed_limit(self, action):
        remaining = self.remaining_progress()
        elapsed = max(0.0, time.time() - self.test_start_time)

        start_factor = clamp(elapsed / self.START_RAMP_TIME, 0.0, 1.0)
        start_limit = self.ANGULAR_START_LIMIT + start_factor * (
            self.MAX_ANGULAR_SPEED - self.ANGULAR_START_LIMIT
        )

        if remaining < self.APPROACH_SLOWDOWN_DISTANCE:
            approach_ratio = clamp(
                remaining / self.APPROACH_SLOWDOWN_DISTANCE,
                0.0,
                1.0,
            )
            approach_limit = self.ANGULAR_APPROACH_LIMIT + approach_ratio * (
                self.MAX_ANGULAR_SPEED - self.ANGULAR_APPROACH_LIMIT
            )
        else:
            approach_limit = self.MAX_ANGULAR_SPEED

        limit = min(start_limit, approach_limit)
        if action in ("FRONT_AVOID", "SIDE_TOO_CLOSE", "OPEN_CORNER", "OUTER_CORNER"):
            limit = max(limit, 0.75)
        return limit
# ------------------------------- Brzine ------------------------------ #
    def apply_motion_profile(self, msg, action):
        if self.state == "ESCAPE":
            return

        if msg.linear.x > 0.0:
            msg.linear.x = min(msg.linear.x, self.forward_speed_limit())

        angular_limit = self.angular_speed_limit(action)
        msg.angular.z = clamp(msg.angular.z, -angular_limit, angular_limit)

    def wall_control(self, error):
        regulator = self.wall_regulators[self.regulator_mode][self.follow_side]     # Preciznije?
        limit = 0.52
        if self.regulator_mode == "Fuzzy":
            limit = 0.72
        elif self.regulator_mode == "SMC":
            limit = 0.58
        return clamp(regulator.compute(error), -limit, limit)

    def wall_speed(self, front, wall_value):                                        # Lin brzina pri regulaciji?
        speed = self.MAX_WALL_SPEED - 0.0022 * front
        if wall_value > self.SIDE_HARD:
            speed *= 0.42
        elif wall_value > self.SIDE_CLOSE:
            speed *= 0.68
        return clamp(speed, self.MIN_WALL_SPEED, self.MAX_WALL_SPEED)

    def turn_away_from_front(self):
        return self.FRONT_AVOID_TURN if self.follow_side == "RIGHT" else -self.FRONT_AVOID_TURN

    def turn_toward_wall(self):
        return -self.OUTER_CORNER_TURN if self.follow_side == "RIGHT" else self.OUTER_CORNER_TURN
# ------------------------------- STANJA ------------------------------ #
    # SEARCH režim
    def run_search_wall(self, msg, left, front, right):                             # WALL SEARCH
        if self.wall_side_choice == "AUTO":
            self.follow_side = self.auto_wall_side(left, right)

        if self.wall_value(left, right) > self.WALL_DETECT:
            self.state = "WALL_FOLLOW"
            self.lost_wall_ticks = 0
            self.reset_regulators()
            return self.run_wall_follow(msg, left, front, right)

        self.state = "SEARCH_WALL"
        msg.linear.x = self.SEARCH_SPEED
       # msg.angular.z = -0.22 if self.follow_side == "RIGHT" else 0.22
        msg.angular.z = 0.0
        return "SEARCH"

    def run_front_avoid(self, msg, front):                                          # FRONT AVOID
        self.state = "FRONT_AVOID"
        msg.linear.x = -0.035 if front > self.FRONT_PANIC else 0.025
        msg.angular.z = self.turn_away_from_front()
        return "FRONT_AVOID"

    def run_outer_corner(self, msg, left, left_front, front, right_front, right):   # OUTER CORNER
        wall = self.wall_value(left, right)
        if wall > self.WALL_DETECT:
            self.state = "WALL_FOLLOW"
            self.lost_wall_ticks = 0
            return self.run_wall_follow(msg, left, front, right)

        if front > self.FRONT_BLOCK or self.inner_corner_detected(left_front, front, right_front):
            avoid_value = max(front, 0.70 * self.front_diag_value(left_front, right_front))
            return self.run_front_avoid(msg, avoid_value)

        if time.time() - self.outer_corner_start_time > self.OUTER_CORNER_TIMEOUT:
            self.state = "SEARCH_WALL"
            return self.run_search_wall(msg, left, front, right)

        self.state = "OUTER_CORNER"
        msg.linear.x = self.OUTER_CORNER_SPEED
        msg.angular.z = self.turn_toward_wall()
        return "OUTER_CORNER"

    def run_wall_follow(self, msg, left, front, right):                             # WALL FOLLOW
        self.state = "WALL_FOLLOW"
        wall = self.wall_value(left, right)

        if wall < self.WALL_LOST:
            self.lost_wall_ticks += 1
            if self.lost_wall_ticks >= self.OPEN_CORNER_LOST_TICKS:
                self.state = "OUTER_CORNER"
                self.outer_corner_start_time = time.time()
                msg.linear.x = self.OUTER_CORNER_SPEED
                msg.angular.z = self.turn_toward_wall()
                return "OPEN_CORNER"

            msg.linear.x = 0.075
            msg.angular.z = self.turn_toward_wall()
            return "WALL_LOST"

        self.lost_wall_ticks = 0
        if self.follow_side == "RIGHT":
            error = wall - self.WALL_SETPOINT
        else:
            error = self.WALL_SETPOINT - wall

        msg.linear.x = self.wall_speed(front, wall)
        msg.angular.z = self.wall_control(error)

        if wall > self.SIDE_HARD:
            msg.linear.x = min(msg.linear.x, 0.060)
            if self.follow_side == "RIGHT":
                msg.angular.z = max(msg.angular.z, 0.42)
            else:
                msg.angular.z = min(msg.angular.z, -0.42)
            return "SIDE_TOO_CLOSE"

        return "FOLLOW"

    def choose_escape_turn_dir(self, left, right):
        if abs(left - right) > 8.0:
            return -1 if left > right else 1
        return 1 if self.follow_side == "RIGHT" else -1

    def start_escape(self, reason, left, front, right):
        self.state = "ESCAPE"
        self.escape_phase = "BACKUP"
        self.escape_start_x = self.current_x
        self.escape_start_y = self.current_y
        self.escape_start_yaw = self.current_yaw
        self.escape_turn_dir = self.choose_escape_turn_dir(left, right)
        self.escape_timeout = time.time() + self.ESCAPE_BACKUP_TIMEOUT
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        print(f"\n[ESCAPE] {reason}. IR={left:.0f}/{front:.0f}/{right:.0f}", flush=True)

def run_escape(self, msg, left, front, right):                                      # ESCAPE
        self.state = "ESCAPE"
        if self.escape_phase == "BACKUP":
            msg.linear.x = self.ESCAPE_BACKUP_SPEED
            msg.angular.z = 0.0

            backed_up = self.travelled_from(self.escape_start_x, self.escape_start_y) > self.ESCAPE_BACKUP_DISTANCE
            timed_out = time.time() > self.escape_timeout
            if backed_up or timed_out:
                self.escape_phase = "TURN"
                self.escape_start_yaw = self.current_yaw
                self.escape_turn_dir = self.choose_escape_turn_dir(left, right)
                self.escape_timeout = time.time() + self.ESCAPE_TURN_TIMEOUT
                self.prev_linear = 0.0
                self.prev_angular = 0.0
            return "ESC_BACKUP"

        if self.escape_phase == "TURN":
            turned = abs(self.angle_wrap(self.current_yaw - self.escape_start_yaw))
            front_safe = front < self.FRONT_BLOCK
            front_open = front < self.FRONT_WARN
            timed_out = time.time() > self.escape_timeout

            msg.linear.x = self.ESCAPE_TURN_FORWARD if front_open else 0.0
            msg.angular.z = self.ESCAPE_TURN_SPEED * self.escape_turn_dir

            if timed_out and not front_safe:
                self.escape_phase = "BACKUP"
                self.escape_start_x = self.current_x
                self.escape_start_y = self.current_y
                self.escape_timeout = time.time() + self.ESCAPE_BACKUP_TIMEOUT
                return "ESC_RETRY"

            if (turned > self.ESCAPE_TURN_ANGLE and front_safe) or timed_out:
                self.state = "SEARCH_WALL"
                self.escape_phase = "NONE"
                self.lost_wall_ticks = 0
                self.prev_linear = 0.0
                self.prev_angular = 0.0
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                return "ESC_DONE"

            return "ESC_TURN"

        self.state = "SEARCH_WALL"
        self.escape_phase = "NONE"
        return "ESC_RESET"

    def control_loop(self):
        if not self.is_running:
            return

        msg = Twist()
        left, left_front, front, right_front, right = self.read_ir()

        if self.safety_hazard_recent():
            self.stop_robot(msg, "[SAFETY] CLIFF/WHEEL_DROP")
            return

        inner_corner = self.inner_corner_detected(left_front, front, right_front)

        if self.state == "ESCAPE":
            action = self.run_escape(msg, left, front, right)
        elif self.contact_hazard_recent():
            self.start_escape("CONTACT", left, front, right)
            action = self.run_escape(msg, left, front, right)
        elif self.dead_end_detected(left, front, right):
            self.start_escape("DEAD_END", left, front, right)
            action = self.run_escape(msg, left, front, right)
        elif self.test_finished() and front < self.FRONT_BLOCK:
            progress = self.progress_along_start_heading()
            self.stop_robot(
                msg,
                f"[FINISH] progress={progress:.2f}/{self.test_length:.2f} "
                f"goal_dist={self.distance_to_goal():.2f}",
            )
            return
        elif self.state == "OUTER_CORNER":
            action = self.run_outer_corner(msg, left, left_front, front, right_front, right)
        elif front > self.FRONT_BLOCK or inner_corner:
            avoid_value = max(front, 0.70 * self.front_diag_value(left_front, right_front))
            action = self.run_front_avoid(msg, avoid_value)
        elif self.state == "WALL_FOLLOW":
            action = self.run_wall_follow(msg, left, front, right)
        else:
            action = self.run_search_wall(msg, left, front, right)

        self.apply_motion_profile(msg, action)
        self.publish_cmd(msg)
        self.log_state(msg, action)

    def stop_robot(self, msg, reason):
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.is_running = False
        self.state = "IDLE"
        self.escape_phase = "NONE"
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.publisher.publish(msg)
        print(f"\n{reason}", flush=True)

    def log_state(self, msg, action):
        now = time.time()
        if now - self.last_log_time < self.log_period:
            return

        progress = self.progress_along_start_heading()
        print(
            f"[{self.state}/{self.follow_side}/{self.regulator_mode}] "
            f"s={progress:.2f}/{self.test_length:.2f} "
            f"IR={self.debug_left:.0f}/{self.debug_front:.0f}/{self.debug_right:.0f} "
            f"v={msg.linear.x:.2f} w={msg.angular.z:.2f} {action}",
            flush=True,
        )
        self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowBenchmark()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    print("=== Create3 WF benchmark ===")

    try:
        while rclpy.ok():
            time.sleep(0.2)
            if node.is_running:
                time.sleep(0.5)
                continue

            print("\n-----------------------------------")
            try:
                print("Odaberite regulaciju: 1-PID, 2-FLC, 3-SMC, 4-APID")
                regulator_choice = input("Regulacija: ").strip() or "1"
                if not node.set_regulator_mode(regulator_choice):
                    print("Neispravan odabir regulacije.")
                    continue

                print("Odaberite stranu zida: 1-lijevo, 2-desno, 3-auto")
                side_choice = input("Strana zida: ").strip() or "3"
                if not node.set_wall_side(side_choice):
                    print("Neispravan odabir strane zida.")
                    continue

                length_text = input("Duljina testa u metrima: ").strip() or "2.0"
                test_length = float(length_text)
            except ValueError:
                print("Molim te unesi ispravan broj.")
                continue

            print("Postavi robota ispred prepreke ili uz zid. Krecem za 1 s...")
            time.sleep(1.0)
            node.start_test(test_length)

    except KeyboardInterrupt:
        print("\nZaustavljanje...")
    finally:
        stop_msg = Twist()
        node.publisher.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

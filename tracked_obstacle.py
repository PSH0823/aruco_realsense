import kalman_filter as kf
import numpy as np

class TrackedObstacle:
    def __init__(self, obstacle_pos):
        self.pos = obstacle_pos
        self.vel = np.zeros(3)
        # input: 0(none), output: 1(position), state: 2(position, velocity)
        self.kf_x = kf.KalmanFilter(0, 1, 2)
        self.kf_y = kf.KalmanFilter(0, 1, 2)
        self.kf_z = kf.KalmanFilter(0, 1, 2)
        self.initKF()
    
    def setSamplingTime(self, tp):
        self.sampling_time = tp

    def setCovariances(self, process_var, process_rate_var, measurement_var):
        self.process_variance = process_var
        self.process_rate_variance = process_rate_var
        self.measurement_variance = measurement_var

    def initKF(self):
        self.kf_x.A(0, 1) = self.sampling_time
        self.kf_y.A(0, 1) = self.sampling_time
        self.kf_z.A(0, 1) = self.sampling_time

        self.kf_x.C(0, 0) = 1.0
        self.kf_y.C(0, 0) = 1.0
        self.kf_z.C(0, 0) = 1.0

        self.kf_x.R(0, 0) = self.measurement_variance
        self.kf_y.R(0, 0) = self.measurement_variance
        self.kf_z.R(0, 0) = self.measurement_variance

        self.kf_x.Q(0, 0) = self.process_variance
        self.kf_y.Q(0, 0) = self.process_variance
        self.kf_z.Q(0, 0) = self.process_variance

        self.kf_x.Q(1, 1) = self.process_rate_variance
        self.kf_y.Q(1, 1) = self.process_rate_variance
        self.kf_z.Q(1, 1) = self.process_rate_variance

        self.kf_x.q_pred(0) = self.pos(0)
        self.kf_y.q_pred(0) = self.pos(1)
        self.kf_z.q_pred(0) = self.pos(2)

        self.kf_x.q_pred(1) = self.vel(0)
        self.kf_y.q_pred(1) = self.vel(1)
        self.kf_y.q_pred(1) = self.vel(2)

        self.kf_x.q_est(0) = self.pos(0)
        self.kf_y.q_est(0) = self.pos(1)
        self.kf_z.q_est(0) = self.pos(2)

        self.kf_x.q_est(1) = self.vel(0)
        self.kf_y.q_est(1) = self.vel(1)
        self.kf_y.q_est(1) = self.vel(2)

    def predictState(self):
        self.kf_x.predictState()
        self.kf_y.predictState()
        self.kf_z.predictState()

        self.pos(0) = self.kf_x.q_pred(0)
        self.pos(1) = self.kf_y.q_pred(0)
        self.pos(2) = self.kf_z.q_pred(0)

        self.vel(0) = self.kf_x.q_pred(1)
        self.vel(1) = self.kf_y.q_pred(1)
        self.vel(2) = self.kf_z.q_pred(1)

    def correctState(self, obstacle_pos):
        self.kf_x.y(0) = obstacle_pos(0)
        self.kf_y.y(0) = obstacle_pos(1)
        self.kf_z.y(0) = obstacle_pos(2)

        self.kf_x.correctState()
        self.kf_y.correctState()
        self.kf_z.correctState()

        self.pos(0) = self.kf_x.q_est(0)
        self.pos(1) = self.kf_y.q_est(0)
        self.pos(2) = self.kf_z.q_est(0)

        self.vel(0) = self.kf_x.q_est(1)
        self.vel(1) = self.kf_y.q_est(1)
        self.vel(2) = self.kf_z.q_est(1)

    def updateState(self):
        self.kf_x.predictState()
        self.kf_y.predictState()
        self.kf_z.predictState()

        self.kf_x.correctState()
        self.kf_y.correctState()
        self.kf_z.correctState()

        self.pos(0) = self.kf_x.q_est(0)
        self.pos(1) = self.kf_y.q_est(0)
        self.pos(2) = self.kf_z.q_est(0)

        self.vel(0) = self.kf_x.q_est(1)
        self.vel(1) = self.kf_y.q_est(1)
        self.vel(2) = self.kf_z.q_est(1)


import numpy as np
import kalman_filter as kf


class TrackedObstacle:
    # --- class-level (static) variables ---
    sampling_time = 0.01
    process_variance = 1.0
    process_rate_variance = 1.0
    measurement_variance = 1.0

    def __init__(self, obstacle_pos):
        self.pos = np.array(obstacle_pos, dtype=float)
        self.vel = np.zeros(3)

        # input: 0(none), output: 1(position), state: 2(position, velocity)
        self.kf_x = kf.KalmanFilter(0, 1, 2)
        self.kf_y = kf.KalmanFilter(0, 1, 2)
        self.kf_z = kf.KalmanFilter(0, 1, 2)

        self.initKF()

    # static-style setters
    @classmethod
    def setSamplingTime(cls, tp):
        cls.sampling_time = tp

    @classmethod
    def setCovariances(cls, process_var, process_rate_var, measurement_var):
        cls.process_variance = process_var
        cls.process_rate_variance = process_rate_var
        cls.measurement_variance = measurement_var

    # -------------------------------
    # Initialize Kalman filters
    # -------------------------------
    def initKF(self):
        dt = TrackedObstacle.sampling_time

        # A matrices (constant velocity model)
        self.kf_x.A[0, 1] = dt
        self.kf_y.A[0, 1] = dt
        self.kf_z.A[0, 1] = dt

        # C matrices (measure position only)
        self.kf_x.C[0, 0] = 1.0
        self.kf_y.C[0, 0] = 1.0
        self.kf_z.C[0, 0] = 1.0

        # R (measurement noise)
        mv = TrackedObstacle.measurement_variance
        self.kf_x.R[0, 0] = mv
        self.kf_y.R[0, 0] = mv
        self.kf_z.R[0, 0] = mv

        # Q (process noise)
        pv = TrackedObstacle.process_variance
        prv = TrackedObstacle.process_rate_variance

        self.kf_x.Q[0, 0] = pv
        self.kf_y.Q[0, 0] = pv
        self.kf_z.Q[0, 0] = pv

        self.kf_x.Q[1, 1] = prv
        self.kf_y.Q[1, 1] = prv
        self.kf_z.Q[1, 1] = prv

        # initial state = initial position + zero velocity
        self.kf_x.q_pred[0] = self.pos[0]
        self.kf_y.q_pred[0] = self.pos[1]
        self.kf_z.q_pred[0] = self.pos[2]

        self.kf_x.q_pred[1] = self.vel[0]
        self.kf_y.q_pred[1] = self.vel[1]
        self.kf_z.q_pred[1] = self.vel[2]

        # q_est = q_pred initially
        self.kf_x.q_est[:] = self.kf_x.q_pred[:]
        self.kf_y.q_est[:] = self.kf_y.q_pred[:]
        self.kf_z.q_est[:] = self.kf_z.q_pred[:]

    # -------------------------------
    # Predict step
    # -------------------------------
    def predictState(self):
        self.kf_x.predictState()
        self.kf_y.predictState()
        self.kf_z.predictState()

        self.pos[0] = self.kf_x.q_pred[0]
        self.pos[1] = self.kf_y.q_pred[0]
        self.pos[2] = self.kf_z.q_pred[0]

        self.vel[0] = self.kf_x.q_pred[1]
        self.vel[1] = self.kf_y.q_pred[1]
        self.vel[2] = self.kf_z.q_pred[1]

    # -------------------------------
    # Correct step
    # -------------------------------
    def correctState(self, obstacle_pos):
        self.kf_x.y[0] = obstacle_pos[0]
        self.kf_y.y[0] = obstacle_pos[1]
        self.kf_z.y[0] = obstacle_pos[2]

        self.kf_x.correctState()
        self.kf_y.correctState()
        self.kf_z.correctState()

        self.pos[0] = self.kf_x.q_est[0]
        self.pos[1] = self.kf_y.q_est[0]
        self.pos[2] = self.kf_z.q_est[0]

        self.vel[0] = self.kf_x.q_est[1]
        self.vel[1] = self.kf_y.q_est[1]
        self.vel[2] = self.kf_z.q_est[1]

    # -------------------------------
    # Full update
    # -------------------------------
    def updateState(self):
        self.predictState()
        self.correctState(self.pos)


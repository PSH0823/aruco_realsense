import numpy as np

class KalmanFilter:
    def __init__(self, dim_in, dim_out, dim_state):
        # dimensions:
        self.l = dim_in                     # input
        self.m = dim_out                    # output
        self.n = dim_state                  # state

        # systen matrices:
        self.A = np.eye(self.n)             # state
        self.B = np.zeros((self.n, self.l)) # input
        self.C = np.zeros((self.m, self.n)) # output

        # covariance matrices:
        self.Q = np.eye(self.n)             # process
        self.R = np.eye(self.m)             # measurement
        self.P = np.eye(self.n)             # estimate error

        # Kalman gain matrix
        self.K = np.eye(self.n, self.m)
        # identity matrix
        self.I = np.eye(self.n, self.n)
        
        # signals
        self.u = np.zeros(self.l)           # input
        self.q_pred = np.zeros(self.n)      # predicted state
        self.q_est = np.zeros(self.n)       # estimated state
        self.y = np.zeros(self.m)           # measurement
    
    def predictState(self):
        self.q_pred = self.A @ self.q_est + self.B @ self.u
        self.P = self.A @ self.P * self.A.T + self.Q

    def correctState(self):
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        self.q_est = self.q_pred + self.K @ (self.y - self.C @ self.q_pred)
        self.P = (self.I - self.K @ self.C) @ self.P
        
    def updateState(self):
        self.predictState()
        self.correctState()
    
        



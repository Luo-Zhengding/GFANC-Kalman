import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanFiltering:
    def __init__(self, dim_x=15, dim_z=15, F=None, H=None, Q=None, R=None, x=None, P=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        self.F = np.eye(dim_x) if F is None else F
        self.H = np.eye(dim_z, dim_x) if H is None else H
        self.kf.F = self.F  # State transition matrix
        self.kf.H = self.H  # Measurement matrix
        
        self.Q = np.eye(dim_x) * 0.5 if Q is None else Q # !!! Q and R can be changed
        self.R = np.eye(dim_z) if R is None else R
        self.kf.Q = self.Q  # Process noise matrix
        self.kf.R = self.R  # Measurement noise matrix

        self.x = np.zeros(dim_x) if x is None else x
        self.P = np.eye(dim_x) if P is None else P
        self.kf.x = self.x  # Predicted state vector
        self.kf.P = self.P  # Covariance matrix

    def update(self, z):
        # Calculate the Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # print("Kalman gain: ", K)
        
        self.kf.predict()
        self.kf.update(z)  # Update x and P using the observed vector z

    def get_state(self):
        return self.kf.x, self.kf.P  # x and P are updated
import numpy as np
import cv2

class KalmanAngleFilter:
    """
    1D Kalman filter for a single angle axis.
    
    State: [angle, angular_velocity]
    Measurement: angle only
    
    Tuning:
        process_noise  — how much we trust the motion model. Higher = faster
                         response to real motion, more noise passes through.
        measure_noise  — how much we trust the measurement. Higher = smoother
                         output but more lag.
    """
    def __init__(self, process_noise=1e-3, measure_noise=1e-1):
        self.kf = cv2.KalmanFilter(2, 1)  # 2 state vars, 1 measurement

        # State transition matrix [angle, velocity]
        # angle(t) = angle(t-1) + velocity(t-1) * dt
        # velocity(t) = velocity(t-1)
        self.kf.transitionMatrix = np.array([
            [1, 1],
            [0, 1]
        ], dtype=np.float32)

        # Measurement matrix — we only measure angle, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0]
        ], dtype=np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * process_noise

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.array([[measure_noise]], dtype=np.float32)

        # Initial error covariance
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)

        self.initialized = False

    def update(self, angle: float) -> float:
        measurement = np.array([[angle]], dtype=np.float32)

        if not self.initialized:
            # Seed the filter with the first measurement
            self.kf.statePost = np.array([[angle], [0.0]], dtype=np.float32)
            self.initialized = True
            return angle

        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0])

    def reset(self):
        self.initialized = False
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)


class PoseKalmanFilter:
    """
    Three independent Kalman filters, one per axis.
    This is what gets used in main.py.
    """
    def __init__(self, process_noise=1e-3, measure_noise=1e-1):
        self.roll_filter  = KalmanAngleFilter(process_noise, measure_noise)
        self.pitch_filter = KalmanAngleFilter(process_noise, measure_noise)
        self.yaw_filter   = KalmanAngleFilter(process_noise, measure_noise)

    def update(self, roll: float, pitch: float, yaw: float):
        roll_f  = self.roll_filter.update(roll)
        pitch_f = self.pitch_filter.update(pitch)
        yaw_f   = self.yaw_filter.update(yaw)
        return roll_f, pitch_f, yaw_f

    def reset(self):
        self.roll_filter.reset()
        self.pitch_filter.reset()
        self.yaw_filter.reset()

def rvec_to_euler(rvec):
    """
    Convert rvec to roll, pitch, yaw in degrees.
    Accounts for the upward-facing camera / downward-facing marker geometry.
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Correct for the 180-degree flip on X axis caused by the
    # camera-looking-up / marker-facing-down geometry
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)

    rotation_matrix = R_flip @ rotation_matrix

    # Extract Euler angles from corrected rotation matrix
    # Using atan2 directly is more stable than decomposeProjectionMatrix
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2( rotation_matrix[2, 1], rotation_matrix[2, 2]))
        roll  = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw   = np.degrees(np.arctan2( rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        roll  = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw   = 0.0

    return roll, pitch, yaw


def average_pose(rvecs, tvecs):
    """
    When multiple markers are detected, average their poses for a more
    stable estimate. This is key for our use case — the drone underside
    will have 4 markers, and averaging reduces noise significantly.
    """
    # Average translation vectors directly
    avg_tvec = np.mean(tvecs, axis=0)

    # For rotation, convert each rvec to a matrix, average, then convert back
    # Simple rvec averaging works for small angle differences between markers
    rotation_matrices = []
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(R)

    avg_R = np.mean(rotation_matrices, axis=0)

    # Re-orthogonalise the averaged matrix using SVD
    # (averaging rotation matrices can break orthogonality)
    U, _, Vt = np.linalg.svd(avg_R)
    avg_R_ortho = U @ Vt

    avg_rvec, _ = cv2.Rodrigues(avg_R_ortho)

    return avg_rvec, avg_tvec
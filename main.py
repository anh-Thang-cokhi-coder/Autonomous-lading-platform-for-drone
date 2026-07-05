import cv2
import numpy as np
import csv
import time
import os
import sys
import threading
import platform
from serial_output import SerialOutput

sys.path.append(os.path.dirname(__file__))
from pose_estimation.detector import load_calibration
from pose_estimation.angles import rvec_to_euler, average_pose, PoseKalmanFilter

ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_M = 0.08125  # 8.125 cm = 3.2 inches, adjust based on your printed marker size
CALIBRATION_FILE = "calibration/camera_params.npz"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Detect platform
IS_PI = os.path.exists('/proc/device-tree/model') and \
        'Raspberry Pi' in open('/proc/device-tree/model').read()


class CameraStreamWindows:
    """Windows camera stream using DirectShow backend."""
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_count += 1

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None), self.frame_count

    def get_resolution(self):
        return (1280, 720)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


class CameraStreamPi:
    """Raspberry Pi camera stream using picamera2 backend."""
    def __init__(self):
        from picamera2 import Picamera2
        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        self.picam.configure(config)
        self.picam.start()

        self.frame = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            frame = self.picam.capture_array()
            # picamera2 returns RGB, convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with self.lock:
                self.frame = frame_bgr
                self.frame_count += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.frame_count
            return True, self.frame.copy(), self.frame_count

    def get_resolution(self):
        return (1280, 720)

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam.stop()


def get_log_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"pose_log_{timestamp}.csv")


def draw_summary(frame, roll, pitch, yaw, n_markers,imu_roll, imu_pitch,imu_yaw, fps, logging_active):
    cv2.rectangle(frame, (20, 20), (520, 300), (0, 0, 0), -1)
    cv2.putText(frame, f"Markers: {n_markers}   FPS: {fps:.1f}",
        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Roll:  {roll:+.2f} deg",
        (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:+.2f} deg",
        (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)
    cv2.putText(frame, f"Yaw:   {yaw:+.2f} deg",
        (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame,
        f"IMU Roll : {imu_roll:+.2f}",
        (30,185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"IMU Pitch: {imu_pitch:+.2f}",
        (30,215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"IMU Yaw  : {imu_yaw:+.2f}",
        (30,245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )
    cv2.putText(frame, "Press Q to quit, L to toggle logging",
        (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if logging_active:
        cv2.circle(frame, (450, 40), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (465, 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main():
    camera_matrix, dist_coeffs = load_calibration()

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.minMarkerPerimeterRate = 0.03
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # Start the right camera stream for this platform
    if IS_PI:
        print("Raspberry Pi detected — using picamera2")
        stream = CameraStreamPi()
    else:
        print("Windows detected — using DirectShow")
        stream = CameraStreamWindows(0)

    print("Waiting for camera to warm up...")
    time.sleep(1.0)

    log_file = None
    log_writer = None
    logging_active = False

    last_frame_count = -1
    prev_time = time.time()
    fps = 0.0

    print("=== Drone Pose Estimator ===")
    print("Press Q to quit, L to start/stop logging to CSV")

    # Initialize the Kalman filter
    pose_filter = PoseKalmanFilter(process_noise=1e-3, measure_noise=1e-1)
    serial_out = SerialOutput()
    imu_roll= 0.0
    imu_pitch=0.0
    imu_yaw=0.0
    last_serial_time =0
    SERIAL_PERIOD=0.05
    while True:
        ret, frame, frame_count = stream.read()

        if not ret or frame is None:
            time.sleep(0.001)
            continue

        if frame_count == last_frame_count:
            time.sleep(0.001)
            continue
        last_frame_count = frame_count

        curr_time = time.time()
        elapsed = curr_time - prev_time
        imu_data= serial_out.receive()
        if imu_data:
            imu_roll, imu_pitch, imu_yaw =imu_data
            print(
            f"IMU: {imu_roll:.2f},"
            f"IMU: {imu_pitch:.2f},"
            f"IMU: {imu_yaw:.2f}"
            )
        fps = 0.1 * (1.0 / (elapsed + 1e-9)) + 0.9 * fps
        prev_time = curr_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        corners_small, ids, _ = detector.detectMarkers(small)
        corners = [c * 2 for c in corners_small] if corners_small else corners_small

        roll, pitch, yaw = 0.0, 0.0, 0.0
        n_markers = 0

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            n_markers = len(ids)

            all_rvecs = []
            all_tvecs = []

            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], MARKER_SIZE_M, camera_matrix, dist_coeffs
                )
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE_M * 0.5)
                all_rvecs.append(rvec.squeeze())
                all_tvecs.append(tvec.squeeze())

            avg_rvec, avg_tvec = average_pose(all_rvecs, all_tvecs)
            roll, pitch, yaw = rvec_to_euler(avg_rvec)
            roll, pitch, yaw = pose_filter.update(roll, pitch, yaw)
            if curr_time-last_serial_time>=SERIAL_PERIOD:
                serial_out.send(roll,pitch,yaw)
                last_serial_time=curr_time
                
            if logging_active and log_writer is not None:
                log_writer.writerow({
                    'timestamp': f"{curr_time:.4f}",
                    'n_markers': n_markers,
                    'roll': f"{roll:.4f}",
                    'pitch': f"{pitch:.4f}",
                    'yaw': f"{yaw:.4f}",
                    'fps': f"{fps:.2f}"
                })
        else:
            pose_filter.reset()
            serial_out.send_no_marker()
        draw_summary(frame, roll, pitch, yaw, n_markers,imu_roll,imu_pitch,imu_yaw , fps, logging_active)
        cv2.imshow("Drone Pose Estimator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            if not logging_active:
                log_path = get_log_filename()
                log_file = open(log_path, 'w', newline='')
                fieldnames = ['timestamp', 'n_markers', 'roll', 'pitch', 'yaw', 'fps']
                log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
                log_writer.writeheader()
                logging_active = True
                print(f"Logging started: {log_path}")
            else:
                logging_active = False
                if log_file:
                    log_file.close()
                    log_file = None
                print("Logging stopped.")

    serial_out.close()
    stream.stop()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pose_estimation.angles import rvec_to_euler, average_pose

ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_SIZE_M = 0.08  # update to your measured tablet marker size
CALIBRATION_FILE = "calibration/camera_params.npz"


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(
            f"Calibration file not found at {CALIBRATION_FILE}. "
            "Run calibration/calibrate.py first."
        )
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    print(f"Calibration loaded. RMS: {data['rms_error']:.4f} px")
    return camera_matrix, dist_coeffs


def draw_overlay(frame, marker_id, roll, pitch, yaw, distance, corner):
    """Draw angle readouts on the frame for a single marker."""
    x, y = int(corner[0]), int(corner[1])
    cv2.putText(frame, f"ID:{marker_id}",
        (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(frame, f"R:{roll:+.1f} P:{pitch:+.1f} Y:{yaw:+.1f} deg",
        (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"dist:{distance:.1f}cm",
        (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def draw_summary(frame, roll, pitch, yaw, n_markers):
    """Draw averaged pose summary in top-left corner."""
    cv2.rectangle(frame, (20, 20), (420, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Markers: {n_markers}",
        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Roll:  {roll:+.2f} deg",
        (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:+.2f} deg",
        (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)
    cv2.putText(frame, f"Yaw:   {yaw:+.2f} deg",
        (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def run_detector():
    camera_matrix, dist_coeffs = load_calibration()

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.minMarkerPerimeterRate = 0.03
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("=== ArUco Pose Detector ===")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            all_rvecs = []
            all_tvecs = []

            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], MARKER_SIZE_M, camera_matrix, dist_coeffs
                )
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE_M * 0.5)

                roll, pitch, yaw = rvec_to_euler(rvec)
                distance = np.linalg.norm(tvec) * 100
                corner = corners[i][0][0]

                draw_overlay(frame, marker_id, roll, pitch, yaw, distance, corner)

                all_rvecs.append(rvec.squeeze())
                all_tvecs.append(tvec.squeeze())

            # Averaged pose across all detected markers
            avg_rvec, avg_tvec = average_pose(all_rvecs, all_tvecs)
            avg_roll, avg_pitch, avg_yaw = rvec_to_euler(avg_rvec)
            draw_summary(frame, avg_roll, avg_pitch, avg_yaw, len(ids))

        else:
            cv2.putText(frame, "No markers detected",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("ArUco Pose Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()
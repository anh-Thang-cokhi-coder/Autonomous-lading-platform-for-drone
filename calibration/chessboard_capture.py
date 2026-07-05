import cv2
import numpy as np
import os

# ChArUco board parameters
SQUARES_X = 7        # number of squares horizontally
SQUARES_Y = 5        # number of squares vertically
SQUARE_LENGTH = 0.04 # metres — will be scaled on screen, exact value matters for calibration
MARKER_LENGTH = 0.02 # metres — must be less than square length
ARUCO_DICT = cv2.aruco.DICT_6X6_250

SAVE_DIR = "calibration/frames"
os.makedirs(SAVE_DIR, exist_ok=True)

def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    return board, dictionary

def generate_board_image():
    """Save the ChArUco board as an image you can open on your tablet."""
    board, _ = create_board()
    board_image = board.generateImage((1400, 1000), marginSize=20, borderBits=1)
    cv2.imwrite("calibration/charuco_board.png", board_image)
    print("Saved: calibration/charuco_board.png")
    print("Open this image fullscreen on your tablet.")
    cv2.imshow("ChArUco Board - open this on your tablet", board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture_frames():
    board, dictionary = create_board()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # Platform-aware camera setup
    IS_PI = os.path.exists('/proc/device-tree/model') and \
            'Raspberry Pi' in open('/proc/device-tree/model').read()

    if IS_PI:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        picam.configure(config)
        picam.start()
        import time
        time.sleep(1.0)

        def read_frame():
            frame = picam.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        def release():
            picam.stop()
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        def read_frame():
            return cap.read()

        def release():
            cap.release()

    saved_count = 0
    print("=== ChArUco Capture ===")
    print("Hold the printed ChArUco board in front of the camera.")
    print("Press SPACE to capture when markers are detected, Q to quit.")
    print("Aim for 20-30 frames from different angles and distances.")

    while True:
        ret, frame = read_frame()
        if not ret:
            print("Camera not accessible.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        corners, ids, rejected = detector.detectMarkers(gray)

        ret_charuco = 0
        charuco_corners = None
        charuco_ids = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            ret_charuco, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

            if ret_charuco and ret_charuco >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                cv2.putText(display,
                    f"Detected {ret_charuco} corners — press SPACE to capture",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(display,
                    f"Markers found but need more corners ({ret_charuco})",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            cv2.putText(display, "No markers detected",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {saved_count}",
            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("ChArUco Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if ids is not None and ret_charuco and ret_charuco >= 6:
                filename = os.path.join(SAVE_DIR, f"frame_{saved_count:03d}.png")
                cv2.imwrite(filename, frame)
                saved_count += 1
                print(f"Captured frame {saved_count}")
            else:
                print("Not enough corners detected — reposition and try again")
        elif key == ord('q'):
            break

    release()
    cv2.destroyAllWindows()
    print(f"Done. {saved_count} frames saved to {SAVE_DIR}/")

if __name__ == "__main__":
    # Step 1: generate the board image for your tablet
    generate_board_image()
    # Step 2: capture calibration frames
    capture_frames()
import cv2
import numpy as np
import os
import glob

# Must match exactly what you used in chessboard_capture.py
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.02
ARUCO_DICT = cv2.aruco.DICT_6X6_250

FRAMES_DIR = "calibration/frames"
OUTPUT_FILE = "calibration/camera_params.npz"

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    print(f"Found {len(frame_files)} frames to process...")

    for filepath in frame_files:
        frame = cv2.imread(filepath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if ret and ret >= 6:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                print(f"  {os.path.basename(filepath)}: {ret} corners detected ✓")
            else:
                print(f"  {os.path.basename(filepath)}: not enough corners, skipping")
        else:
            print(f"  {os.path.basename(filepath)}: no markers detected, skipping")

    print(f"\nUsable frames: {len(all_charuco_corners)}")

    if len(all_charuco_corners) < 5:
        print("Not enough usable frames for calibration. Capture more frames.")
        return

    print("Running calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )

    print(f"\nCalibration complete.")
    print(f"Reprojection error (RMS): {ret:.4f} px")
    print(f"\nCamera matrix:\n{camera_matrix}")
    print(f"\nDistortion coefficients:\n{dist_coeffs}")

    # Save to file
    np.savez(OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=ret
    )
    print(f"\nSaved to {OUTPUT_FILE}")

    # Quality check
    if ret < 1.0:
        print("Quality: GOOD — ready for pose estimation")
    elif ret < 2.0:
        print("Quality: ACCEPTABLE — will work but more varied frames would help")
    else:
        print("Quality: POOR — consider recapturing with more angle variety")

if __name__ == "__main__":
    main()
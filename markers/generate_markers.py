import cv2
import numpy as np
import os

# We use the 6x6 dictionary with 250 markers.
# 6x6 means each marker has a 6x6 grid of black/white cells.
# More cells = more robust detection at distance and angle.
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Marker size in pixels for display/export.
# On a tablet you want this large enough to fill most of the screen.
MARKER_SIZE_PX = 800

output_dir = "markers"

def generate_marker(marker_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_image = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE_PX, marker_image, 1)

    filename = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(filename, marker_image)
    print(f"Saved: {filename}")

    # Show it so you can display on tablet immediately
    cv2.imshow(f"ArUco Marker ID {marker_id} — press any key to close", marker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Generate markers 0, 1, 2, 3 — we'll use 4 markers on the drone underside
    for i in range(4):
        generate_marker(i)
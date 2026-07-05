import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check what resolution it actually opened at
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Current resolution: {width}x{height} @ {fps}fps")

# Try forcing 1080p
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"After setting 1080p: {width}x{height} @ {fps}fps")

# Also try setting the backend explicitly
cap.release()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow — Windows native
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"With DirectShow backend: {width}x{height}")

cap.release()
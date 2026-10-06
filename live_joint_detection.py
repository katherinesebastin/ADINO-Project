import cv2
from rtmlib import Body, draw_skeleton

body = Body(
    mode='lightweight',
    backend='openvino',
    device='cpu'
)

# Camera 0 = external camera
cam0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Camera 1 = laptop webcam
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Starting two-camera joint detection...")
print("Press Q to quit.")

while cam0.isOpened() and cam1.isOpened():

    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 or not ret1:
        print("Could not read from one of the cameras.")
        break

    # Detect joints
    keypoints0, scores0 = body(frame0)
    keypoints1, scores1 = body(frame1)

    # Draw skeletons
    frame0 = draw_skeleton(
        frame0,
        keypoints0,
        scores0,
        kpt_thr=0.3
    )

    frame1 = draw_skeleton(
        frame1,
        keypoints1,
        scores1,
        kpt_thr=0.3
    )

    # Show both cameras
    cv2.imshow("External Camera - Camera 0", frame0)
    cv2.imshow("Laptop Webcam - Camera 1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# Pose model
body = Body(
    mode='lightweight',
    backend='openvino',
    device='cpu'
)

# Camera 0 = external camera
cam0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Camera 1 = laptop webcam
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Calculate angle
def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle

# Draw angle next to joint
def draw_angle(
    frame,
    keypoints,
    scores,
    joint1,
    joint2,
    joint3,
    name
):

    # Check that all three joints exist
    if (
        joint1 >= len(keypoints) or
        joint2 >= len(keypoints) or
        joint3 >= len(keypoints)
    ):
        return

    # Check confidence
    if (
        scores[joint1] < 0.5 or
        scores[joint2] < 0.5 or
        scores[joint3] < 0.5
    ):
        return

    # Calculate angle
    angle = calculate_angle(
        keypoints[joint1],
        keypoints[joint2],
        keypoints[joint3]
    )

    if angle is None:
        return

    # Position of middle joint
    x = int(keypoints[joint2][0])
    y = int(keypoints[joint2][1])

    # Create label and draw angle near the joint
    text = f"{name}: {angle:.0f} °"

    text_position = (x + 10, y - 10)

    # Label colour
    cv2.putText(
        frame,
        text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# Draw all angles
def draw_all_angles(frame, keypoints, scores):
    # Left elbow
    # Shoulder -> Elbow -> Wrist
    draw_angle(
        frame,
        keypoints,
        scores,
        5,
        7,
        9,
        "L E"
    )

    # Right elbow
    # Shoulder -> Elbow -> Wrist
    draw_angle(
        frame,
        keypoints,
        scores,
        6,
        8,
        10,
        "R E"
    )

    # Left knee
    # Hip -> Knee -> Ankle
    draw_angle(
        frame,
        keypoints,
        scores,
        11,
        13,
        15,
        "L K"
    )

    # Right knee
    # Hip -> Knee -> Ankle
    draw_angle(
        frame,
        keypoints,
        scores,
        12,
        14,
        16,
        "R K"
    )

# Start
print("Starting two-camera joint detection...")
print("Press Q to quit.")

# Main loop
while cam0.isOpened() and cam1.isOpened():

    # Read cameras
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0 or not ret1:
        print("Could not read from one of the cameras.")
        break

    # Pose detection
    keypoints0, scores0 = body(frame0)
    keypoints1, scores1 = body(frame1)

    # Draw skeleton
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

    # Get first detected person
    if len(keypoints0) > 0:

    #   kp = keypoints0[0]
    #   sc = scores0[0]

        for person_idx in range(len(keypoints0)):
            kp0 = keypoints0[person_idx]
            sc0 = scores0[person_idx]

            draw_all_angles(
                frame0,
                kp0,
                sc0
            )


    if len(keypoints1) > 0:

        # kp1 = keypoints1[0]
        # sc1 = scores1[0] 

        for person_idx in range(len(keypoints1)):
            kp1 = keypoints1[person_idx]
            sc1 = scores1[person_idx]

            draw_all_angles(
                frame1,
                kp1,
                sc1
            )

    # Show cameras
    cv2.imshow(
        "External Camera",
        frame0
    )

    cv2.imshow(
        "Laptop Webcam",
        frame1
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam0.release()
cam1.release()

cv2.destroyAllWindows()
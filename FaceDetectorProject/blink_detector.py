import cv2
import time
import math
import os

import mediapipe as mp


# ============================================================
# 1. MODEL SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 2. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera.")
    landmarker.close()
    exit()


# ============================================================
# 3. BLINK SETTINGS
# ============================================================

# Eyes are considered CLOSED below this value.
CLOSED_THRESHOLD = 0.20

# Eyes are considered OPEN above this value.
# We keep it slightly higher than CLOSED_THRESHOLD
# to prevent flickering between OPEN and CLOSED.
OPEN_THRESHOLD = 0.23


# ============================================================
# 4. BLINK VARIABLES
# ============================================================

eyes_closed = False

eye_closed_start_time = None
blink_count = 0

last_closed_duration = 0.0


# ============================================================
# 5. DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# 6. EYE RATIO FUNCTION
# ============================================================

def calculate_eye_ratio(landmarks):

    # ---------------- LEFT EYE ----------------
    left_top = landmarks[159]
    left_bottom = landmarks[145]

    left_left = landmarks[33]
    left_right = landmarks[133]

    left_vertical = distance(left_top, left_bottom)
    left_horizontal = distance(left_left, left_right)

    left_ratio = left_vertical / left_horizontal


    # ---------------- RIGHT EYE ----------------
    right_top = landmarks[386]
    right_bottom = landmarks[374]

    right_left = landmarks[362]
    right_right = landmarks[263]

    right_vertical = distance(right_top, right_bottom)
    right_horizontal = distance(right_left, right_right)

    right_ratio = right_vertical / right_horizontal


    # Average both eyes
    average_ratio = (left_ratio + right_ratio) / 2

    return average_ratio


# ============================================================
# 7. MAIN CAMERA LOOP
# ============================================================

print("========================================")
print("       BLINK DETECTOR STARTED")
print("========================================")
print("Press Q to quit.")
print()


while True:

    success, frame = cap.read()

    if not success:
        print("❌ Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Convert frame to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect face landmarks
    result = landmarker.detect(mp_image)


    # ========================================================
    # 8. IF FACE FOUND
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # Calculate eye ratio
        eye_ratio = calculate_eye_ratio(landmarks)


        current_time = time.time()


        # ====================================================
        # 9. EYES OPEN → CLOSED
        # ====================================================

        if not eyes_closed:

            if eye_ratio < CLOSED_THRESHOLD:

                eyes_closed = True

                eye_closed_start_time = current_time


        # ====================================================
        # 10. EYES CLOSED → OPEN
        # ====================================================

        else:

            if eye_ratio > OPEN_THRESHOLD:

                eyes_closed = False


                # Calculate how long eyes were closed
                if eye_closed_start_time is not None:

                    last_closed_duration = (
                        current_time - eye_closed_start_time
                    )


                    # Count as a blink if eyes were
                    # closed for at least 0.08 seconds
                    if last_closed_duration >= 0.08:

                        blink_count += 1


                eye_closed_start_time = None


        # ====================================================
        # 11. CURRENT CLOSED TIME
        # ====================================================

        if eyes_closed:

            closed_duration = current_time - eye_closed_start_time

        else:

            closed_duration = 0


        # ====================================================
        # 12. DISPLAY STATUS
        # ====================================================

        if eyes_closed:

            eye_status = "CLOSED"

        else:

            eye_status = "OPEN"


        # ====================================================
        # 13. DRAW TEXT
        # ====================================================

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Eye Ratio: {eye_ratio:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        if eyes_closed:

            cv2.putText(
                frame,
                f"Closed: {closed_duration:.2f}s",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                f"Last Closed: {last_closed_duration:.2f}s",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # 14. DRAW EYE LANDMARKS
        # ====================================================

        eye_points = [
            159, 145,
            33, 133,
            386, 374,
            362, 263
        ]


        for index in eye_points:

            point = landmarks[index]

            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 255),
                -1
            )


    # ========================================================
    # 15. NO FACE FOUND
    # ========================================================

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 16. SHOW CAMERA
    # ========================================================

    cv2.imshow("Blink Detector", frame)


    # ========================================================
    # 17. QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


    # Also stop if the window is closed using X
    try:

        if cv2.getWindowProperty(
            "Blink Detector",
            cv2.WND_PROP_VISIBLE
        ) < 1:

            break

    except:

        break


# ============================================================
# 18. CLEANUP
# ============================================================

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print()
print("========================================")
print("Blink Detector stopped.")
print(f"Total blinks: {blink_count}")
print("========================================")
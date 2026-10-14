import cv2
import mediapipe as mp
import os
from datetime import datetime

# ==========================================
# CREATE SAVE FOLDER
# ==========================================

save_folder = "saved_faces"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="blaze_face_short_range.tflite"
    ),
    running_mode=RunningMode.VIDEO,
    min_detection_confidence=0.5
)

detector = FaceDetector.create_from_options(options)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

timestamp = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not working")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    # ==========================================
    # DETECT FACES
    # ==========================================

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    face_count = len(result.detections)


    # ==========================================
    # DRAW FACE BOXES
    # ==========================================

    for i, detection in enumerate(result.detections):

        box = detection.bounding_box

        x = box.origin_x
        y = box.origin_y
        w = box.width
        h = box.height

        # Make sure coordinates stay inside image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        # Green face box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Face number
        cv2.putText(
            frame,
            f"Face {i + 1}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ==========================================
    # FACE COUNT
    # ==========================================

    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Instructions
    cv2.putText(
        frame,
        "Press S = Save Faces",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q = Quit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "Multi Face Detector",
        frame
    )


    # ==========================================
    # KEYBOARD
    # ==========================================

    key = cv2.waitKey(1) & 0xFF


    # ==========================================
    # SAVE FACES
    # ==========================================

    if key == ord("s"):

        if face_count == 0:

            print("No faces detected!")

        else:

            current_time = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            for i, detection in enumerate(result.detections):

                box = detection.bounding_box

                x = box.origin_x
                y = box.origin_y
                w = box.width
                h = box.height

                # Keep coordinates inside image
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                # Crop face from original frame
                face_image = frame[y1:y2, x1:x2]

                if face_image.size == 0:
                    continue

                # File name
                filename = (
                    f"face_{i + 1}_{current_time}.jpg"
                )

                filepath = os.path.join(
                    save_folder,
                    filename
                )

                # Save image
                cv2.imwrite(
                    filepath,
                    face_image
                )

                print(
                    f"Saved Face {i + 1}: {filepath}"
                )

            print(
                f"{face_count} face(s) saved!"
            )


    # ==========================================
    # QUIT
    # ==========================================

    if key == ord("q") or cv2.getWindowProperty(
        "Multi Face Detector",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

detector.close()

cv2.destroyAllWindows()
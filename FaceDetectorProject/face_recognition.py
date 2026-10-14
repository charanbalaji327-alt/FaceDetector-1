import cv2
import mediapipe as mp
import numpy as np
import json
import os
import re
import time
import threading
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

FACE_DETECTOR_MODEL = "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "face_landmarker.task"

DATABASE_FILE = "face_database.json"
FACES_FOLDER = "registered_faces"

# Increased from 0.12 because expressions can change landmarks
RECOGNITION_THRESHOLD = 0.18

# How long an actually unknown person must remain unknown
# before registration starts
UNKNOWN_CONFIRMATION_TIME = 5.0

# Number of samples for a new person
NUMBER_OF_SAMPLES = 5

# Time between registration samples
SAMPLE_DELAY = 0.7

# How long recognition memory is kept
RECOGNITION_MEMORY_TIME = 2.0


# ============================================================
# CREATE FOLDER
# ============================================================

os.makedirs(FACES_FOLDER, exist_ok=True)


# ============================================================
# LOAD DATABASE
# ============================================================

if os.path.exists(DATABASE_FILE):

    try:
        with open(DATABASE_FILE, "r") as file:
            database = json.load(file)

    except Exception:
        database = {}

else:
    database = {}


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# ============================================================
# FACE DETECTOR
# ============================================================

detector_options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_DETECTOR_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

detector = FaceDetector.create_from_options(
    detector_options
)


# ============================================================
# FACE LANDMARKER
# ============================================================

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_LANDMARKER_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(
    landmarker_options
)


# ============================================================
# SAVE DATABASE
# ============================================================

def save_database():

    with open(DATABASE_FILE, "w") as file:

        json.dump(
            database,
            file,
            indent=4
        )


# ============================================================
# CLEAN NAME
# ============================================================

def clean_name(name):

    name = name.strip()

    name = re.sub(
        r'[<>:"/\\|?*]',
        '',
        name
    )

    return name


# ============================================================
# GET FACE EMBEDDING
# ============================================================

def get_face_embedding(frame, bbox):

    h, w, _ = frame.shape

    x = max(
        0,
        int(bbox.origin_x)
    )

    y = max(
        0,
        int(bbox.origin_y)
    )

    width = int(bbox.width)
    height = int(bbox.height)

    x2 = min(
        w,
        x + width
    )

    y2 = min(
        h,
        y + height
    )

    if x2 <= x or y2 <= y:
        return None

    face = frame[y:y2, x:x2]

    if face.size == 0:
        return None

    # BGR -> RGB
    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_face
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in landmarks
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # CENTER ON NOSE
    # --------------------------------------------------------

    nose = points[1].copy()

    points = points - nose

    # --------------------------------------------------------
    # SCALE USING EYE DISTANCE
    # --------------------------------------------------------

    left_eye = points[33]
    right_eye = points[263]

    eye_distance = np.linalg.norm(
        left_eye - right_eye
    )

    if eye_distance < 0.0001:
        return None

    points = points / eye_distance

    return points.flatten()


# ============================================================
# CALCULATE FACE DISTANCE
# ============================================================

def calculate_distance(
    embedding1,
    embedding2
):

    a = np.array(
        embedding1,
        dtype=np.float32
    )

    b = np.array(
        embedding2,
        dtype=np.float32
    )

    if a.shape != b.shape:
        return float("inf")

    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)

    distances = np.linalg.norm(
        a - b,
        axis=1
    )

    return float(
        np.mean(distances)
    )


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_face(embedding):

    if embedding is None:

        return None, float("inf")

    best_name = None

    best_distance = float("inf")

    # --------------------------------------------------------
    # Compare against EVERY saved sample
    # --------------------------------------------------------

    for name, person_data in database.items():

        samples = person_data.get(
            "embeddings",
            []
        )

        for stored_embedding in samples:

            distance = calculate_distance(
                embedding,
                stored_embedding
            )

            if distance < best_distance:

                best_distance = distance

                best_name = name

    # --------------------------------------------------------
    # Recognition threshold
    # --------------------------------------------------------

    if best_distance <= RECOGNITION_THRESHOLD:

        return (
            best_name,
            best_distance
        )

    return (
        None,
        best_distance
    )


# ============================================================
# ASK FOR NAME
# ============================================================

entered_name = None
name_input_finished = False


def ask_for_name():

    global entered_name
    global name_input_finished

    print()
    print("=" * 55)
    print("NEW PERSON CONFIRMED")
    print("=" * 55)

    name = input(
        "Enter the person's name: "
    )

    name = clean_name(name)

    entered_name = name

    name_input_finished = True


# ============================================================
# START NAME THREAD
# ============================================================

def start_name_thread():

    global entered_name
    global name_input_finished

    entered_name = None

    name_input_finished = False

    thread = threading.Thread(
        target=ask_for_name,
        daemon=True
    )

    thread.start()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open camera.")

    detector.close()

    landmarker.close()

    exit()


# ============================================================
# REGISTRATION STATE
# ============================================================

registration_state = "normal"

# normal
# waiting_for_name
# collecting


registration_name = None

registration_samples = []

registration_photos = []

last_sample_time = 0


# ============================================================
# UNKNOWN FACE TRACKING
# ============================================================

unknown_start_time = None

unknown_face_active = False


# ============================================================
# LAST RECOGNIZED PERSON
# ============================================================

last_known_name = None

last_known_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    # --------------------------------------------------------
    # Convert to MediaPipe image
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    detection_result = detector.detect(
        mp_image
    )

    current_known_found = False
    current_unknown_found = False

    first_unknown_bbox = None

    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    if detection_result.detections:

        for detection in detection_result.detections:

            bbox = detection.bounding_box

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)

            width = int(bbox.width)
            height = int(bbox.height)

            x2 = x + width
            y2 = y + height

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )

            # ------------------------------------------------
            # Calculate face embedding
            # ------------------------------------------------

            embedding = get_face_embedding(
                frame,
                bbox
            )

            name, distance = recognize_face(
                embedding
            )

            # =================================================
            # KNOWN FACE
            # =================================================

            if name is not None:

                current_known_found = True

                # Remember this person
                last_known_name = name

                last_known_time = time.time()

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Hello, {name}",
                    (
                        x,
                        max(
                            30,
                            y - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Show distance for debugging
                cv2.putText(
                    frame,
                    f"Match: {distance:.3f}",
                    (
                        x,
                        y2 + 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # =================================================
            # UNKNOWN FACE
            # =================================================

            else:

                current_unknown_found = True

                if first_unknown_bbox is None:

                    first_unknown_bbox = bbox

                # ------------------------------------------------
                # IMPORTANT:
                #
                # If this person was JUST recognized,
                # don't immediately call them unknown.
                # ------------------------------------------------

                time_since_known = (
                    time.time()
                    - last_known_time
                )

                if (
                    last_known_name is not None
                    and
                    time_since_known
                    <= RECOGNITION_MEMORY_TIME
                ):

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Hello, {last_known_name}",
                        (
                            x,
                            max(
                                30,
                                y - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Unknown",
                        (
                            x,
                            max(
                                30,
                                y - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

    # ========================================================
    # NORMAL STATE
    # ========================================================

    if registration_state == "normal":

        # ----------------------------------------------------
        # Only start unknown timer if we DON'T have
        # recent recognition memory.
        # ----------------------------------------------------

        recent_known = (
            last_known_name is not None
            and
            time.time() - last_known_time
            <= RECOGNITION_MEMORY_TIME
        )

        if current_unknown_found and not recent_known:

            if not unknown_face_active:

                unknown_face_active = True

                unknown_start_time = time.time()

            unknown_time = (
                time.time()
                - unknown_start_time
            )

            remaining = max(
                0,
                UNKNOWN_CONFIRMATION_TIME
                - unknown_time
            )

            cv2.putText(
                frame,
                f"Unknown confirmation: {remaining:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            # ------------------------------------------------
            # Only register after LONG confirmation
            # ------------------------------------------------

            if unknown_time >= UNKNOWN_CONFIRMATION_TIME:

                registration_state = (
                    "waiting_for_name"
                )

                start_name_thread()

                unknown_face_active = False

                unknown_start_time = None

        else:

            unknown_face_active = False

            unknown_start_time = None

    # ========================================================
    # WAITING FOR NAME
    # ========================================================

    elif registration_state == "waiting_for_name":

        cv2.putText(
            frame,
            "Enter name in terminal...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Camera is still running",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Name entered
        # ----------------------------------------------------

        if name_input_finished:

            if entered_name:

                if entered_name in database:

                    print()
                    print(
                        f"{entered_name} already exists."
                    )

                    registration_state = "normal"

                else:

                    registration_name = entered_name

                    registration_samples = []

                    registration_photos = []

                    last_sample_time = 0

                    registration_state = "collecting"

            else:

                print(
                    "No valid name entered."
                )

                registration_state = "normal"

    # ========================================================
    # COLLECTING SAMPLES
    # ========================================================

    elif registration_state == "collecting":

        cv2.putText(
            frame,
            f"Registering: {registration_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Samples: "
                f"{len(registration_samples)}/"
                f"{NUMBER_OF_SAMPLES}"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Change your expression slightly",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Detect a face
        # ----------------------------------------------------

        if detection_result.detections:

            current_detection = (
                detection_result.detections[0]
            )

            current_bbox = (
                current_detection.bounding_box
            )

            current_embedding = (
                get_face_embedding(
                    frame,
                    current_bbox
                )
            )

            current_time = time.time()

            # ------------------------------------------------
            # Collect sample
            # ------------------------------------------------

            if (
                current_embedding is not None
                and
                current_time - last_sample_time
                >= SAMPLE_DELAY
            ):

                registration_samples.append(
                    current_embedding.tolist()
                )

                # ------------------------------------------------
                # Save face photo
                # ------------------------------------------------

                x = max(
                    0,
                    int(current_bbox.origin_x)
                )

                y = max(
                    0,
                    int(current_bbox.origin_y)
                )

                x2 = min(
                    frame.shape[1],
                    x + int(current_bbox.width)
                )

                y2 = min(
                    frame.shape[0],
                    y + int(current_bbox.height)
                )

                face_crop = frame[
                    y:y2,
                    x:x2
                ]

                if face_crop.size > 0:

                    timestamp = (
                        datetime.now()
                        .strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )
                    )

                    filename = (
                        f"{registration_name}_"
                        f"sample_"
                        f"{len(registration_samples)}_"
                        f"{timestamp}.jpg"
                    )

                    filepath = os.path.join(
                        FACES_FOLDER,
                        filename
                    )

                    cv2.imwrite(
                        filepath,
                        face_crop
                    )

                    registration_photos.append(
                        filepath
                    )

                last_sample_time = current_time

        # ====================================================
        # FINISHED REGISTRATION
        # ====================================================

        if len(registration_samples) >= NUMBER_OF_SAMPLES:

            database[registration_name] = {

                "embeddings":
                    registration_samples,

                "photos":
                    registration_photos,

                "registered_at":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
            }

            save_database()

            print()
            print("=" * 55)
            print(
                f"{registration_name} "
                f"registered successfully!"
            )
            print(
                f"{NUMBER_OF_SAMPLES} samples saved."
            )
            print("=" * 55)
            print()

            # Reset
            registration_state = "normal"

            registration_name = None

            registration_samples = []

            registration_photos = []

            unknown_start_time = None

            unknown_face_active = False

            # Give the newly registered person
            # recognition memory
            last_known_name = entered_name

            last_known_time = time.time()


    # ========================================================
    # QUIT TEXT
    # ========================================================

    cv2.putText(
        frame,
        "Q = Quit",
        (
            20,
            frame.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Face Recognition",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

    # X button
    if cv2.getWindowProperty(
        "Face Recognition",
        cv2.WND_PROP_VISIBLE
    ) < 1:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

landmarker.close()

cv2.destroyAllWindows()
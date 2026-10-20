
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Find the AI model in this project's folder
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Place face_landmarker.task beside this Python file."
    )

# Set up MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)

# Landmark indices for drawing the eye and lip outlines
LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]
RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]
OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270,
    409, 291, 375, 321, 405, 314, 17, 84,
    181, 91, 146
]
INNER_LIPS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310,
    415, 308, 324, 318, 402, 317, 14, 87,
    178, 88, 95
]


def draw_landmark_group(frame, landmarks, indices, color):
    height, width = frame.shape[:2]
    points = []

    for index in indices:
        point = landmarks[index]
        x = int(point.x * width)
        y = int(point.y * height)
        points.append((x, y))

    for point in points:
        cv2.circle(frame, point, 2, color, -1)

    if len(points) > 2:
        import numpy as np
        cv2.polylines(
            frame,
            [np.array(points, dtype="int32")],
            True,
            color,
            1,
            cv2.LINE_AA
        )


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    start_time = time.monotonic()
    previous_time_ms = -1

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                # Mirror the image for a natural webcam preview
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp_ms = max(
                    timestamp_ms, previous_time_ms + 1
                )
                previous_time_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    # Eye outlines
                    draw_landmark_group(
                        frame, landmarks, LEFT_EYE, (0, 255, 0)
                    )
                    draw_landmark_group(
                        frame, landmarks, RIGHT_EYE, (0, 255, 0)
                    )

                    # Lip outline
                    draw_landmark_group(
                        frame, landmarks, OUTER_LIPS, (0, 165, 255)
                    )

                    draw_landmark_group(
                        frame, landmarks, INNER_LIPS, (255, 0, 255)
                    )

                    # A small status label
                    cv2.putText(
                        frame,
                        "Face tracked",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("AI Face Detector", frame)

                # Press Esc to exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

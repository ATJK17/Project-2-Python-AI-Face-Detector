
# pip install opencv-python

import cv2

a = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if a.empty():
    raise RuntimeError("Could not load the face detection model.")

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if eye_cascade.empty():
    raise RuntimeError("Could not load the eye detection model.")

def detect_eyes(gray_face, color_frame, x_offset, y_offset):
    eye_rect = eye_cascade.detectMultiScale(
        gray_face,  # <-- Missing argument!
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
    )

    for (x, y, w, h) in eye_rect:
        cv2.rectangle(
            color_frame,
            (x + x_offset, y + y_offset),
            (x + w + x_offset, y + h + y_offset),
            (0, 255, 0),
            2
        )

b = cv2.VideoCapture(0)

if not b.isOpened():
    raise RuntimeError("Could not open the webcam.")

while True:
    c_Rectangle, d_Image = b.read()

    if not c_Rectangle:
        print("Could not read a webcam frame.")
        break

    e = cv2.cvtColor(d_Image, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        cv2.rectangle(
            d_Image,
            (x1, y1),
            (x1 + w1, y1 + h1),
            (255, 0, 0),
            5
        )

        gray_face = e[y1:y1 + h1, x1:x1 + w1]

        detect_eyes(
            gray_face,
            d_Image,
            x1,
            y1
        )

    

    cv2.imshow("img", d_Image)
    h = cv2.waitKey(40) & 0xFF

    if h == 27:  # Press Esc to quit
        break


b.release()
cv2.destroyAllWindows()

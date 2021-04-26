# Importing the OpenCV Library :
import cv2

# Importing the Trained Dataset :
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    sfr, frame = webcam.read()
    # Changing the Color Format to Grayscale :
    grayScaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Taking The Face Coordinates :
    fc = trained_face_data.detectMultiScale(grayScaled_img)

    # Draw The Rectangle :
    for (x, y, w, h) in fc:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Camera Viewer", frame)
    key = cv2.waitKey(1)

    # Stop Key :
    if key == 81 or key == 113:
        break

webcam.release()

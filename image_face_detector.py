# Importing the OpenCV Library :
import cv2

# Importing the Trained Dataset :
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Reading the Image :
img = cv2.imread('assets/hc.jpg')

# Changing the Color Format to Grayscale :
grayScaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Taking The Face Coordinates :
fc = trained_face_data.detectMultiScale(grayScaled_img)

# Draw The Rectangle :
for (x, y, w, h) in fc:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Image Viewer", img)

cv2.waitKey()

print("Code Completed")

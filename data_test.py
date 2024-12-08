import cv2  # Install opencv-python
import copy
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1)
offset = 20
x, y = 10, 15

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    img_show = copy.deepcopy(image)

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    hands, img = detector.findHands(img_show)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        cv2.rectangle(img_show, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
    # Show the image in a window
    cv2.putText(img_show, f"{class_name[2:-1]} - CS: {np.round(confidence_score * 100)}%", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    cv2.imshow("Webcam Image", img_show)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

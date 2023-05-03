import sys
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
import imutils
from threading import Thread
from pygame import mixer
mixer.init()

# parameters for loading data and images

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

count = 0

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
vs = cv2.VideoCapture(0)

def play_out(file):
        try:
            print(file)
            mixer.music.load("songs/"+ file +".mp3") #Loading Music File
            mixer.music.play() #Playing Music with Pygame
        except:
                pass
        
def audioalert(file):
            t = Thread(target = play_out,args=[file])
            t.deamon = True
            t.start()
    
while True:
    # loading images
    grab,frame = vs.read()
    bgr_image = frame
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    color = (0,0,255)
    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        count = count + 1
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    try:
        if count % 25 == 0:
            count = 0
            audioalert(emotion_text)
    except:
        pass
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        mixer.music.stop()
    
vs.release()
print("[INFO] Stopped \n")
cv2.destroyAllWindows()

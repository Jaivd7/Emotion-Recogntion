# %%
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import cv2
import time
import playsound


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

# %%
class Model(object):

    emotions = ["Angry", "Disgust","Fear", "Happy","Neutral", "Sad","Surprise"]
    audio=["Tracks/Angry.wav","Tracks/Disgust.wav","Tracks/Fear.wav","Tracks/Happy.wav","Tracks/Neutral.wav","Tracks/Sad.wav","Tracks/Surprise.wav"]

    def __init__(self, model_json_file, model_weights_file):
        with open("m.json", "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("mw.h5")

    def emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return Model.emotions[np.argmax(self.preds)]

    def play(self,img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        a = np.argmax(self.preds)
        playsound.playsound(Model.audio[a],True)
        return;

# %%
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = Model("m.json", "mw.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture('Retro.mp4') #Put a the full fie name of the file you would like to scan in quotes inside the brackets or leave it as 0 if you want to use the webcam
now = time.time()
future = now + 5

# %%
while True:
    ret,f=cap.read()
    if not ret:
        continue
    grayscale = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    face = haar.detectMultiScale(grayscale, 1.3, 5)
    for (x, y, w, h) in face:
        fc = grayscale[y:y+h, x:x+w]
        image = cv2.resize(fc, (48, 48))
        emotion = model.emotion(image[np.newaxis, :, :, np.newaxis])

        cv2.putText(f, emotion, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)


    _, jpeg = cv2.imencode('.jpg', f)

    resized_img = cv2.resize(f, (1000, 700))
    cv2.imshow('Retro',resized_img)
    if time.time() > future:
        if bool(len(face))==True:
            fc = grayscale[y:y+h, x:x+w]
            image = cv2.resize(fc, (48, 48))
            model.play((image[np.newaxis, :, :, np.newaxis]))
        future=future+5

    if cv2.waitKey(10) == ord('0'): #Quits the program once 0 is clicked
        break

cap.release()
cv2.destroyAllWindows


# %%


# %%

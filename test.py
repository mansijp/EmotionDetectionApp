from flask import Flask, render_template, Response
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./CPS843_Emotion_Detection_Model.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Global flag to control webcam streaming
streaming = False

def generate_frames():
    global streaming
    while True:
        if streaming:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # Loop over the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Make a prediction on the ROI
                    preds = classifier.predict(roi)[0]
                    label = class_labels[preds.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Convert frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Yield the frame to be sent to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index.html')

@app.route('/start')
def start_stream():
    global streaming
    streaming = True
    return "Streaming started"

@app.route('/stop')
def stop_stream():
    global streaming
    streaming = False
    return "Streaming stopped"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

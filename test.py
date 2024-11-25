from flask import Flask, render_template, Response, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import threading

# Initialize Flask app
EmotionDetectionApp = Flask(__name__)

# Load the face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./CPS843_Emotion_Detection_Model.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Shock']

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Global flag to control webcam streaming
streaming = False

# Global counter for emotion labels
global happy_count, sad_count, shock_count, angry_count, neutral_count
happy_count = 0
sad_count = 0
shock_count = 0
angry_count = 0
neutral_count = 0

previous_label = None

def generate_frames():
    global streaming, happy_count, sad_count, shock_count, angry_count, neutral_count, previous_label
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
                    print("prediction = ",preds)
                    label = class_labels[preds.argmax()]
                    print("prediction max = ",preds.argmax())
                    print("label = ",label)
                    label_position = (x, y)
                    
                    # If label is one of the 5 classes, increment the appropriate counter
                    if label == "Happy" and label != previous_label:
                        happy_count += 1
                        print("Number of times happy is detected:", happy_count)

                    elif label == "Shock" and label != previous_label:
                        shock_count += 1
                        print("Number of times shock is detected:", shock_count)

                    elif label == "Sad" and label != previous_label:
                        sad_count += 1
                        print("Number of times sad is detected:", sad_count)

                    elif label == "Angry" and label != previous_label:
                        angry_count += 1
                        print("Number of times angry is detected:", angry_count)

                    elif label == "Neutral" and label != previous_label:
                        neutral_count += 1
                        print("Number of times neutral is detected:", neutral_count)

                    previous_label = label

                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Convert frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Yield the frame to be sent to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@EmotionDetectionApp.route('/')
def index():
    global happy_count, sad_count, angry_count, shock_count, neutral_count
    happy_count = 0
    sad_count = 0
    angry_count = 0
    shock_count = 0
    neutral_count = 0

    return render_template('index.html')

@EmotionDetectionApp.route('/start')
def start_stream():
    global streaming
    streaming = True
    return "Streaming started"

@EmotionDetectionApp.route('/stop')
def stop_stream():
    global streaming
    streaming = False
    return "Streaming stopped"

@EmotionDetectionApp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@EmotionDetectionApp.route('/happy_count')
def get_happy_count():
    global happy_count
    return jsonify({'happy_count': happy_count})

@EmotionDetectionApp.route('/angry_count')
def get_angry_count():
    global angry_count
    return jsonify({'angry_count': angry_count})

@EmotionDetectionApp.route('/sad_count')
def get_sad_count():
    global sad_count
    return jsonify({'sad_count': sad_count})

@EmotionDetectionApp.route('/neutral_count')
def get_neutral_count():
    global neutral_count
    return jsonify({'neutral_count': neutral_count})

@EmotionDetectionApp.route('/shock_count')
def get_shock_count():
    global shock_count
    return jsonify({'shock_count': shock_count})

if __name__ == '__main__':
    EmotionDetectionApp.run(debug=True)

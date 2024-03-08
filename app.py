from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model('Deepfake2d.h5')

def extract_frames(video_path, interval_seconds):
    capture = cv2.VideoCapture(video_path)
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    interval_frames = int(frame_rate * interval_seconds)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            frames.append(frame)
        frame_idx += 1

    capture.release()
    return frames

def extract_face(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if faces:
        x, y, w, h = faces[0]['box']
        face = frame[y:y + h, x:x + w]
        return cv2.resize(face, (224, 224))
    else:
        return None
    
def predict_from_video(video_path,time_interval=3):
    if not os.path.exists(video_path):
        return "Video Not Found"
    frames = extract_frames(video_path, time_interval)
    faces = [extract_face(frame) for frame in frames]
    faces = [face for face in faces if face is not None]
    if len(faces) == 0:
        return "No Faces Detected, Try Changing the num_frame"
    faces = np.array(faces)
    faces = preprocess_input(faces)
    # plt.imshow(cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB))
    # plt.show()
    predictions = model.predict(faces)
    prob = predictions
    predictions = (predictions > 0.5).astype(int)
    return predictions,prob

def predict_from_image(image_path,face_detection=True):
    if not os.path.exists(image_path):
        return "Image Not Found"
    frame = cv2.imread(image_path)
    if face_detection:
        face = extract_face(frame)
        if face is None:
            return "No Face Detected"
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        # print(face.shape)
        prediction = model.predict(face)
        # prediction = (prediction > 0.5).astype(int)
        return prediction
    else:
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)
        print(frame.shape)
        prediction = model.predict(frame)
        print(prediction)
        # prediction = (prediction > 0.5).astype(int)
        return prediction

@app.route('/',methods=['GET'])
def home():
    return "Welcome to Deepfake Detection API"

@app.route('/detect_deepfake_video', methods=['POST'])
def detect_deepfake_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)

    predictions,prob = predict_from_video(video_path)
    os.remove(video_path)
    predictions = np.squeeze(predictions)
    prob = np.squeeze(prob)
    predictions = predictions.tolist()
    prob = prob.tolist()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            predictions[i] = "Fake"
        else:
            predictions[i] = "Real"
            prob[i] = 1-prob[i]
    return jsonify({"predictions": predictions,"probability":prob}), 200

@app.route('/detect_deepfake_image', methods=['POST'])
def detect_deepfake_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    # print(image_file)
    image_path = 'uploaded_image.jpg'
    image_file.save(image_path)
    flag = request.form.get('face_detection')
    # print(flag)
    if int(flag):
        predictions = predict_from_image(image_path,face_detection=True)
        prediction = (predictions > 0.5).astype(int)
    else:
        predictions = predict_from_image(image_path,face_detection=False)
        prediction = (predictions > 0.5).astype(int)
    # print("DSD",predictions)
    # print("PROB",probability[0])
    prediction = prediction[0]
    if prediction == 1:
        prediction = "Fake"
        probability = predictions[0][0]
    else:
        prediction = "Real"
        probability = 1-predictions[0][0]
    os.remove(image_path)
    return jsonify({"predictionssss": prediction,"probability":str(probability)}), 200

if __name__ == '__main__':
    app.run(debug=True)
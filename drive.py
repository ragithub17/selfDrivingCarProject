import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
import keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 15

def img_preprocess(img):
    img = img[60:135, :, :]  # Crop image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur
    img = cv2.resize(img, (200, 66))  # Resize image
    img = img / 255  # Normalize image
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    # print(f"Received telemetry data from {sid}: {data}")  # Log the data received
    try:
        speed = float(data['speed'])
        print(f"Speed: {speed}")  # Log the speed
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.array([image])  # Add batch dimension
        steering_angle = float(model.predict(image))  # Predict steering angle
        throttle = 1.0 - speed / speed_limit  # Calculate throttle
        print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')  # Log values
        send_control(steering_angle, throttle)
    except Exception as e:
        print(f"Error in telemetry: {e}")



@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)  # Initial control values

def send_control(steering_angle, throttle):
    print(f'Sending control: Steering angle: {steering_angle}, Throttle: {throttle}')  # Log control values
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),  # Send steering angle as string
        'throttle': str(throttle)  # Send throttle as string
    })

if __name__ == '__main__':
    try:
        model = load_model('Model/model.keras')  # Load the trained model
        print("Model loaded successfully.")
        app = socketio.Middleware(sio, app)
        print("Server started on http://0.0.0.0:4567")
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  # Run server
    except Exception as e:
        print(f"Failed to load model or start server: {e}")

from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model outside the route to ensure it's loaded only once
model = load_model('LSTM_Attention_128HUs.h5')

# ... (your existing imports)
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
import math
import winsound


from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (LSTM, Dense, Concatenate, Attention, Dropout, Softmax,
                                     Input, Flatten, Activation, Bidirectional, Permute, multiply,
                                     ConvLSTM2D, MaxPooling3D, TimeDistributed, Conv2D, MaxPooling2D)

from scipy import stats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

# suppush_ups untraced functions warning
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(225,225,225), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2)
                                 )

cap = cv2.VideoCapture(0) # camera object
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # webcam video frame height
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # webcam video frame width
FPS = int(cap.get(cv2.CAP_PROP_FPS)) # webcam video fram rate

colors = [(5, 32, 74), (0, 148, 198), (192,192,192)]


# Set and test mediapipe model using webcam
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        # Make detection
        image, results = mediapipe_detection(frame, pose)

        set_counter = 0

        # Define the hand landmark model
        hand_landmark_model = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cv2.rectangle(image, (0,0), (640, 40), (5,21,123), -1)
        # cv2.putText(image, 'CHECK DISTANCE', (30,30), )
        cv2.putText(image, 'CHECK DISTANCE', (210,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Render detections
        draw_landmarks(image, results)

        # Display frame on screen
        cv2.imshow('OpenCV Feed', image)

        # Exit / break out logic
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)


num_landmarks = len(landmarks)
num_values = len(test)
num_input_values = num_landmarks*num_values

# This is an example of what we would use as an input into our AI models
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

def extract_keypoints(results):
    """
    Processes and organizes the keypoints detected from the pose estimation model
    to be used as inputs for the exercise decoder models

    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose

"""# 3. Setup Folders for Collection"""

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os. getcwd(),'data')
print(DATA_PATH)

# make directory if it does not exist yet
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Actions/exercises that we try to detect
actions = np.array(['bicep_curl', 'push_ups', 'squat'])
num_classes = len(actions)

# How many videos worth of data
no_sequences = 10

# Videos are going to be this many frames in length
sequence_length = FPS*1


"""# 4. Collect Keypoint Values for Training and Testing"""

# Colors associated with each exercise (e.g., bicep_curls are denoted by blue, squats are denoted by orange, etc.)

model = tf.keras.models.load_model('LSTM_Attention_128HUs.h5')
model_name = 'AttnLSTM'

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def get_coordinates(landmarks, mp_pose, side, joint):
    coord = getattr(mp_pose.PoseLandmark,side.upper()+"_"+joint.upper())
    x_coord_val = landmarks[coord.value].x
    y_coord_val = landmarks[coord.value].y
    return [x_coord_val, y_coord_val]

def viz_joint_angle(image, angle, joint):
    cv2.putText(image, str(int(angle)),
                   tuple(np.multiply(joint, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
    return

def count_reps(image, current_action, landmarks, mp_pose):
    global bicep_curl_counter, push_ups_counter, squat_counter, bicep_curl_stage, push_ups_stage, squat_stage, total

    if current_action == 'bicep_curl':
        # Get coords
        shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
        wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

        # calculate elbow angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # bicep_curl counter logic
        if angle < 30:
            bicep_curl_stage = "up"
        if angle > 140 and bicep_curl_stage =='up':
            bicep_curl_stage="down"
            bicep_curl_counter +=1
            total += 1
        push_ups_stage = None
        squat_stage = None

        # Viz joint angle
        viz_joint_angle(image, angle, elbow)

    elif current_action == 'push_ups':

        # Get coords
        shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        elbow = get_coordinates(landmarks, mp_pose, 'left', 'elbow')
        wrist = get_coordinates(landmarks, mp_pose, 'left', 'wrist')

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Compute distances between joints
        shoulder2elbow_dist = abs(math.dist(shoulder,elbow))
        shoulder2wrist_dist = abs(math.dist(shoulder,wrist))

        # push_ups counter logic
        if (elbow_angle > 150) and (shoulder2elbow_dist < shoulder2wrist_dist):
            push_ups_stage = "up"
        if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (push_ups_stage =='up'):
            push_ups_stage='down'
            push_ups_counter += 1
            total += 1
        bicep_curl_stage = None
        squat_stage = None

        # Viz joint angle
        viz_joint_angle(image, elbow_angle, elbow)

    elif current_action == 'squat':
        # Get coords
        # left side
        left_shoulder = get_coordinates(landmarks, mp_pose, 'left', 'shoulder')
        left_hip = get_coordinates(landmarks, mp_pose, 'left', 'hip')
        left_knee = get_coordinates(landmarks, mp_pose, 'left', 'knee')
        left_ankle = get_coordinates(landmarks, mp_pose, 'left', 'ankle')
        # right side
        right_shoulder = get_coordinates(landmarks, mp_pose, 'right', 'shoulder')
        right_hip = get_coordinates(landmarks, mp_pose, 'right', 'hip')
        right_knee = get_coordinates(landmarks, mp_pose, 'right', 'knee')
        right_ankle = get_coordinates(landmarks, mp_pose, 'right', 'ankle')

        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate hip angles
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Squat counter logic
        thr = 165
        if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
            squat_stage = "down"
        if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage =='down'):
            squat_stage='up'
            squat_counter += 1
            total += 1
        bicep_curl_stage = None
        push_ups_stage = None

        # Viz joint angles
        viz_joint_angle(image, left_knee_angle, left_knee)
        viz_joint_angle(image, left_hip_angle, left_hip)

    else:
        pass


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

# 1. New detection variables
sequence = []
predictions = []
res = []
threshold = 0.5 # minimum confidence to classify as an action/exercise
current_action = ''

# Rep counter logic variables
bicep_curl_counter = 0
push_ups_counter = 0
squat_counter = 0
total = 0
bicep_curl_stage = None
push_ups_stage = None
squat_stage = None

# Your existing code for mediapipe_detection, draw_landmarks, and other functions...

def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image, results = mediapipe_detection(frame, pose)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                current_action = actions[np.argmax(res)]
                confidence = np.max(res)

                if confidence >= threshold:
                    image = prob_viz(res, actions, image, colors)
                    try:
                        landmarks = results.pose_landmarks.landmark
                        count_reps(image, current_action, landmarks, mp_pose)
                    except:
                        pass

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

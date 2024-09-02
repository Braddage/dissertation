import tkinter as tk
from tkinter import filedialog
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import sys
from sys import platform
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader


class squatRegressionModel(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super(squatRegressionModel, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        self.fc1 = nn.Linear(32 * 6, 32)  # 6 here is the number of features
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # regression model, single output

        # batch normalization layers
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        # dropout layers
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # apply batch norm, activation functions and dropout after each convolutional layer
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)

        # flatten output for fully connected layers
        x = x.view(-1, 32 * 6)  # 6 here is the number of features

        # apply activation functions and dropout after each fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


def extractKeypoints(video_path):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        try:
            # Windows Import
            if platform == "win32":
                sys.path.append(dir_path + '/../../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../../x64/Release;' + dir_path + '/../../../bin;'
                import pyopenpose as op
            else:
                sys.path.append('../../../python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--video_path", default=video_path, help="Process a video file.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../../models/"
        params["model_pose"] = "BODY_25"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Video
        cap = cv2.VideoCapture(args[0].video_path)
        keypoints = []  # list to store all detected keypoints
        frames = []  # list to store frames
        frames_to_take = 20
        step_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frames_to_take
        frame_step = 0
        frames_taken = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process every step_size-th frame
            if frame_step % step_size == 0 and frames_taken < frames_to_take:
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                # Check if keypoints are detectable
                if datum.poseKeypoints is not None:
                    # Keypoints detected
                    keypoints.append(datum.poseKeypoints[0])
                    frames.append(frame)
                    frames_taken += 1

            frame_step += 1
        cap.release()

        # remove axis of confidence values
        keypoints = np.delete(keypoints, 2, axis=2)

        return np.array(keypoints), frames

    except Exception as e:
        print(e)
        sys.exit(-1)


def calculateAngle(joint1, joint2, joint3):
    # calculate vectors
    vector1 = np.array(joint1) - np.array(joint2)
    vector2 = np.array(joint3) - np.array(joint2)

    # calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # calculate cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # convert cosine to angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def calculateDistance(joint1, joint2):
    distance = np.sqrt((joint1[0] - joint2[0]) ** 2 + (joint1[1] - joint2[1]) ** 2)
    return distance


def extractSquatFeatures(keypoints):
    features = np.zeros((len(keypoints), 6))

    for i, kp in enumerate(keypoints):
        neck, Rshoulder, Relbow, Rwrist, Lshoulder, Lelbow, Lwrist, hip, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle = kp[1:15]

        # create mid-knee keypoint
        mid_knee_x = (Rknee[0] + Lknee[0]) / 2
        mid_knee_y = (Rknee[1] + Lknee[1]) / 2
        mid_knee = (mid_knee_x, mid_knee_y)

        knee_width = calculateDistance(Lknee, Rknee)

        L_knee_angle = calculateAngle(Lhip, Lknee, Lankle)
        R_knee_angle = calculateAngle(Rhip, Rknee, Rankle)
        hip_angle = calculateAngle(mid_knee, hip, neck)
        hip_knee_y_diff = hip[1] - mid_knee_y

        stance_width = calculateDistance(Lankle, Rankle)

        features[i] = [L_knee_angle, R_knee_angle, hip_angle, knee_width, hip_knee_y_diff, stance_width]

    return features


def extractSquatAdvice(features, keypoints, frames):
    advice = []
    issue_frames = []

    mid_wrist_x_start = (keypoints[0][3][0] + keypoints[0][6][0]) / 2
    mid_wrist_y_end = (keypoints[-1][3][0] + keypoints[-1][6][0]) / 2
    mid_wrist_x_diff = abs(mid_wrist_x_start - mid_wrist_y_end)
    if mid_wrist_x_diff > 30:
        advice.append("Try to keep the bar path vertical.")
        issue_frames.append(frames[-1])

    lockout_L_knee_angle = features[-1][0]
    lockout_R_knee_angle = features[-1][1]
    if lockout_L_knee_angle < 150 or lockout_R_knee_angle < 150:
        advice.append("Try to get closer to lockout.")
        issue_frames.append(frames[-1])

    half_length = len(features) // 2
    avg_hip_angle = sum(feature[2] for feature in features[:half_length]) / half_length
    if avg_hip_angle < 70:
        advice.append("Try to stay more upright and avoid folding forwards.")
        issue_frames.append(frames[4])

    hip_knee_y_diff = features[0][4]
    if hip_knee_y_diff < -30:
        advice.append("Try to break parallel and squat deeper for a better stretch on the quads.")
        issue_frames.append(frames[0])

    # check for knee cave / inconsistent knee space throughout lift
    lockout_knee_width = features[-1][3]
    for i in range(len(features)):
        knee_width = features[i][3]
        # check knee cave
        if (knee_width < 0.8 * lockout_knee_width):
            advice.append("Significant knee cave detected.")
            issue_frames.append(frames[i])
            break

    stance_width = features[-1][5]
    shoulder_width = calculateDistance(keypoints[-1][1], keypoints[-1][4])
    stance_shoulder_width_ratio = stance_width / shoulder_width
    if stance_shoulder_width_ratio > 1.3:
        advice.append("Squat stance is too wide.")
        issue_frames.append(frames[0])
    elif stance_shoulder_width_ratio < 0.8:
        advice.append("Squat stance is too narrow.")
        issue_frames.append(frames[0])
    return advice, issue_frames


def calculateSquatScore(features):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    model_path = os.path.join(current_dir, '..', 'squatRegressionModelTuned.pth')
    model = torch.load(model_path)

    # create dataloader
    dataset = TensorDataset(torch.Tensor(features))
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    # get output from model
    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            outputs = model(inputs)
        score = torch.mean(outputs, dim=0).item()

    if (score <= 4):
        score = 0
    elif (score >= 7):
        score = 10
    else:
        score = ((score - 4) / 3) * 10
    return score


def processSquat(video_path):
    keypoints, frames = extractKeypoints(video_path)
    keypoints = np.array(keypoints)
    features = extractSquatFeatures(keypoints)
    advice, issue_frames = extractSquatAdvice(features, keypoints, frames)
    score = calculateSquatScore(features)
    if score == 'nan':  # catch case
        'Error in scoring, please ensure you face the camera in a way such that the shoulders are visible throughout.'
    if not advice:
        advice = "Exercised performed with good form. No advice necessary!"
    return score, advice, issue_frames

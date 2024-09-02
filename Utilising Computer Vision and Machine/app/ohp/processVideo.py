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


class OHPRegressionModel(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super(OHPRegressionModel, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        self.fc1 = nn.Linear(32 * 10, 32)  # 10 here is the number of features
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
        x = x.view(-1, 32 * 10)  # 10 here is the number of features

        # apply activation functions and dropout after each fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


# section used from packaged example from openpose
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
        frames = []  # array to store frames
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


def extractOHPFeatures(keypoints):
    num_frames = len(keypoints)
    features = np.zeros((num_frames, 10))

    for i, kp in enumerate(keypoints):
        neck, Rshoulder, Relbow, Rwrist, Lshoulder, Lelbow, Lwrist, hip, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle = kp[1:15]

        # create mid-knee keypoint
        mid_knee_x = (Rknee[0] + Lknee[0]) / 2
        mid_knee_y = (Rknee[1] + Lknee[1]) / 2
        mid_knee = (mid_knee_x, mid_knee_y)

        # create mid-shoulder keypoint
        mid_shoulder_x = (Rshoulder[0] + Lshoulder[0]) / 2
        mid_shoulder_y = (Rshoulder[1] + Lshoulder[1]) / 2

        # create mid-wrist keypoint

        mid_wrist_x = (Rwrist[0] + Lwrist[0]) / 2
        mid_wrist_y = (Rwrist[1] + Lwrist[1]) / 2

        # calculate x offset between mid wrist and mid shoulder
        shoulder_wrist_x_diff = abs(mid_wrist_x - mid_shoulder_x)
        shoulder_wrist_y_diff = mid_wrist_y - mid_shoulder_y

        L_knee_angle = calculateAngle(Lhip, Lknee, Lankle)
        R_knee_angle = calculateAngle(Rhip, Rknee, Rankle)
        hip_angle = calculateAngle(mid_knee, hip, neck)
        L_elbow_angle = calculateAngle(Lshoulder, Lelbow, Lwrist)
        R_elbow_angle = calculateAngle(Rshoulder, Relbow, Rwrist)
        L_shoulder_angle = calculateAngle(Lelbow, Lshoulder, Lhip)
        R_shoulder_angle = calculateAngle(Relbow, Rshoulder, Rhip)
        grip_shoulder_ratio = calculateDistance(Lwrist, Rwrist) / calculateDistance(Lshoulder, Rshoulder)

        features[i] = [L_knee_angle, R_knee_angle, hip_angle, shoulder_wrist_x_diff, shoulder_wrist_y_diff, L_elbow_angle, R_elbow_angle,
                       L_shoulder_angle, R_shoulder_angle, grip_shoulder_ratio]

    return features


def extractOHPAdvice(features, keypoints, frames):
    advice = []
    issue_frames = []

    # get keypoint values of shoulder and grip width
    for i in range(len(features)):
        grip_shoulder_ratio = features[i][9]
        if grip_shoulder_ratio > 2.8:
            advice.append("Grip width is too wide.")
            issue_frames.append(frames[i])
            break
        elif grip_shoulder_ratio < 1.4:
            advice.append("Grip width is too narrow.")
            issue_frames.append(frames[i])
            break

    # calculate average knee bend at beginning of lift
    half_length = len(features) // 2
    avg_knee_bend = sum((feature[0] + feature[1]) / 2 for feature in features[:half_length]) / half_length

    if avg_knee_bend < 160:
        advice.append("Try to keep your legs straight to maximise shoulder recruitment.")
        issue_frames.append(frames[4])

    lockout_L_elbow_angle = features[-1][5]
    lockout_R_elbow_angle = features[-1][6]

    if lockout_L_elbow_angle < 125 or lockout_R_elbow_angle < 125:
        advice.append("Try to get closer to lockout.")
        issue_frames.append(frames[-1])

    avg_hip_angle = sum(feature[2] for feature in features) / len(features)

    if avg_hip_angle < 160:
        advice.append("Try to keep hips stacked underneath shoulders, avoid leaning!")
        issue_frames.append(frames[9])

    for i in range(len(features)):
        shoulder_wrist_x_diff = features[i][3]
        if shoulder_wrist_x_diff > 120:
            advice.append("Try to keep the bar path vertical. Avoid 'throwing' the bar laterally")
            issue_frames.append(frames[i])
            break

    shoulder_wrist_y_diff = features[0][4]
    if shoulder_wrist_y_diff < -20:
        advice.append("Start the rep closer to the height of your shoulders for a better stretch.")
        issue_frames.append(frames[i])

    return advice, issue_frames


def calculateOHPScore(features):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    model_path = os.path.join(current_dir, '..', 'ohpRegressionModelTuned.pth')
    model = torch.load(model_path)
    batch_size = 20

    # create dataloader
    dataset = TensorDataset(torch.Tensor(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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


def processOHP(video_path):
    keypoints, frames = extractKeypoints(video_path)
    keypoints = np.array(keypoints)
    features = extractOHPFeatures(keypoints)
    advice, issue_frames = extractOHPAdvice(features, keypoints, frames)
    score = calculateOHPScore(features)
    if score == 'nan':  # catch case
        'Error in scoring, please ensure you face the camera in a way such that the shoulders are visible throughout.'
    if not advice:
        advice = "Exercised performed with good form. No advice necessary!"
    return score, advice, issue_frames

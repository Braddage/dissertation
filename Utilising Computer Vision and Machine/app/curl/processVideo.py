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


class curlRegressionModel(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super(curlRegressionModel, self).__init__()
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


def extractCurlFeatures(keypoints):
    num_frames = len(keypoints)
    features = np.zeros((num_frames, 6))

    for i, kp in enumerate(keypoints):
        neck, Rshoulder, Relbow, Rwrist, Lshoulder, Lelbow, Lwrist, hip, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle = kp[1:15]

        L_shoulder_angle = calculateAngle(Lelbow, Lshoulder, Lhip)
        R_shoulder_angle = calculateAngle(Relbow, Rshoulder, Rhip)

        # calculate x offset between mid wrist and mid shoulder
        L_elbow_angle = calculateAngle(Lshoulder, Lelbow, Lwrist)
        R_elbow_angle = calculateAngle(Rshoulder, Relbow, Rwrist)
        grip_shoulder_ratio = calculateDistance(Lwrist, Rwrist) / calculateDistance(Lshoulder, Rshoulder)

        torso_angle = calculateAngle(hip, neck, (neck[0], neck[0] + 20))

        features[i] = [L_elbow_angle, R_elbow_angle, grip_shoulder_ratio, torso_angle, L_shoulder_angle, R_shoulder_angle]

    return features


def extractCurlAdvice(features, keypoints, frames):
    advice = []
    issue_frames = []
    # get keypoint values of shoulder and grip width
    for i in range(len(features)):
        grip_shoulder_ratio = features[i][2]
        if grip_shoulder_ratio < 0.7:
            advice.append("Grip width is too narrow")
            issue_frames.append(frames[i])
            break
        elif grip_shoulder_ratio > 1.6:
            advice.append("Grip width is too wide.")
            issue_frames.append(frames[i])
            break

    for i in range(len(features)):
        torso_angle = features[i][3]
        if torso_angle > 5 and torso_angle < 175:
            advice.append("Keep your core stable and upright, swinging and/or leaning detected.")
            issue_frames.append(frames[i])
            break

    start_L_elbow_angle = features[0][0]
    start_R_elbow_angle = features[0][1]

    if start_L_elbow_angle < 145 or start_R_elbow_angle < 145:
        advice.append("Make sure to extend arms close to being locked out in the starting position.")
        issue_frames.append(frames[0])

    end_L_elbow_angle = features[-1][0]
    end_R_elbow_angle = features[-1][1]

    if end_L_elbow_angle > 45 or end_R_elbow_angle > 45:
        advice.append("Partial range of motion detected. Make sure to squeeze at the top.")
        issue_frames.append(frames[-1])

    return advice, issue_frames


def calculateCurlScore(features):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    model_path = os.path.join(current_dir, '..', 'curlRegressionModelTuned.pth')
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
    if (score <= 3):
        score = 0
    elif (score >= 6.5):
        score = 10
    else:
        score = ((score - 3) / 3.5) * 10
    return score


def processCurl(video_path):
    keypoints, frames = extractKeypoints(video_path)
    keypoints = np.array(keypoints)
    features = extractCurlFeatures(keypoints)
    advice, issue_frames = extractCurlAdvice(features, keypoints, frames)
    score = calculateCurlScore(features)
    if score == 'nan':  # catch case
        'Error in scoring, please ensure you face the camera in a way such that the shoulders are visible throughout.'
    if not advice:
        advice = "Exercised performed with good form. No advice neccessary!"
    return score, advice, issue_frames

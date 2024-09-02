import sys
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
from sys import platform
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader


class deadliftRegressionModel(nn.Module):
    def __init__(self):
        super(deadliftRegressionModel, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # fully connected layers
        self.fc1 = nn.Linear(32 * 8, 32)  # 8 here is the number of features
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
        x = x.view(-1, 32 * 8)  # 8 here is the number of features

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


def extractDeadliftFeatures(keypoints):
    num_frames = len(keypoints)
    features = np.zeros((num_frames, 8))

    for i, kp in enumerate(keypoints):
        neck, Rshoulder, Relbow, Rwrist, Lshoulder, Lelbow, Lwrist, hip, Lhip, Lknee, Lankle, Rhip, Rknee, Rankle = kp[1:15]

        mid_knee_x = (Rknee[0] + Lknee[0]) / 2
        mid_knee_y = (Rknee[1] + Lknee[1]) / 2
        mid_knee = (mid_knee_x, mid_knee_y)

        mid_shoulder_x = (Rshoulder[0] + Lshoulder[0]) / 2
        mid_shoulder_y = (Rshoulder[1] + Lshoulder[1]) / 2

        mid_wrist_x = (Rwrist[0] + Lwrist[0]) / 2
        shoulder_wrist_diff = abs(mid_shoulder_x - mid_wrist_x)

        L_knee_angle = calculateAngle(Lhip, Lknee, Lankle)
        R_knee_angle = calculateAngle(Rhip, Rknee, Rankle)
        hip_angle = calculateAngle(mid_knee, hip, neck)
        L_elbow_angle = calculateAngle(Lshoulder, Lelbow, Lwrist)
        R_elbow_angle = calculateAngle(Rshoulder, Relbow, Rwrist)
        Knee_width = calculateDistance(Lknee, Rknee)
        shoulder_hip_y_diff = mid_shoulder_y - hip[1]

        features[i] = [L_knee_angle, R_knee_angle, hip_angle, shoulder_wrist_diff, Knee_width, L_elbow_angle, R_elbow_angle,
                       shoulder_hip_y_diff]

    return features


def extractDeadliftAdvice(features, keypoints, frames):
    advice = []
    issue_frames = []
    # get keypoint values of shoulder and grip width
    for i, kp in enumerate(keypoints):
        Rshoulder, Rwrist, Lshoulder, Lwrist = kp[2], kp[4], kp[5], kp[7]

        # calculate ratio between shoulder width and grip width
        grip_shoulder_ratio = calculateDistance(Lwrist, Rwrist) / calculateDistance(Lshoulder, Rshoulder)
        if grip_shoulder_ratio > 2:
            advice.append("Grip width is too wide.")
            issue_frames.append(frames[i])
            break
        if grip_shoulder_ratio < 0.8:
            advice.append("Grip width is too narrow.")
            issue_frames.append(frames[i])
            break

    # calculate average knee bend at beginning of lift
    half_length = len(features) // 2
    avg_knee_bend = sum((feature[0] + feature[1]) / 2 for feature in features[:half_length]) / half_length

    if avg_knee_bend > 160:
        advice.append("Try to incorporate more leg drive.")
        issue_frames.append(frames[5])
    elif avg_knee_bend < 120:
        advice.append("Hinge more about the hips to allow yourself to reach the bar without excessive knee bend.")
        issue_frames.append(frames[5])

    # calculate average elbow bend throughout lift
    avg_L_elbow_angle = sum(feature[5] for feature in features) / len(features)
    avg_R_elbow_angle = sum(feature[6] for feature in features) / len(features)

    # check if average elbow bend is excessive
    if avg_L_elbow_angle < 155 or avg_R_elbow_angle < 155:
        advice.append("Keep both arms straight")
        issue_frames.append(frames[10])

    # check for knee cave / inconsistent knee space throughout lift
    lockout_knee_width = features[len(features) - 1][4]
    for i in range(len(features)):
        knee_width = features[i][4]
        # check knee cave
        if (knee_width / lockout_knee_width) < 0.6:
            advice.append("Significant knee cave detected.")
            issue_frames.append(frames[i])
            break
        # check sumo style
        if (knee_width / lockout_knee_width) > 1.4:
            advice.append("Sumo deadlift detected. Please note, this system is designed for conventional.")
            issue_frames.append(frames[i])
            break

    for i in range(len(features)):
        shoulder_wrist_diff = features[i][3]
        if shoulder_wrist_diff > 70:
            advice.append("Try to keep the bar underneath the shoulders. Avoid 'throwing' the bar laterally")
            issue_frames.append(frames[i])
            break

    # check if hips are too high relative to shoulders
    shoulder_hip_y_diff = features[0][7]
    if shoulder_hip_y_diff > -20:
        advice.append(
            "Starting hip position is too high. Hips should be higher than knees and lower than shoulders. This may also indicate back rounding. Maintain a neutral spine.")
        issue_frames.append(frames[0])

    # catch case should exercise require no advice
    if not advice:
        advice = "Exercised performed with good form. No advice necessary!"

    return advice, issue_frames


def calculateDeadliftScore(features):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory of the script
    model_path = os.path.join(current_dir, '..','deadliftRegressionModelTuned.pth')
    model = torch.load(model_path)

    # create dataloader
    dataset = TensorDataset(torch.Tensor(features))
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    # get output from model
    model.eval()
    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            outputs = model(inputs)  # pass input data through model
        score = torch.mean(outputs, dim=0).item()

    # normalize outputs from range 6-8 into a 1-10 scale
    if (score <= 6):
        score = 0
    elif (score >= 8):
        score = 10
    else:
        score = ((score - 6) / 2) * 10
    return score


def processDeadlift(video_path):
    # extract keypoints with OpenPose
    keypoints, frames = extractKeypoints(video_path)

    # feature extraction for neural network and advice generation
    features = extractDeadliftFeatures(keypoints)

    # advice extraction using features and thresholds
    advice, issue_frames = extractDeadliftAdvice(features, keypoints, frames)

    # score calculation from trained models
    score = calculateDeadliftScore(features)

    # return score, advice and frames containing mistakes found during advice extraction
    return score, advice, issue_frames


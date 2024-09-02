import sys
import numpy as np

# data = np.load('deadliftCalculations.npz')
data = np.load('ohpRegression.npz')
keypoints = data['keypoints']
labels = data['scores']
keypoints = np.delete(keypoints, 2, axis=3)

print(keypoints.shape)
print(labels.shape)


def calculate_angle(joint1, joint2, joint3):
    if np.any(joint1 != 0) and np.any(joint2 != 0) and np.any(joint3 != 0):
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
    else:
        return 0


def calculate_distance(joint1, joint2):
    # check if both keypoints are detected
    if joint1 is not None and joint2 is not None:
        # calculate the distance using the euclidean distance formula
        distance = np.sqrt((joint1[0] - joint2[0]) ** 2 + (joint1[1] - joint2[1]) ** 2)
        return distance
    else:
        print("Knee keypoints not detected for both sides.")
        return None


num_videos = len(keypoints)
num_frames = len(keypoints[0])
features = np.zeros((num_videos, num_frames, 10))
for i in range(len(keypoints)):
    for j in range(len(keypoints[i])):
        neck = keypoints[i][j][1]
        Rshoulder = keypoints[i][j][2]
        Relbow = keypoints[i][j][3]
        Rwrist = keypoints[i][j][4]
        Lshoulder = keypoints[i][j][5]
        Lelbow = keypoints[i][j][6]
        Lwrist = keypoints[i][j][7]
        hip = keypoints[i][j][8]
        Lhip = keypoints[i][j][9]
        Lknee = keypoints[i][j][10]
        Lankle = keypoints[i][j][11]
        Rhip = keypoints[i][j][12]
        Rknee = keypoints[i][j][13]
        Rankle = keypoints[i][j][14]

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

        L_knee_angle = calculate_angle(Lhip, Lknee, Lankle)
        R_knee_angle = calculate_angle(Rhip, Rknee, Rankle)
        hip_angle = calculate_angle(mid_knee, hip, neck)
        L_elbow_angle = calculate_angle(Lshoulder, Lelbow, Lwrist)
        R_elbow_angle = calculate_angle(Rshoulder, Relbow, Rwrist)
        L_shoulder_angle = calculate_angle(Lelbow, Lshoulder, Lhip)
        R_shoulder_angle = calculate_angle(Relbow, Rshoulder, Rhip)
        grip_shoulder_ratio = calculate_distance(Lwrist, Rwrist) / calculate_distance(Lshoulder, Rshoulder)

        features[i][j] = [L_knee_angle, R_knee_angle, hip_angle, shoulder_wrist_x_diff, shoulder_wrist_y_diff, L_elbow_angle, R_elbow_angle,
                          L_shoulder_angle, R_shoulder_angle, grip_shoulder_ratio]

np.savez('ohpFeatures.npz', features=features, labels=labels)

sys.exit(-1)

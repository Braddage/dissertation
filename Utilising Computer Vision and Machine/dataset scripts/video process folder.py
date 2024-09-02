# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import numpy as np
from sys import platform
import argparse


def processFolder(folder_path):
    # iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            # create path
            video_path = os.path.join(folder_path, filename)
            processVideo(video_path, filename)


def processVideo(video_path, filename):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                sys.path.append('../../python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--video_path", default=video_path, help="Process a video file.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"
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

        # starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # process Video
        cap = cv2.VideoCapture(args[0].video_path)
        all_keypoints = []  # List to store all detected keypoints
        # Total number of frames in the video
        frames_to_take = 20
        step_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frames_to_take
        frame_count = 0
        frames_taken = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # process every x frames
            if frame_count % step_size == 0 and frames_taken < frames_to_take:
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                # check if keypoints are detectable
                if datum.poseKeypoints is not None:
                    all_keypoints.append(datum.poseKeypoints[0])
                    frames_taken += 1

            frame_count += 1

            # display Image
            cv2.namedWindow('lift', cv2.WINDOW_NORMAL)
            cv2.imshow('lift', datum.cvOutputData)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit if 'q' is pressed
                break

        key = cv2.waitKey(0) & 0xFF
        score = int(key) - 48
        cap.release()
        cv2.destroyAllWindows()

        # save the array to a file
        if os.path.exists('ohpRegression.npz'):
            data = np.load('ohpRegression.npz', allow_pickle=True)
            appended_keypoints = np.append(data['keypoints'], [np.array(all_keypoints, dtype=float)], axis=0)
            appended_scores = np.append(data['scores'], score)
            np.savez('ohpRegression.npz', keypoints=appended_keypoints, scores=appended_scores)
        else:
            np.savez('ohpRegression.npz', keypoints=[np.array(all_keypoints, dtype=float)], scores=[score])

        return

    except Exception as e:
        print(e)
        sys.exit(-1)


folder_path = "../../../examples/media/ohp"
processFolder(folder_path)
sys.exit(-1)
import numpy as np


data = np.load('ohpFeatures.npz')
features = data['features']
labels = data['labels']

num_videos=len(features)
num_frames=len(features[0])

flattened_features = np.zeros((num_videos*num_frames, 10))
flattened_labels = np.zeros((num_videos*num_frames))

count = 0
for i in range(num_videos):
    for j in range(num_frames):
        flattened_features[count] = features[i][j]
        flattened_labels[count] = labels[i]
        count += 1

print(len(features))
print(labels.shape)
# convert to np array

print(flattened_features.shape)
print(flattened_labels.shape)

np.savez('ohpRegressionDataset.npz', features = flattened_features, labels = flattened_labels)
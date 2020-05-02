import os
from matplotlib.image import imread
import numpy as np


class PreProcessing:

    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self,data_src, dataMode):
        self.data_src = data_src
        self.data_mode = dataMode
        print("Loading Geological Similarity Dataset...")
        self.images, self.labels = self.preprocessing()
        self.unique_label = np.unique(self.labels)
        self.map_label_indices = {label: np.flatnonzero(self.labels == label) for label in
                                        self.unique_label}
        print('Preprocessing Done. Summary:')
        print("Images :", self.images.shape)
        print("Labels :", self.labels.shape)
        print("Unique label :", self.unique_label)

    def normalize(self,x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        X = []
        y = []
        minDirectory = 0
        maxDirectory = 100
        if self.data_mode == "training":
            minDirectory = 0
            maxDirectory = 64
        elif self.data_mode == "validation":
            minDirectory = 65
            maxDirectory = 84
        elif self.data_mode == "test":
            minDirectory = 85
            maxDirectory = 100
        print("Setting min directory to %d and max directory to %d in data mode %s" %(minDirectory, maxDirectory, self.data_mode))
        for directory in os.listdir(self.data_src):
            if directory.startswith("."):
                continue
            # pick which directories you would like to read based on data mode
            if int(directory) < minDirectory or int(directory) > maxDirectory:
                continue
            try:
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = imread(os.path.join(self.data_src, directory, pic))
                    X.append(np.squeeze(np.asarray(img)))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X,y

    def preprocessing(self):
        X, y = self.read_dataset()
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])
        X = [self.normalize(x) for x in X]                                  # normalize images

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

        return np.asarray(x_shuffled), np.asarray(y_shuffled)


    def get_triplets(self):
        label_l, label_r = np.random.choice(self.unique_label, 2, replace=False)
        a, p = np.random.choice(self.map_label_indices[label_l],2, replace=False)
        n = np.random.choice(self.map_label_indices[label_r])
        return a, p, n

    def get_triplets_batch(self,n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        return self.images[idxs_a,:], self.images[idxs_p, :], self.images[idxs_n, :]


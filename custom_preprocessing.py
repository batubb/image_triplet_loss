import os
from matplotlib.image import imread
import numpy as np
import random


class PreProcessing:

    def __init__(self,data_src):
        # Given an index, return the file path of the image
        self.indexToFilePath = {}
        # Given a label return the directory name of the label
        self.labelToDirName = {}
        # Given a label what are all the training indices that are in that label
        self.labelToTrainFileIndices = {}
        # Given a label what are all the validation indices that are in that label
        self.labelToValFileIndices = {}
        self.data_src = data_src
        print("Loading Signature Dataset...")
        self.X_train_idx, self.X_val_idx, self.y_train, self.y_val = self.preprocessing(0.9)
        self.unique_train_label = np.unique(self.labelToTrainFileIndices.keys())

        print('Preprocessing Done. Summary:')
        print("Num images train :", len(self.X_train_idx))
        print("Num labels train :", len(self.y_train))
        print("Num images validation  :", len(self.X_val_idx))
        print("Num labels validation  :", len(self.y_val))
        print("Unique train label :", self.unique_train_label)


    def populateLabelToIndices(self, X, y, labelToIdx):
        for i in range(len(X)):
            curLabel = y[i]
            curIndex = X[i]
            if curLabel not in labelToIdx:
                labelToIdx[curLabel] = []
            labelToIdx[curLabel].append(curIndex)

    def normalize(self,x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        data = []
        i = 0
        latestLabel = 0
        for directory in os.listdir(self.data_src):
            if directory.startswith("."):
                continue
            # don't touch excludes
            if directory.endswith("_exclude"):
                continue
            # don't touch test dataset
            if int(directory) >= 80:
                continue
            try:
                for picFileName in os.listdir(os.path.join(self.data_src, directory)):
                    # map index i to a file name
                    self.indexToFilePath[i] = os.path.join(directory, picFileName)
                    # map label to a directory name
                    curLabel = 0
                    # If we have not inserted this label to our map yet
                    if latestLabel not in self.labelToDirName:
                        self.labelToDirName[latestLabel] = directory
                    # If we have inserted it, but it does not map to our current dir
                    if self.labelToDirName[latestLabel] != directory:
                        latestLabel += 1
                        self.labelToDirName[latestLabel] = directory
                    # insert to data the label of the current data and its index
                    data.append((i, latestLabel))
                    i += 1
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return data

    def preprocessing(self,train_test_ratio):
        # TODO: add normalization 
        data = self.read_dataset()
        # every label has the same amount of data so we can randomly shuffle
        random.shuffle(data)
        X_indices, y = map(list, zip(*data))    

        size_of_dataset = len(X_indices)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))
        X_train_idx = np.asarray(X_indices[0:n_train])
        y_train = np.asarray(y[0:n_train])
        X_val_idx = np.asarray(X_indices[n_train:])
        y_val = np.asarray(y[n_train:])
        self.populateLabelToIndices(X_train_idx, y_train, self.labelToTrainFileIndices)
        self.populateLabelToIndices(X_val_idx, y_val, self.labelToValFileIndices)
        return X_train_idx, X_val_idx, y_train, y_val


    def get_triplets(self):
        # pick two labels and ensure that one has at least two data points
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        while len(self.labelToTrainFileIndices[label_l]) < 2:
            label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(self.labelToTrainFileIndices[label_l], 2, replace=False)
        n = np.random.choice(self.labelToTrainFileIndices[label_r])
        return a, p, n

    
    def loadImagesToMatrix(self, indices):
        X = []
        for index in indices:
            filePath = self.indexToFilePath[index]
            img = imread(os.path.join(self.data_src, filePath))
            X.append(np.squeeze(np.asarray(img)))
        return np.asarray(X)

    def get_triplets_batch(self,n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        # now load images and return
        anchorImages = self.loadImagesToMatrix(idxs_a)
        positiveImages = self.loadImagesToMatrix(idxs_p)
        negativeImages = self.loadImagesToMatrix(idxs_n)
        return anchorImages, positiveImages, negativeImages


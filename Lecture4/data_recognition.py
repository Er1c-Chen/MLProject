import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def loadData():
    path = './data/project1-data-Recognition/train'
    files = os.listdir(path)
    image = []
    for file in files:
        img = cv2.imread(path + '/' + file, 0)
        image.append(img)
    return np.mat(image)


loadData()
def preProcess():
    pass
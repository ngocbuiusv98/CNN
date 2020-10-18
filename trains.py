import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import dlib
from model import create_model
from align import AlignDlib
import glob
import imutils
import os

# INITIALIZE MODELS
nn4_small2 = create_model()

nn4_small2.summary()

# load dữ liệu để tối ưu mô hình cnn
nn4_small2.load_weights('weights/nn4.small2.v1.h5')

# bộ dữ liệu 68 điểm đặc trưng để căn chỉnh mặt.
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

# LOAD Image Data
train_paths = glob.glob("image/*")
print(train_paths)
train_paths_v2 = glob.glob("image/csdl/*")
nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['index', 'image', 'label', 'name'])

index = 0
for i, train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)] = [index, image, i, name]
        index += 1
print(df_train)

def align_face(face):
    # print(img.shape)
    (h, w, c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    # print(bb)
    return alignment.align(96, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
# lấy dữ liệu để tính embedding vector

def load_and_align_images(filepaths):
    aligned_images = []
    for filepath in filepaths:
        # print(filepath)
        img = cv2.imread(filepath)
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)

    return np.array(aligned_images)


def calc_embs(filepaths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(
            filepaths[start:start+batch_size])
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
    embs = np.array(pd)

    return np.array(embs)

# sau khi tính emb của tập train sẽ lưu vào train_embs.npy
train_embs = calc_embs(df_train.image)
np.save("train_embs.npy", train_embs)

print("Done!!!")
# 원하는 모델로 바꾸어 사용하기
from model_face import NN, ResNet_50, BottleneckBlock

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import pandas as pd

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사용할 모델 선언하기
NN = NN().to(device)

# train이 아닌, evaluation 과정
NN.eval()

# 기존에 학습한 모델 불러오기. XXXXXXXXXXXX에 epoch 번호 작성.
path = './checkpoint/model_weights_NN/weights_epoch_' + '66' + '.pth.tar'
checkpoint = torch.load(path, map_location=device)
NN.load_state_dict(checkpoint['NN'])

test = pd.read_csv("./data/fer2013.csv")["pixels"][22]  # 10, 15, 22, 24
test = test.split(' ')
test = np.array(list(map(int, test)), 'float32')
test = test.reshape([48, 48])

img = cv2.imread("./test_image1.png")  # 이미지 불러오기
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (48, 48))
img = cv2.equalizeHist(img)
img = np.array(img, 'float32')

face_frame = img / 255
input_face = torch.tensor(face_frame)
input_face = input_face.to(device)
input_face = input_face.reshape(1, 1, 48, 48)


def get_label_emotion(label):
    """
    label 값에 대응되는 감정의 이름 문자열을 구하기 위한 함수

    매개변수 (Parameters)
    ----------
    label : int
        emotion label 번호

    반환 값 (Returns)
    -------
    String
        label 번호에 대응되는 감정의 이름 문자열

    """
    # 여기에 코드 작성

    data = {0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'}

    return data[label]


softmax = torch.nn.Softmax()

emotion_vec = NN(input_face.float()).squeeze()

# 7차원 감정 확률 벡터
vision_vec = softmax(emotion_vec)

vision_vec = vision_vec.cpu().detach().numpy()  # .reshape(-1,1)

vision_label_for_Qn = np.argmax(vision_vec)

vision_emotion_label = get_label_emotion(np.argmax(vision_vec))
print(vision_emotion_label)
vision_percentage = np.max(vision_vec)

# 감정 벡터 상수 설정
Happy_vec = np.array([0.6, 0.85])
Surprise_vec = np.array([0.05, 0.5])
anger_vec = np.array([-0.6, 0.85])
disgust_vec = np.array([-0.3, 0.45])
fear_vec = np.array([-0.62, 0.1])
sad_vec = np.array([-0.7, -0.2])
neutral_vec = np.array([0, 0])


def emotion_to_A_V_vec(vision_vec):
    A_V_vec = vision_vec[0]*anger_vec + vision_vec[1]*disgust_vec + vision_vec[2]*fear_vec +\
        vision_vec[3]*Happy_vec + vision_vec[4]*sad_vec + vision_vec[5]*Surprise_vec +\
        vision_vec[6]*neutral_vec*np.array([0, 0])
    return A_V_vec
######################################


data_PATH = "./MMD/"
df1 = pd.read_csv(data_PATH + f"MMD.csv")
print(df1)
df2 = df1[['arousal', 'valence']]
a = df2['arousal'] * 2 - 1
b = df2['valence'] * 2 - 1
df4 = pd.concat([b, a], axis=1)

val = np.array(df4['valence'])
ener = np.array(df4['arousal'])
table1 = pd.DataFrame({
    'valence': val,
    'arousal': ener
})

# happy
#valence_min = 0.58
#valence_max = 0.62
#energy_min = 0.8
#energy_max = 0.9

table2 = table1.loc[(table1['valence'] > float(sys.argv[1])) & (table1['valence'] < float(sys.argv[2])) & (
    table1['arousal'] > float(sys.argv[3])) & (table1['arousal'] < float(sys.argv[4]))]

print(table2)
index_list = list(table2.index)
df5 = df1.loc[index_list]
df5.to_csv(f'./filtered_MMD/MMD.csv', index=False)
print(sys.argv)

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms.transforms as transforms
import torchvision.transforms as T

class FER2013(Dataset):
    """
    FER2013의 Custom Dataset.
    
    FER2013 데이터셋 탐색하기.ipynb 참고하여 작성하기
    """
    def __init__(self, path='./data', mode = 'train', transform = None):
        ## 여기에 코드 작성
        self.path = path + '/fer2013.csv'
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        
        self.transform = transform
        
        # FER2013 데이터셋 불러오기 (pd.read_csv)
        # self.data에 불러온 데이터 프레임 저장
        self.data = pd.read_csv(self.path)
        
        # mode에 따라 데이터 구분
        # train <= Training / val <= PrivateTest / test <= PublicTest
        if self.mode == 'train':
            self.data = self.data[self.data['Usage'] == 'Training']
        elif self.mode == 'val':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']
        else:
            self.data = self.data[self.data['Usage'] == 'PublicTest']







            
    def __len__(self) -> int:        
        ## 여기에 코드 작성

        return self.data.index.size
    
    def __getitem__(self, index: int):
        ## 여기에 코드 작성

        """
        반환 값 (Returns)
        ----------------
        (numpy.ndarray, numpy.int64). index에 대응되는 이미지와 emotion
        """
        # 전체 Dataframe에서 index번째 행에 있는 값들만 추출
        item = self.data.iloc[index]
        
        emotion = item['emotion'] # numpy.int64
        pixels  = item['pixels'] # str        
        # str인 pixels로부터 numpy.ndarray 얻기
        face = list(map(int, pixels.split(' '))) # char 값들에 int 함수를 적용 후 list 변환
        face = np.array(face).reshape(48,48).astype(np.uint8) # 48 x 48 size의 numpy 배열로 변환
        
        # 추가. transform 적용
        if self.transform:
            # 학습 진행을 원활히 하기 위해 히스토그램 평활화를 적용
            face = cv2.equalizeHist(face)
            face = self.transform(face)
        
        return face, emotion



        

def create_train_dataloader(root='./data', batch_size=16):
    """
    train용 dataloader 함수
    
    FER2013 데이터셋 탐색하기.ipynb 참고하여 작성하기
    """
    ## 여기에 코드 작성
    transform = transforms.Compose([transforms.ToPILImage(),
                                    
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=(-20,20)),
                                    transforms.ToTensor()])
    
    dataset = FER2013(root, mode='train', transform=transform)
    
    
    dataloader = DataLoader(dataset, batch_size,shuffle=True)
    return dataloader



    
def create_val_dataloader(root='./data', batch_size=16):
    """
    validation용 dataloader 함수
    
    FER2013 데이터셋 탐색하기.ipynb 참고하여 작성하기
    """
    ## 여기에 코드 작성
    transform = transforms.Compose([transforms.ToPILImage(),
                                    
                                    transforms.ToTensor()])
    
    dataset = FER2013(root, mode='val', transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader



    
def create_test_dataloader(root='./data', batch_size=16):
    """
    test용 dataloader 함수
    
    FER2013 데이터셋 탐색하기.ipynb 참고하여 작성하기
    """
    ## 여기에 코드 작성
    transform = transforms.Compose([transforms.ToPILImage(),
                                  
                                    transforms.ToTensor()])
    
    dataset = FER2013(root, mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader




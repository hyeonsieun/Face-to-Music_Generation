from torchsummary import summary as summary
from torch.nn.modules.activation import ReLU

import torch
import torch.nn as nn


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    """
    합성곱 연산, Batch Normalization, ReLU 활성함수를 연속적으로 거치도록 하는 Sequential을 return하는 함수

    매개변수(Parameters)
    ----------------------
    in_channels: int형, 입력으로 들어오는 이미지의 채널 개수
    out_channels: int형, 출력으로 반환할 이미지의 채널 개수
    kernel_size: int 혹은 tuple 형, 사용할 필터의 크기 정보, default 값 3
    stride: int 혹은 tuple 형, 스트라이드 값, defalut 값 1
    padding: int 혹은 tuple 혹은 str형, 패딩에 대한 정보, defalut 값 0(패딩 없음)

    반환 값(Returns)
    ----------------------
    합성곱 연산, Batch Normalization, ReLU 활성함수를 연속적으로 거치도록 하는 torch.nn.Sequential
    """
    return nn.Sequential(
        # 입력받은 매개변수에 따라, 합성곱 연산을 진행
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        # 배치 정규화 진행
        nn.BatchNorm2d(out_channels),
        # 활성함수인 ReLU 함수 거치기
        nn.ReLU(inplace=True)
    )


def SeparableConv2D(in_channels, out_channels, kernel=3):
    """
    Separable Convolution 연산을 진행하는 Sequential을 return하는 함수

    매개변수(Parameters)
    ----------------------
    in_channels: int형, 입력으로 들어오는 이미지의 채널 개수
    out_channels: int형, 출력으로 반환할 이미지의 채널 개수
    kernel: int 혹은 tuple 형, 사용할 필터의 크기 정보, default 값 3

    반환 값(Returns)
    ----------------------
    Separable Convolution 연산을 진행하는 Sequential
    """
    return nn.Sequential(
        # 입력받은 채널의 개수를 보존하는 형태로, 채널 수가 1인 필터를 합성곱 연산
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel,
                  stride=1, groups=in_channels, padding=1, bias=False),
        # 1x1 합성곱 연산 진행, 채널 개수를 원하는 출력 채널 개수로 조정
        nn.Conv2d(in_channels, out_channels, kernel_size=1,
                  stride=1, bias=False),
    )


class ResidualXceptionBlock(nn.Module):
    """
    MiniXception 구조의 핵심인 부분으로, 잔차 연결로 두 경로의 계층 연산을 합해주는 클래스
    """

    def __init__(self, in_channels, out_channels, kernel=3):
        """
        필요한 계층을 정의하는 부분들이 담긴 생성자(constructor)

        매개변수(Parameters)
        ----------------------
        in_channels: int형, 입력으로 들어오는 이미지의 채널 개수
        out_channels: int형, 출력으로 반환할 이미지의 채널 개수
        kernel: int 혹은 tuple 형, 사용할 필터의 크기 정보, default 값 3
        """
        # 상속받은 nn.Module 클래스의 생성자 호출
        super().__init__()

        # 그림 21에서, ResidualXceptionBlock을 구성하는 4개의 사각형 중 좌측 가장 하단
        # 이 계층에서의 합성곱 연산을 통해 출력 데이터의 채널 수로 조정
        # Separable convolution 연산을 진행하는 계층, Sequential 형태. 앞서 정의한 함수 활용
        # 여기에 코드 작성
        self.depthwise_conv1 = SeparableConv2D(
            in_channels, out_channels, kernel)
        # 배치 정규화 진행
        # 여기에 코드 작성
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 활성함수로 ReLU 함수를 사용
        # 여기에 코드 작성
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)
        # 그림 21에서, ResidualXceptionBlock을 구성하는 4개의 사각형 중 좌측 가운데
        # 앞선 계층에서 출력 데이터의 채널 개수를 맞추었으므로, 채널 개수에는 변화를 주지 않음
        # Separable convolution 연산을 진행하는 계층, Sequential 형태. 앞서 정의한 함수 활용
        # 여기에 코드 작성
        self.depthwise_conv2 = SeparableConv2D(
            out_channels, out_channels, kernel)
        # 배치 정규화 진행
        # 여기에 코드 작성
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 그림 21에서, ResidualXceptionBlock을 구성하는 4개의 사각형 중 좌측 상단
        # 최대 풀링 진행
        # 여기에 코드 작성

        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(0.1)
        # 그림 21에서, ResidualXceptionBlock을 구성하는 4개의 사각형 중 우측
        # 잔차 연결을 진행하기 위한 갈래의 계층을 구현
        # 합성곱 연산 진행, 필요한 출력 데이터의 채널 수가 되도록 조정
       # 여기에 코드 작성
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # 배치 정규화 진행
        # 여기에 코드 작성
        self.residual_bn = nn.BatchNorm2d(out_channels)
        self.residual_relu = nn.ReLU(inplace=True)
        self.residual_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)
        self.residual_dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        """
        순전파를 진행하도록 하는 함수

        매개변수(Parameters)
        ----------------------
        x: Tensor, 입력 데이터
        """
        # 그림 21에서, ResidualXceptionBlock을 구성하는 두 갈래 중 우측
        # 우측 갈래를 따른 순전파 진행 결과를 residual에 저장
        # 여기에 코드 작성
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        residual = self.residual_relu(residual)
        residual = self.residual_maxpool(residual)
        residual = self.residual_dropout(residual)
        # 그림 21에서, ResidualXceptionBlock을 구성하는 두 갈래 중 좌측
        # 우측 갈래를 따른 순전파 진행 결과를 x에 저장
        # 여기에 코드 작성
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        # 두 갈래의 순전파 결과를 합한 Tensor 결과를 반환하여 잔차 연결 구현
        return x + residual


class Mini_Xception(nn.Module):
    """
    전체 MiniXception 구조를 구현한 클래스
    """

    def __init__(self):
        """
        필요한 계층을 정의하는 부분들이 담긴 생성자(constructor)
        """
        # 상속받은 nn.Module 클래스의 생성자 호출
        super().__init__()

        # 그림 21에서, ResidualXceptionBlock 이전의 합성곱-배치 정규화-ReLU 계층
        # 여기에 코드 작성
        self.conv1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0)
        # 그림 21에서, ResidualXceptionBlock 4개를 모아둔 부분
        # 순전파가 진행될수록, 이미지 채널 개수를 2배씩 증가
        # 이전 계층의 출력 채널 개수가 다음 채널의 입력 채널의 개수와 동일하도록 설정 필요
        # 채널 개수를 증가시키는 점에 대한 기본적인 아이디어가 궁금하다면, VGGNet이라는 구조에 대해 찾아보자.
        # 여기에 코드 작성

        self.residual_blocks = nn.ModuleList([
            ResidualXceptionBlock(8, 16),
            ResidualXceptionBlock(16, 32),
            ResidualXceptionBlock(32, 64),
            ResidualXceptionBlock(64, 128)
        ])

        # 그림 21에서, ResidualXceptionBlock 4개를 거친 이후의 부분
        # 합성곱 계층
        # Global Average Pooling 단계로 넘어가기 직전의 채널 수를 7로 설정함에 주목할 것.
        # 여기에 코드 작성
        self.conv3 = nn.Conv2d(128, 7, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(7)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout2d(0.1)

        # Global Average Pooling 계층
        # 여기에 코드 작성
        self.fc1 = nn.Linear(7*44*44, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_relu1 = nn.ReLU(inplace=True)
        self.fc_maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc_dropout1 = nn.Dropout1d(0.1)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc_relu2 = nn.ReLU(inplace=True)
        self.fc_maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc_dropout2 = nn.Dropout1d(0.1)

        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        """
        순전파를 진행하도록 하는 함수

        매개변수(Parameters)
        ----------------------
        x: Tensor, 입력 데이터
        """
        # 그림 21에서, ResidualXceptionBlock 이전의 순전파
        # 여기에 코드 작성
        x = self.conv1(x)

        x = self.conv2(x)

        # 그림 21에서, ResidualXceptionBlock 4개에 대한 순전파
        # ModuleList를 순회하면서, 저장된 계층들을 차례로 적용하도록 함
        # 여기에 코드 작성
        for block in self.residual_blocks:
            x = block(x)

        # 그림 21에서, ResidualXceptionBlock 이후의 합성곱 계층
        # 여기에 코드 작성
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        # 그림 21에서, Global Average Pooling
        # 여기에 코드 작성
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = self.fc_relu1(x)
        x = self.fc_maxpool1(x)
        x = self.fc_dropout1(x)

        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = self.fc_relu2(x)
        x = self.fc_maxpool2(x)
        x = self.fc_dropout2(x)

        x = self.fc3(x)
        return x


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*1*1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        # 여기에 코드 작성
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            # 3 224 224
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        self.avg_pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()

        # 512 1 1
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        #x = self.softmax(x)
        return x

############################
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet_50(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(ResNet_50, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # PyTorch v0.4.0
    model = VGG16().to(device)
    summary(model, (1, 224, 224))

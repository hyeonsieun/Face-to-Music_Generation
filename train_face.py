# TensorBoard를 사용하면 손실 및 정확도와 같은 측정 항목을 추적 및 시각화하는 것, 모델 그래프를 시각화하는 것, 히스토그램을 보는 것, 이미지를 출력하는 것 등이 가능
import gc
import matplotlib.pyplot as plt
from torch.nn import Parameter
from torch import nn
import math
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
import numpy as np

import os

from model_face import Mini_Xception, NN, VGG16, ResNet_50, BottleneckBlock
from data_face import create_train_dataloader, create_val_dataloader

import torch
import torch.nn as nn
import torch.optim

from sklearn.metrics import accuracy_score
import random

torch.cuda.empty_cache()

seed_val = 2022
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# tensorboard 만들기(epoch마다 loss와 정확도 변화의 경향을 파악하는 데 좋다.) -> 선택 사항
writer = tensorboard.SummaryWriter('checkpoint/model_weights_NN/')
# 모델 정의하기
# 여기에 코드 작성
# 모델 학습 파라미터 지정 # todo code
learning_rate = 1e-1  # 무조건 작다고 좋지 않다.
epochs = 300

model = NN().to(device)  # 모델 instantiation for training


# 옵티마이저 정의
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,  # 학습률
                             eps=1e-8,  # 0으로 나누는 것을 방지하기 위한 epsilon 값,
                             )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                       factor=0.75, patience=5, verbose=True)

loss_fn = FocalLoss()

model.zero_grad()  # 그래디언트 초기화
PATH = "./checkpoint/model_weights_NN/weights_epoch_"

# image_size=[128,192,224,380]
# magnitude=[5,10,15,20]


def main():  # 코드가 실행될 main 함수를 만들어봅시다.

    train_accs = []
    train_losses = []
    val_losses = []
    val_accs = []
    lrs = []
    train_dataloader = create_train_dataloader(root='./data', batch_size=256)
    val_dataloader = create_val_dataloader(root='./data', batch_size=256)
    val_acc_present = 0.65
    val_acc_next = 0
    print("-------Training!---------")

    for epoch in range(0, epochs):
        train_loss, train_acc = train(model,   train_dataloader, epoch)
        val_loss, val_acc = validation(model,  val_dataloader, epoch)

        scheduler.step(val_acc)
        lrs.append(
            optimizer.param_groups[0]["lr"]
        )
        print(f"lr : {lrs[-1]}")
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        val_acc_next = val_acc
        #final_loss, val_acc=validation(model,val_dataloader,epoch)
        # 매 epoch 마다 checkpoint로 model을 저장할 필요가 있습니다.
        # https://tutorials.pytorch.kr/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        # 여기에 코드 작성
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        checkpoint_state = {
            'NN': model.state_dict(),
            "epoch": epoch
        }
        if (val_acc_next > val_acc_present):
            torch.save(checkpoint_state, PATH+str(epoch+1)+'.pth.tar')
            print(f'이전 best acc : {val_acc_present}')
            val_acc_present = val_acc_next
            print('weight 저장됨!')
    plt.figure(1)
    plt.plot(np.arange(1,epochs+1),train_accs)
    plt.plot(np.arange(1,epochs+1),val_accs)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')
    
    plt.savefig('Mini_Accuracy_graph.png',
            format='png', dpi=200)
    plt.figure(2)
    plt.plot(np.arange(1,epochs+1),train_losses)
    plt.plot(np.arange(1,epochs+1),val_losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')

    
    plt.savefig('Mini_Loss_graph.png',
            format='png', dpi=200)
    
    
    
    plt.figure(3)
    plt.plot(np.arange(1,epochs+1),lrs)
    plt.xlabel('epoch')
    plt.ylabel('learning_rate')
    plt.title('ReduceLROnPlateau')
    plt.savefig('Mini_learning_rate.png',
            format='png', dpi=200)
    
    
    
    
    
    writer.close() # Tensorboard 닫기

def train(model, dataloader, epoch):  # train 과정에서의 손실을 계산하는 함수 작성
    model.train()
    model.to(device)
    # 여기에 코드 작성
    # 더 필요한 변수나 작업이 있다면 작성해봅시다. 정해진 틀은 없으며 자유롭게 작성하시면 됩니다. loss나 optimizer로 어떤 것을 사용할건지 등..

    losses = []
    train_total_pred = []
    train_total_labels = []
    print("")
    print('========{:}번째 Epoch / 전체 {:}회 ========'.format(epoch + 1, epochs))
    print('훈련 중')

    for images, labels in dataloader:
        # 여기에 코드 작성
        # (batch, 1, 48, 48) -> images.shape[0]을 mini batch size로 나중에 설정
        images = images.to(device)
        labels = labels.to(device)  # (batch,)
        outputs = model(images)  # -> 예측값

        # (batch, 7, 1, 1) -> (batch, 7)
        # Returns a tensor with all the dimensions of input of size 1 removed.
        outputs = torch.squeeze(outputs)
        loss = loss_fn(outputs, labels)  # 예측값과 label간 loss
        losses.append(loss.cpu().item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        _, indexes = torch.max(outputs, axis=1)

        train_total_pred.extend(indexes.cpu().detach().numpy())
        train_total_labels.extend(labels.cpu().detach().numpy())

        # 여기에 코드 작성
        # back propagation을 떠올려보며 train 코드를 마무리하기
    # 평균 로스 계산
    Train_loss = np.mean(losses).item()
    Train_acc = accuracy_score(train_total_labels, train_total_pred)

    print(
        f'training @ epoch {epoch+1} .. Train loss = {Train_loss} Train Accuracy = {Train_acc} ')

    #### 검증 ####

    print("")
    print("검증 중")

    return Train_loss, Train_acc


def validation(model, dataloader, epoch):
  # 중간미션 1에서 with torch.no_grad(): 부분에 작성하였던 test accuracy를 구하는 방법을 떠올려보면 됩니다.
  # train 과정에서의 손실도 중요하나 최종적으로 random data을 넣어서 test를 한 후의 정확도를 얻어야 합니다.
    model.eval()  # 왜 설정할까요?
    model.to(device)
    # 여기에 코드 작성
    # 변수 필요한거 설정 (loss로 어떤걸 사용할건지 등)
    losses = []

    val_total_pred = []
    val_total_labels = []

    with torch.no_grad():
        # 학습이 x 모델 성능 평가 따라서 autograd하지 않음!!
        correct = 0  # 맞은 개수 세기 위해 int 정의
        total = 0  # 전체 실행 개수를 세서 정확도를 계산하기 위해 int 정의
        for images, labels in dataloader:
            minibatch = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            # 이제 images을 모델에 통과시켜 얻은 예측값으로 loss을 구해야 합니다.
            # 이를 criterion 함수에 넣기 위해선 먼저 예측값을 (minibatch, # of class)의 형태로 shape을 맞춰주어야 합니다. minibatch의 수를 알 수 있다면 편합니다!
            # 여기에 코드 작성

            # # ============== Evaluation ===============
            # index of the max value of each sample (shape = (batch,))

            outputs = model(images)
            outputs = torch.squeeze(outputs)
            # mini_batch로 shape 맞춤(계산위해서)
            outputs = outputs.reshape(minibatch, -1)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            losses.append(loss.cpu().item())

            # 이제 model을 거쳐 얻은 텐서에서 가장 큰 값을 갖는 곳의 index을 알아야 emotion이 무엇인지 알 수 있습니다.
            # index을 구해보고 이를 통해 index와 label간 차이로 정확도를 구합시다. 참고: https://pytorch.org/docs/stable/generated/torch.max.html
            # Scikit learn의 accuracy_score을 이용하고 싶다면 어떤식으로 예측값과 정답을 처리해야 할 지 고민해 봅시다.
            # 여기에 코드 작성
            _, indexes = torch.max(outputs, axis=1)
            # print(indexes.shape, labels.shape)
            val_total_pred.extend(indexes.cpu().detach().numpy())
            val_total_labels.extend(labels.cpu().detach().numpy())

        val_loss = np.mean(losses).item()
        val_acc = accuracy_score(val_total_labels, val_total_pred)

        # 최종 loss와 정확도(소수점 넷째 자리까지)를 출력해봅시다.
        print(f'Val loss = {val_loss} .. Val Accuracy = {val_acc} ')

        return val_loss, val_acc


if __name__ == "__main__":  # 앞서 작성한 main 함수 실행
    main()


gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_reserved()

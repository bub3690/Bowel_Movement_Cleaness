# # baseline
# 
# - classification baseline code
# 사용법: python baseline_train.py --batch-size 32 --epochs 40 --lr 0.0001 --sublabel label --wandb True --project-name BMC_vision_classification
# 
# 


import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
from torchvision import transforms, datasets
import torchvision.models as models
import cv2
from torchvision.ops import masks_to_boxes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import sys
from tqdm import tqdm

from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import argparse
import wandb


p = os.path.abspath('../../Utils') # 상위 폴더를 사용하기 위해서.
sys.path.insert(1, p)
from pytorchtools.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.

# # 데이터 정의


class BowelDataset(Dataset):
    def __init__(self, data_path_list,label_df,to_tensor,transform,sublabel):
        self.data_path_list = data_path_list
        self.label_df = label_df
        self.to_tensor = to_tensor
        self.transform = transform
        self.sublabel = sublabel #sublabel : color,residue,turbidity,label

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=self.to_tensor(image)
        
        if self.transform:
            #1. 이미지 사이즈 변환
            image=self.transform(image).type(torch.float32)# 이미지 0~1 정규화
        return image, torch.tensor(self.label_df.iloc[idx][self.sublabel])

class BowelDatasetSegment(Dataset):
    def __init__(self, data_path_list,mask_path_list,label_df,to_tensor,transform,sublabel):
        self.data_path_list = data_path_list
        self.label_df = label_df
        self.to_tensor = to_tensor
        self.transform = transform
        self.sublabel = sublabel #sublabel : color,residue,turbidity,label
        self.mask_path_list = mask_path_list

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=self.to_tensor(image)

        mask_path = self.mask_path_list[idx]
        mask = cv2.imread(mask_path)
        mask.sum(axis=2)
        mask = mask.sum(axis=2) > 0
        mask=self.to_tensor(mask)

        box = masks_to_boxes(mask)[0] # (xmin, ymin, xmax, ymax) 
        box= box.to(torch.int32)
        mask=mask[0,box[1]:box[3],box[0]:box[2]]
        image=image[:,box[1]:box[3],box[0]:box[2]]

        #여기서 MASK 확장
        mask = mask.expand(3,-1,-1)
        image = mask * image
        
        if self.transform:
            #1. 이미지 사이즈 변환
            image=self.transform(image).type(torch.float32)# 이미지 0~1 정규화
        return image, torch.tensor(self.label_df.iloc[idx][self.sublabel])

def load_dataloader(add_seg,X,Y_df,sublabel,BATCH_SIZE,mask_path_list=None):
    if add_seg==False:
        loader = torch.utils.data.DataLoader(dataset = 
                                                BowelDataset(X,
                                                            Y_df,
                                                            to_tensor = transforms.ToTensor(),
                                                            transform = torch.nn.Sequential(
                                                                    transforms.Resize([512,512]),
                                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                ),
                                                            sublabel=sublabel # color,residue,turbidity, label 중 어느것을 맞추려는지 입력.
                                                            ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                num_workers=0
                                                ) # 순서가 암기되는것을 막기위해.
    else:
        #segmentation을 사용하는 경우
        loader = torch.utils.data.DataLoader(dataset = 
                                                BowelDatasetSegment(X,
                                                            mask_path_list,
                                                            Y_df,
                                                            to_tensor = transforms.ToTensor(),
                                                            transform = torch.nn.Sequential(
                                                                    transforms.Resize([512,512]),
                                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                                ),
                                                            sublabel=sublabel # color,residue,turbidity, label 중 어느것을 맞추려는지 입력.
                                                            ),
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                num_workers=0
                                                ) # 순서가 암기되는것을 막기위해.        
    return loader


# # 모델 설계
# 
# - 기본 resnet18


# pretrained

class ResLayer(nn.Module):
    def __init__(self,sublabel_count,DEVICE):
        super(ResLayer, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)
        self.num_ftrs = self.model.fc.out_features
        
        self.fc = nn.Sequential(
            #nn.BatchNorm1d(self.num_ftrs+self.n_mfcc),                
            nn.Linear(self.num_ftrs, 64),
                             nn.BatchNorm1d(64),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(64,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(50,sublabel_count)
                            )
        

    def forward(self, x):
        x = self.model(x)
        x  = self.fc(x)
        return x
    
def model_initialize(sublabel_count,DEVICE):
    model = ResLayer(sublabel_count,DEVICE).to(DEVICE)
    return model


#8. 학습
def train(model,train_loader,optimizer,criterion,DEVICE):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx,(image,label) in tqdm(enumerate(train_loader)):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        #데이터들 장비에 할당
        optimizer.zero_grad() # device 에 저장된 gradient 제거
        output = model(image) # model로 output을 계산
        loss = criterion(output, label) #loss 계산
        train_loss += loss.item()
        prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
        correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
        loss.backward() # loss 값을 이용해 gradient를 계산
        optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    train_loss/=len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss,train_accuracy



#9. 학습 진행하며, validation 데이터로 모델 성능확인
def evaluate(model,valid_loader,criterion,DEVICE):
    model.eval()
    valid_loss = 0
    correct = 0
    #no_grad : 그래디언트 값 계산 막기.
    with torch.no_grad():
        for image,label in valid_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            valid_loss += criterion(output, label).item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            #true.false값을 sum해줌. item
        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = 100. * correct / len(valid_loader.dataset)
        return valid_loss,valid_accuracy

# # test


#confusion matrix 계산
#test set 계산.
def test_evaluate(model,test_loader,criterion,DEVICE):
    model.eval()
    test_loss = 0
    predictions = []
    answers = []
    #no_grad : 그래디언트 값 계산 막기.
    with torch.no_grad():
        for image,label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            answers +=label
            predictions +=prediction
            
        return predictions,answers,test_loss

def get_num(file_str):
    return int(file_str.split("\\")[-1].split(".")[0])

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--sublabel',type=str, default='label',
                        help='select one of [color,residue,turbidity,label]')                        
    parser.add_argument('--wandb',type=bool, default=False,
                        help='Use wandb log')                        
    parser.add_argument('--model',type=str, default='res18',
                        help='select one of [color,residue,turbidity,label]')
    parser.add_argument('--add-seg',type=bool, default=False,
                        help='use annotations')
    parser.add_argument('--descript',type=str, default='baseline',
                            help='write descript for wandb')
    parser.add_argument('--project-name',type=str, default='BMC_vision_classification',
                            help='project name for wandb')

    args = parser.parse_args()

    if args.wandb:
        project_name = args.project_name
        wandb.init(project=project_name, entity="bub3690")
        wandb_run_name = args.model+'_512x512'+args.descript+'_classification'+'_segment_'+str(args.add_seg)
        wandb.run.name = wandb_run_name
        wandb.run.save()


    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    #DEVICE = torch.device('cpu')
    print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)

    #3. 하이퍼 파라미터
    BATCH_SIZE =  args.batch_size #한 배치당 16개 이미지
    EPOCHS = args.epochs # 전체 데이터 셋을 50번 반복
    lr = args.lr

    #어떤 타겟을 맞출지
    sublabel = args.sublabel

    # # 데이터 분류
    # ## train / valid / test
    #1. train, test 나누기

    X_train = glob('../../data/bmc_label_voc_split/images/train/*.jpg')
    X_valid = glob('../../data/bmc_label_voc_split/images/val/*.jpg')

    if args.add_seg==True:
        X_train_mask = glob('../../data/bmc_label_voc_split/annotations/train/*.png')
        X_valid_mask = glob('../../data/bmc_label_voc_split/annotations/val/*.png')
   


    X_train_name=list(map(get_num,X_train))
    X_valid_name=list(map(get_num,X_valid))

    #첫번째 열 이름 바꿔주기
    label_df = pd.read_csv('../../bmc.csv')
    column_names = list(label_df.columns)
    column_names[0]='file_name'
    label_df.columns = column_names

    # Y값 찾아오기
    Y_train_df=pd.merge(pd.DataFrame(X_train_name,columns=['file_name']),label_df,left_on='file_name',right_on='file_name',how='inner')
    Y_valid_df=pd.merge(pd.DataFrame(X_valid_name,columns=['file_name']),label_df,left_on='file_name',right_on='file_name',how='inner')
    print("---")
    print("훈련 셋 : ",len(Y_train_df),Counter(Y_train_df['label']))
    print("검증 셋 : ",len(Y_valid_df),Counter(Y_valid_df['label']))
    print("---")

    
    if args.add_seg==False:
        train_loader = load_dataloader(args.add_seg,X_train,Y_train_df,sublabel,BATCH_SIZE)
        valid_loader = load_dataloader(args.add_seg,X_valid,Y_valid_df,sublabel,BATCH_SIZE)
        test_loader = load_dataloader(args.add_seg,X_valid,Y_valid_df,sublabel,BATCH_SIZE)
    else:
        #segmentation을 사용한 경우
        train_loader = load_dataloader(args.add_seg,X_train,Y_train_df,sublabel,BATCH_SIZE,X_train_mask)
        valid_loader = load_dataloader(args.add_seg,X_valid,Y_valid_df,sublabel,BATCH_SIZE,X_valid_mask)
        test_loader = load_dataloader(args.add_seg,X_valid,Y_valid_df,sublabel,BATCH_SIZE,X_valid_mask)

    sublabel_count=len(set(label_df[sublabel]))
    # 학습 
    check_path = './checkpoint/baseline_'+'get_'+args.sublabel+'_'+args.model+'_512_'+'segment_'+str(args.add_seg)+'.pt'
    print(check_path)
    early_stopping = EarlyStopping(patience = 3, verbose = True, path=check_path)

    best_train_acc=0 # accuracy 기록용
    best_valid_acc=0

    model=model_initialize(sublabel_count,DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    print("학습 시작")
    for Epoch in range(1,EPOCHS+1):
        train_loss,train_accuracy = train(model,train_loader,optimizer,criterion,DEVICE)
        valid_loss,valid_accuracy = evaluate(model, valid_loader,criterion,DEVICE)

        print("\n[EPOCH:{}]\t Train Loss:{:.4f}\t Train Acc:{:.2f} %  | \tValid Loss:{:.4f} \tValid Acc: {:.2f} %\n".
            format(Epoch,train_loss,train_accuracy,valid_loss,valid_accuracy))
        wandb.log({
            "valid Accuracy": valid_accuracy,
            "valid loss": valid_loss},
            commit=True,
            step=Epoch)

        early_stopping(valid_loss, model)
        if -early_stopping.best_score == valid_loss:
            best_train_acc, best_valid_acc = train_accuracy,valid_accuracy
            wandb.run.summary.update({"best_valid_acc" : best_valid_acc})
            wandb.run.summary.update({"best_valid_loss" : valid_loss})

        if early_stopping.early_stop:
                #train_accs.append(best_train_acc)
                #valid_accs.append(best_valid_acc)
                print("Early stopping")
                break

        if Epoch==EPOCHS:
            #만약 early stop 없이 40 epoch라서 중지 된 경우.
            print(EPOCHS," Stop") 
            pass
    

    print("train ACC : {:.4f} |\t valid ACC: {:.4f} ".format(best_train_acc,best_valid_acc ))

    # Confusion matrix (resnet18)
    # 모델을 각각 불러와서 test set을 평가한다.

    model=model_initialize(sublabel_count,DEVICE)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(check_path))

    predictions,answers,test_loss = test_evaluate(model, test_loader,criterion,DEVICE)
    predictions=[ dat.cpu().numpy() for dat in predictions]
    answers=[ dat.cpu().numpy() for dat in answers]

    cf = confusion_matrix(answers, predictions)

    #fscroe macro추가
    fscore = f1_score(answers,predictions,average='macro')
    acc = accuracy_score(answers,predictions)
    wandb.run.summary.update({"valid_acc" : acc*100})
    wandb.run.summary.update({"valid_f1" : fscore})

    print("Accuracy : {:.4f}% ".format(acc*100))
    print("f score : {:.4f} ".format(fscore))
    print(cf)
    print("-----")                        
    return



if __name__ == '__main__':
    main()


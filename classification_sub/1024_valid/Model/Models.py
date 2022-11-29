import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로
import torchvision.models as models

from torchvision.models import ResNet



# # 모델 설계
# 
# - 기본 resnet18


# pretrained

class ResLayer(nn.Module):
    def __init__(self,sublabel_count,DEVICE):
        super(ResLayer, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)
        self.num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
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
        x  = self.model(x)
        # x  = self.fc(x)
        x  = F.log_softmax(x,dim=1) 
        return x

class ResLayer_multilabel(nn.Module):
    def __init__(self,sublabel_count,DEVICE):
        super(ResLayer_multilabel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)
        self.num_ftrs = self.model.fc.out_features
        
        
        
        self.fc1 = nn.Sequential(
            #nn.BatchNorm1d(self.num_ftrs+self.n_mfcc),                
            nn.Linear(self.num_ftrs, 64),
                             nn.BatchNorm1d(64),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                            )
        self.fc2 = nn.Sequential(
            #nn.BatchNorm1d(self.num_ftrs+self.n_mfcc),                
                             nn.Linear(64+3,50),
                             nn.BatchNorm1d(50),
                             nn.ReLU(),
                             nn.Linear(50,sublabel_count)
                            )
        

    def forward(self, x, sublabel):
        x = self.model(x)
        x  = self.fc1(x)
        x  = self.fc2(torch.cat([x,sublabel],axis=1))
        x = F.log_softmax(x,dim=1)
        return x

class ResLayer_multilabel_ver2(nn.Module):
    def __init__(self,sublabel_count,DEVICE):
        super(ResLayer_multilabel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)
        self.num_ftrs = self.model.fc.out_features
        
        self.residue = nn.Sequential(
                        nn.Linear(self.num_ftrs, 3),
                    )

        self.color = nn.Sequential(
                        nn.Linear(self.num_ftrs, 3),
                    )
        self.turbidity = nn.Sequential(
                        nn.Linear(self.num_ftrs, 2),
                    )

        self.fc1 = nn.Sequential(              
                        nn.Linear(self.num_ftrs + 8, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),        
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Linear(50,3)
                    )

    def forward(self, x, sublabel):
        
        back = self.model(x)

        res = self.residue(back)
        col = self.color(back)
        tur = self.turbidity(back)
        #x  = self.fc1(torch.cat([back,res,col,tur],axis=1))
        return res,col,tur


def model_initialize(sublabel_count,DEVICE,multilabel=False):
    if multilabel:
        model = ResLayer_multilabel(sublabel_count,DEVICE).to(DEVICE)
    else:
        model = ResLayer(sublabel_count,DEVICE).to(DEVICE)
    return model
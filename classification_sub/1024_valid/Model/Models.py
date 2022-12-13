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

class ResLayer_multilabel_stage1(nn.Module):
    def __init__(self,DEVICE):
        super(ResLayer_multilabel_stage1, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)
        self.num_ftrs = self.backbone.fc.out_features
        
        self.residue = nn.Sequential(
                        nn.Linear(self.num_ftrs, 3),
                    )

        self.color = nn.Sequential(
                        nn.Linear(self.num_ftrs, 3),
                    )
        self.turbidity = nn.Sequential(
                        nn.Linear(self.num_ftrs, 2),
                    )


    def forward(self, x):
        
        back = self.backbone(x)

        res = self.residue(back)
        col = self.color(back)
        tur = self.turbidity(back)
        #x  = self.fc1(torch.cat([back,res,col,tur],axis=1))
        return res,col,tur

class ResLayer_multilabel_stage2_base(nn.Module):
    def __init__(self,sublabel_count,DEVICE,check_point):
        super(ResLayer_multilabel_stage2_base, self).__init__()
        #여기서 모델 체크포인트 읽어오기.
        
        self.model = ResLayer_multilabel_stage1(DEVICE).to(DEVICE)
        self.model.load_state_dict(torch.load(check_point))
        #backbone freeze
        #for param in self.model.backbone.parameters():
        #    param.requires_grad = False
        

        #self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)

        self.num_ftrs = self.model.backbone.fc.out_features
        
        self.fc1 = nn.Sequential(              
                        nn.Linear(self.num_ftrs, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),        
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Linear(50,sublabel_count)
                    )

    def forward(self, x):
        
        back = self.model.backbone(x)

        #의문, res,col,tur는 backpropagation이 작동할까? forawrd를 통과안했는데? 안될것으로 추정.
        
        res = self.model.residue(back)
        col = self.model.color(back)
        tur = self.model.turbidity(back)

        x  = self.fc1(back)
        return x,res,col,tur

# class ResLayer_multilabel_stage2(nn.Module):
#     def __init__(self,sublabel_count,DEVICE,check_point):
#         super(ResLayer_multilabel_stage2, self).__init__()
#         #여기서 모델 체크포인트 읽어오기.
        
#         self.model = ResLayer_multilabel_stage1(DEVICE).to(DEVICE)
#         self.model.load_state_dict(torch.load(check_point))
#         #head freeze
#         for param in self.model.residue.parameters():
#             param.requires_grad = False
#         for param in self.model.color.parameters():
#             param.requires_grad = False
#         for param in self.model.turbidity.parameters():
#             param.requires_grad = False        

#         #self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)

#         self.num_ftrs = self.model.backbone.fc.out_features
        
#         self.fc1 = nn.Sequential(              
#                         nn.Linear(self.num_ftrs + 8, 64),
#                         nn.BatchNorm1d(64),
#                         nn.ReLU(),
#                         nn.Dropout(p=0.5),        
#                         nn.Linear(64,50),
#                         nn.BatchNorm1d(50),
#                         nn.ReLU(),
#                         nn.Linear(50,sublabel_count)
#                     )

#     def forward(self, x):
        
#         back = self.model.backbone(x)

#         #의문, res,col,tur는 backpropagation이 작동할까? forawrd를 통과안했는데? 안될것으로 추정.
        
#         res = self.model.residue(back)
#         col = self.model.color(back)
#         tur = self.model.turbidity(back)

#         x  = self.fc1(torch.cat([back,res,col,tur],axis=1))
#         return x,res,col,tur

class ResLayer_multilabel_stage2(nn.Module):
    def __init__(self,sublabel_count,DEVICE,check_point):
        super(ResLayer_multilabel_stage2, self).__init__()
        #여기서 모델 체크포인트 읽어오기.
        
        self.model = ResLayer_multilabel_stage1(DEVICE).to(DEVICE)
        if check_point != '':
            self.model.load_state_dict(torch.load(check_point))

        for param in self.model.parameters():
            param.requires_grad = False

        self.model_2stage = ResLayer_multilabel_stage1(DEVICE).to(DEVICE)

        if check_point != '':
            self.model_2stage.load_state_dict(torch.load(check_point))

        #head freeze    

        #self.model = models.resnet18(weights='IMAGENET1K_V1').to(DEVICE)

        self.num_ftrs = self.model.backbone.fc.out_features
        
        self.fc1 = nn.Sequential(              
                        nn.Linear(self.num_ftrs + 8, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),        
                        nn.Linear(64,50),
                        nn.BatchNorm1d(50),
                        nn.ReLU(),
                        nn.Linear(50,sublabel_count)
                    )

    def forward(self, x):
        
        back = self.model.backbone(x)

        #의문, res,col,tur는 backpropagation이 작동할까? forawrd를 통과안했는데? 안될것으로 추정.
        # freeze를 안할시, 학습이 같이 된다. 
        
        res = self.model.residue(back)
        col = self.model.color(back)
        tur = self.model.turbidity(back)

        back = self.model_2stage.backbone(x)


        x  = self.fc1(torch.cat([back,res,col,tur],axis=1))
        return x,res,col,tur



def model_initialize(sublabel_count,DEVICE,model_name='baseline',check_point=''):
    #import pdb;pdb.set_trace()
    if model_name == 'baseline_multi':
        model = ResLayer_multilabel(sublabel_count,DEVICE).to(DEVICE)
    elif model_name == 'sub_1stage':
        model = ResLayer_multilabel_stage1(DEVICE).to(DEVICE)
    elif model_name == 'sub_2stage':
        model = ResLayer_multilabel_stage2(sublabel_count,DEVICE,check_point).to(DEVICE)
    else:
        model = ResLayer(sublabel_count,DEVICE).to(DEVICE)


    return model
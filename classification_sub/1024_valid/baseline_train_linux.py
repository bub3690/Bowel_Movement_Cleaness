# # baseline
# 
# - classification baseline code
# 사용법: python baseline_train.py --batch-size 32 --epochs 40 --lr 0.0001 --sublabel label --multilabel True --augment None --wandb True --seed 1004 --tag baseline_multi --project-name BMC_vision_classification
# 
# 


import torch
import torch.nn as nn # 인공 신경망 모델들 모아놓은 모듈
import torch.nn.functional as F #그중 자주 쓰이는것들을 F로



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import sys
from tqdm import tqdm

from collections import Counter
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # train , test 분리에 사용.

import argparse
import wandb


#Modules
from Model.Models import model_initialize
from Trainer.Trainer import train,train_multilabel,\
    evaluate,test_evaluate,\
    test_evaluate_multilabel,evaluate_multilabel
from Dataset.Dataset import load_dataloader

##



#p = os.path.abspath('../../Utils') # 상위 폴더를 사용하기 위해서.
#sys.path.insert(1, p)
from Utils.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.


# 기타 함수들
def get_num(file_str):
    return int(file_str.split("/")[-1].split(".")[0])






def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BMC baseline')
    parser.add_argument('--batch-size', type=int, default=32, metavar='batch',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='EPOCH',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--sublabel',type=str, default='label',
                        help='select one of [color,residue,turbidity,label]')
    parser.add_argument('--multilabel',type=bool,default=False,
                        help="use sublabel data")                        
    parser.add_argument('--wandb',type=bool, default=False,
                        help='Use wandb log')                        
    parser.add_argument('--model',type=str, default='res18',
                        help='default : res18')
    parser.add_argument('--add-seg',type=bool, default=False,
                        help='use annotations')
    parser.add_argument('--augment',type=str,default='',help='[None,Base,Erasing,BrightnessContrast,SunFlare]')
    parser.add_argument('--descript',type=str, default='baseline',
                            help='write descript for wandb')
    parser.add_argument('--project-name',type=str, default='BMC_vision_classification',
                            help='project name for wandb')
    parser.add_argument('--tag',type=str,nargs='+',default='',help='tag for experiment')

    parser.add_argument('--seed',type=int,default=1004,help='set the validation seed')

    args = parser.parse_args()

    if args.wandb:
        project_name = args.project_name
        wandb.init(project=project_name, entity="bub3690",tags=args.tag)
        wandb_run_name = args.model+'_512x512'+args.descript+'_classification'+'_segment_'+str(args.add_seg)+'_augment_'+args.augment+'_multilabel_'+str(args.multilabel)+'_seed_'+str(args.seed)
        wandb.run.name = wandb_run_name
        wandb.run.save()
        wandb.run.summary.update({"seed" : args.seed,"multilabel":args.multilabel,"augment":args.augment})

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


    if args.add_seg==False:
        X_images = glob('../../data/bmc_label_voc/JPEGImages/*.jpg')
        X_train, X_valid = train_test_split(X_images, test_size=0.2,random_state=args.seed)
    else:
        X_mask = glob('../../data/bmc_label_voc/SegmentationMaskedJPG/*.jpg') # 수정 필요.
        X_train, X_valid = train_test_split(X_mask, test_size=0.2,random_state=1004)

    X_train_name = list(map(get_num,X_train))
    X_valid_name = list(map(get_num,X_valid))

    #첫번째 열 이름 바꿔주기
    label_df = pd.read_csv('../../bmc.csv')
    column_names = list(label_df.columns)
    column_names[0] = 'file_name'
    label_df.columns = column_names

    # Y값 찾아오기
    Y_train_df=pd.merge(pd.DataFrame(X_train_name,columns=['file_name']),label_df,left_on='file_name',right_on='file_name',how='inner')
    Y_valid_df=pd.merge(pd.DataFrame(X_valid_name,columns=['file_name']),label_df,left_on='file_name',right_on='file_name',how='inner')
    
    print("---")
    print("훈련 셋 : ",len(Y_train_df),Counter(Y_train_df['label']))
    print("검증 셋 : ",len(Y_valid_df),Counter(Y_valid_df['label']))
    print("---")
    
    train_loader = load_dataloader(X_train,Y_train_df,sublabel,BATCH_SIZE,args.multilabel,args.augment, is_train = True)
    valid_loader = load_dataloader(X_valid,Y_valid_df,sublabel,BATCH_SIZE,args.multilabel,args.augment, is_train = False)
    test_loader = load_dataloader(X_valid,Y_valid_df,sublabel,BATCH_SIZE,args.multilabel,args.augment, is_train = False)

    
    sublabel_count=len(set(label_df[sublabel]))
    
    # 학습 
    check_path = './checkpoint/baseline_'+'get_'+args.sublabel+'_'+args.model+'_512_'+'segment_'+str(args.add_seg)+'_augment_'+args.augment+'_multilabel_'+str(args.multilabel)+'_seed_'+str(args.seed)+'.pt'
    print(check_path)
    early_stopping = EarlyStopping(patience = 10, verbose = True, path=check_path)

    best_train_acc=0 # accuracy 기록용
    best_valid_acc=0

    model=model_initialize(sublabel_count,DEVICE,args.multilabel)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)


    print("학습 시작")
    for Epoch in range(1,EPOCHS+1):
        if args.multilabel:
            train_loss,train_accuracy = train_multilabel(model,train_loader,optimizer,criterion,DEVICE)
            valid_loss,valid_accuracy = evaluate_multilabel(model, valid_loader,criterion,DEVICE)            
        else:
            train_loss,train_accuracy = train(model,train_loader,optimizer,criterion,DEVICE)
            valid_loss,valid_accuracy = evaluate(model, valid_loader,criterion,DEVICE)

        print("\n[EPOCH:{}]\t Train Loss:{:.4f}\t Train Acc:{:.2f} %  | \tValid Loss:{:.4f} \tValid Acc: {:.2f} %\n".
            format(Epoch,train_loss,train_accuracy,valid_loss,valid_accuracy))

        if args.wandb:
            wandb.log({
                "train Accuracy": train_accuracy,
                "train loss": train_loss,
                "valid Accuracy": valid_accuracy,
                "valid loss": valid_loss,},
                commit=True,
                step=Epoch)

        early_stopping(valid_loss, model)
        if -early_stopping.best_score == valid_loss:
            best_train_acc, best_valid_acc = train_accuracy,valid_accuracy
            if args.wandb:
                wandb.run.summary.update({"best_valid_acc" : best_valid_acc,
                                         "best_valid_loss" : valid_loss})

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

    model=model_initialize(sublabel_count,DEVICE,args.multilabel)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(check_path))

    if args.multilabel:
        predictions,answers,test_loss = test_evaluate_multilabel(model, test_loader,criterion,DEVICE)
    else:
        predictions,answers,test_loss = test_evaluate(model, test_loader,criterion,DEVICE)
    predictions=[ dat.cpu().numpy() for dat in predictions]
    answers=[ dat.cpu().numpy() for dat in answers]

    cf = confusion_matrix(answers, predictions)

    #fscroe macro추가
    fscore = f1_score(answers,predictions,average='macro')
    acc = accuracy_score(answers,predictions)
    wandb.run.summary.update({"last valid_acc" : acc*100,
                              "last valid_f1" : fscore})

    print("Accuracy : {:.4f}% ".format(acc*100))
    print("f score : {:.4f} ".format(fscore))
    print(cf)
    print("-----")                        
    return



if __name__ == '__main__':
    main()


# # baseline
# 
# - classification baseline code
# 사용법: python baseline_train.py --batch-size 32 --epochs 40 --lr 0.0001 --sublabel label --augment None --wandb True --seed 1004 --tag baseline_multi --project-name BMC_vision_classification
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
from Trainer.Trainer import train,evaluate,test_evaluate
from Dataset.Dataset import load_dataloader

##



#p = os.path.abspath('../../Utils') # 상위 폴더를 사용하기 위해서.
#sys.path.insert(1, p)
from Utils.pytorchtools import EarlyStopping # 상위 폴더에 추가된 모듈.


# 기타 함수들
def get_num(file_str):
    return int(os.path.abspath(file_str).replace("\\","/").split("/")[-1].split(".")[0])






def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BMC baseline')
    parser.add_argument('--batch-size', type=int, default=32, metavar='batch',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='numworkers',
                            help='dataloader multiprocess (default: 0)')                        
    parser.add_argument('--epochs', type=int, default=50, metavar='EPOCH',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--sublabel',type=str, default='label',
                        help='select one of [color,residue,turbidity,label]')
    #parser.add_argument('--multilabel',type=bool,default=False,
    #                    help="use sublabel data")                        
    parser.add_argument('--wandb',type=bool, default=False,
                        help='Use wandb log')                        
    parser.add_argument('--model',type=str, default='baseline',
                        help='default : baseline, list : [baseline,baseline_multi,sub_1stage,sub_2stage,GoodAugmentations]') #1201 이거에 맞게 데이터셋, 모델, evaluate 모두 고치기
    parser.add_argument('--add-seg',type=bool, default=False,
                        help='use annotations')
    parser.add_argument('--augment',type=str,default='',help='[None,Base,Erasing,RandomShadow,Flip,BrightnessContrast,SunFlare]')
    parser.add_argument('--descript',type=str, default='baseline',
                            help='write descript for wandb')
    parser.add_argument('--project-name',type=str, default='BMC_vision_classification',
                            help='project name for wandb')
    parser.add_argument('--pretrained-chkpt',type=str,default='',
                            help='pretrained model. only for 2stage learning')
    parser.add_argument('--inference-chkpt',type=str,default='',
                            help='2stage learner chkpoint')
    parser.add_argument('--tag',type=str,nargs='+',default='',help='tag for experiment')
    parser.add_argument('--name',type=str,default='baseline',help='모델에 관한 간단한 추가 설명')

    parser.add_argument('--seed',type=int,default=1004,help='set the validation seed')

    args = parser.parse_args()

    if args.wandb:
        project_name = args.project_name
        wandb.login(key='9b0830eae021991e53eaabb9bb697d9efef8fd58')
        wandb.init(project=project_name, entity="bub3690",tags=args.tag)
        wandb_run_name = args.model+'_512x512_'+args.name+'_classification'+'_segment_'+str(args.add_seg)+'_augment_'+args.augment+'_seed_'+str(args.seed)
        wandb.run.name = wandb_run_name
        wandb.run.save()
        wandb.run.summary.update({"seed" : args.seed,"model":args.model,"augment":args.augment,"descript":args.descript})

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
        X_images = glob('../../data/bmc_label_05-10/*.jpg') # test-set 폴더 입력
        # split이 아니라 바로 inference 
    else:
        pass
        #X_mask = glob('../../data/bmc_label_voc/SegmentationMaskedJPG/*.jpg') # 수정 필요.
        #X_train, X_valid = train_test_split(X_mask, test_size=0.3,random_state=args.seed)
    X_test_name = list(map(get_num,X_images))

    #첫번째 열 이름 바꿔주기
    label_df = pd.read_csv('../../bmc_05-10.csv')
    column_names = list(label_df.columns)
    column_names[0] = 'file_name'
    label_df.columns = column_names

    # Y값 찾아오기
    Y_test_df=pd.merge(pd.DataFrame(X_test_name,columns=['file_name']),label_df,left_on='file_name',right_on='file_name',how='inner')


    print("---")
    print("테스트 셋 : ",len(Y_test_df),Counter(Y_test_df['label']))
    print("---")
    
    test_loader = load_dataloader(X_images, Y_test_df, sublabel, BATCH_SIZE, args.model, args.augment, is_train = False, num_workers=args.num_workers)


    sublabel_count = len(set(label_df[sublabel]))


    # 학습 
    check_path = args.inference_chkpt
    print('실행 체크포인트 : ',check_path)

    # Confusion matrix (resnet18)
    # 모델을 각각 불러와서 test set을 평가한다.

    model = model_initialize(sublabel_count, DEVICE, model_name=args.model, check_point=args.pretrained_chkpt)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(check_path))

    if args.model == 'sub_1stage':
        #test 따로 미실시.
        return
    elif args.model == 'sub_2stage':
        predictions,prediction_res,prediction_col,prediction_tur,answers,answers_res,answers_col,answers_tur,test_loss = test_evaluate(model, test_loader,criterion,DEVICE,args.model)
    else:
        predictions,answers,test_loss = test_evaluate(model, test_loader,criterion,DEVICE,args.model)
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


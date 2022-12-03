import torch
from tqdm import tqdm


#8. 학습
def train(model,train_loader,optimizer,criterion,DEVICE,model_name='baseline'):
    model.train()
    correct = 0
    train_loss = 0

    data_length = len(train_loader.dataset)

    if model_name == 'baseline_multi':
        for batch_idx,(image,label,sublabel) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            sublabel = sublabel.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output = model(image,sublabel) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
    elif model_name == 'sub_1stage':
        correct_res = 0
        correct_col = 0
        correct_tur = 0
        for batch_idx,(image,sublabel) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            sublabel = sublabel.to(DEVICE)
            #데이터들 장비에 할당
            optimizer.zero_grad() # device 에 저장된 gradient 제거
            res,col,tur = model(image) # model로 output을 계산
            # output : res:012 ,col:012 ,tur:01
            
            #import pdb;pdb.set_trace()
            loss_res = criterion(res, sublabel[:,1])
            loss_col = criterion(col, sublabel[:,0])
            loss_tur = criterion(tur, sublabel[:,2])

            w_res,w_col,w_tur = 0.3,0.35,0.35
            loss = w_res*loss_res + w_col*loss_col + w_tur*loss_tur
            #loss = criterion(output, label) #loss 계산

            train_loss += loss.item()

            #예측 확률을 어떻게 계산할 것 인가?
            #각각의 예측 accuracy를 출력해서 보낼 것.

            prediction_res = res.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_res += prediction_res.eq(sublabel[:,1].view_as(prediction_res)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            prediction_col = col.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_col += prediction_col.eq(sublabel[:,0].view_as(prediction_col)).sum().item()# 아웃풋이 배치 사이즈 32개라서.

            prediction_tur = tur.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_tur += prediction_tur.eq(sublabel[:,2].view_as(prediction_tur)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                                
            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
        train_loss/=data_length
        train_residue_accuracy = 100. * correct_res / data_length
        train_color_accuracy = 100. * correct_col / data_length
        train_turbidity_accuracy = 100. * correct_tur / data_length
        return train_loss,train_residue_accuracy,train_color_accuracy,train_turbidity_accuracy
    elif model_name == 'sub_2stage':
        correct_res = 0
        correct_col = 0
        correct_tur = 0
        for batch_idx,(image,label,sublabel) in tqdm(enumerate(train_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            sublabel = sublabel.to(DEVICE)
            
            #데이터들 장비에 할당

            optimizer.zero_grad() # device 에 저장된 gradient 제거
            output,res,col,tur = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            train_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            # 결과 확인용
            prediction_res = res.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_res += prediction_res.eq(sublabel[:,1].view_as(prediction_res)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            prediction_col = col.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_col += prediction_col.eq(sublabel[:,0].view_as(prediction_col)).sum().item()# 아웃풋이 배치 사이즈 32개라서.

            prediction_tur = tur.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_tur += prediction_tur.eq(sublabel[:,2].view_as(prediction_tur)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                    

            loss.backward() # loss 값을 이용해 gradient를 계산
            optimizer.step() # Gradient 값을 이용해 파라미터 업데이트.
        train_loss/=data_length
        train_accuracy = 100. * correct / len(train_loader.dataset)        
        train_residue_accuracy = 100. * correct_res / data_length
        train_color_accuracy = 100. * correct_col / data_length
        train_turbidity_accuracy = 100. * correct_tur / data_length
        return train_loss, train_accuracy, train_residue_accuracy,train_color_accuracy,train_turbidity_accuracy
    else:
        #baseline 등
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
def evaluate(model,valid_loader,criterion,DEVICE,model_name='baseline'):
    model.eval()
    valid_loss = 0
    correct = 0
    data_length = len(valid_loader.dataset)
    #no_grad : 그래디언트 값 계산 막기.
    if model_name == 'baseline_multi':
        with torch.no_grad():
            for image,label,sublabel in valid_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                sublabel = sublabel.to(DEVICE)
                output = model(image,sublabel)
                valid_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
                #true.false값을 sum해줌. item
            valid_loss /= len(valid_loader.dataset)
            valid_accuracy = 100. * correct / len(valid_loader.dataset)
    elif model_name == 'sub_1stage':
        correct_res = 0
        correct_col = 0
        correct_tur = 0
        for batch_idx,(image,sublabel) in tqdm(enumerate(valid_loader)):
            image = image.to(DEVICE)
            sublabel = sublabel.to(DEVICE)
            #데이터들 장비에 할당
            res,col,tur = model(image) # model로 output을 계산
            # output : res:012 ,col:012 ,tur:01
            
            loss_res = criterion(res, sublabel[:,1])
            loss_col = criterion(col, sublabel[:,0])
            loss_tur = criterion(tur, sublabel[:,2])

            w_res,w_col,w_tur = 0.3,0.35,0.35
            loss = w_res*loss_res + w_col*loss_col + w_tur*loss_tur
            #loss = criterion(output, label) #loss 계산

            valid_loss += loss.item()

            #예측 확률을 어떻게 계산할 것 인가?
            #각각의 예측 accuracy를 출력해서 보낼 것.

            prediction_res = res.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_res += prediction_res.eq(sublabel[:,1].view_as(prediction_res)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            prediction_col = col.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_col += prediction_col.eq(sublabel[:,0].view_as(prediction_col)).sum().item()# 아웃풋이 배치 사이즈 32개라서.

            prediction_tur = tur.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_tur += prediction_tur.eq(sublabel[:,2].view_as(prediction_tur)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
        valid_loss/=data_length
        valid_residue_accuracy = 100. * correct_res / data_length
        valid_color_accuracy = 100. * correct_col / data_length
        valid_turbidity_accuracy = 100. * correct_tur / data_length
        return valid_loss,valid_residue_accuracy,valid_color_accuracy,valid_turbidity_accuracy
    elif model_name == "sub_2stage":
        correct_res = 0
        correct_col = 0
        correct_tur = 0
        for batch_idx,(image,label,sublabel) in tqdm(enumerate(valid_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            sublabel = sublabel.to(DEVICE)
            
            #데이터들 장비에 할당
            output,res,col,tur = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            valid_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct += prediction.eq(label.view_as(prediction)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            # 결과 확인용
            prediction_res = res.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_res += prediction_res.eq(sublabel[:,1].view_as(prediction_res)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
            
            prediction_col = col.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_col += prediction_col.eq(sublabel[:,0].view_as(prediction_col)).sum().item()# 아웃풋이 배치 사이즈 32개라서.

            prediction_tur = tur.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            correct_tur += prediction_tur.eq(sublabel[:,2].view_as(prediction_tur)).sum().item()# 아웃풋이 배치 사이즈 32개라서.
        valid_loss/=data_length
        valid_accuracy = 100. * correct / data_length    
        valid_residue_accuracy = 100. * correct_res / data_length
        valid_color_accuracy = 100. * correct_col / data_length
        valid_turbidity_accuracy = 100. * correct_tur / data_length
        return valid_loss,valid_accuracy,valid_residue_accuracy,valid_color_accuracy,valid_turbidity_accuracy
    else:
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
            #print("dataset 수 : ",len(valid_loader.dataset))
    return valid_loss,valid_accuracy

# # test


#confusion matrix 계산
#test set 계산.
def test_evaluate(model,test_loader,criterion,DEVICE,model_name='baseline'):
    model.eval()
    test_loss = 0
    predictions = []
    answers = []
    data_length = len(test_loader.dataset)
    #no_grad : 그래디언트 값 계산 막기.
    if model_name == 'baseline_multi':
        with torch.no_grad():
            for image,label,sublabel in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                sublabel = sublabel.to(DEVICE)
                output = model(image,sublabel)
                test_loss += criterion(output, label).item()
                prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
                answers +=label
                predictions +=prediction
                #true.false값을 sum해줌. item
            test_loss /= len(test_loader.dataset)
    elif model_name == "sub_2stage":
        answers_res = []
        answers_col = []
        answers_tur = []
        for batch_idx,(image,label,sublabel) in tqdm(enumerate(test_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            sublabel = sublabel.to(DEVICE)
            #데이터들 장비에 할당
            output,res,col,tur = model(image) # model로 output을 계산
            loss = criterion(output, label) #loss 계산
            test_loss += loss.item()
            prediction = output.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            
            answers +=label
            answers_res += sublabel[:,1]
            answers_col += sublabel[:,0]
            answers_tur += sublabel[:,2]


            
            # 결과 확인용
            prediction_res = res.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            prediction_col = col.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
            prediction_tur = tur.max(1,keepdim=True)[1] # 가장 확률이 높은 class 1개를 가져온다.그리고 인덱스만
        test_loss/=data_length
        return prediction,prediction_res,prediction_col,prediction_tur, answers,answers_res,answers_col,answers_tur,test_loss
    else:
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




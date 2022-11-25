
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms, datasets
from torchvision.ops import masks_to_boxes
import albumentations
from .Transforms import get_augementation


# # 데이터 정의
class BowelDataset(Dataset):
    def __init__(self, data_path_list,label_df,to_tensor,transform,sublabel,multilabel,augmentation=None):
        self.data_path_list = data_path_list
        self.label_df = label_df
        self.to_tensor = to_tensor
        self.transform = transform
        self.sublabel = sublabel #sublabel : color,residue,turbidity,label
        self.multilabel = multilabel # True or False


        self.torch_augmentation = augmentation['torch']
        self.album_augmentation = augmentation['album']

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_augmentation:
            image=self.album_augmentation(image=image)['image']
        
        image=self.to_tensor(image)
        
        if self.torch_augmentation:
            image = self.torch_augmentation(image)

        if self.transform:
            #1. 이미지 사이즈 변환
            image=self.transform(image).type(torch.float32)# 이미지 0~1 정규화


        if self.multilabel:
            return image, torch.tensor(self.label_df.iloc[idx][self.sublabel]), torch.tensor(self.label_df.iloc[idx][['color','residue','turbidity']]) 


        return image, torch.tensor(self.label_df.iloc[idx][self.sublabel])

def load_dataloader(X,Y_df,sublabel,BATCH_SIZE,multilabel,augmentation):
    augment_transform=get_augementation(augmentation) #dictionary가 넘어온다.

    loader = torch.utils.data.DataLoader(dataset = 
                                            BowelDataset(X,
                                                        Y_df,
                                                        to_tensor = transforms.ToTensor(),
                                                        transform = torch.nn.Sequential( # 기본 transform
                                                                transforms.Resize([512,512]),
                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                            ),
                                                        sublabel=sublabel, # color,residue,turbidity, label 중 어느것을 맞추려는지 입력.
                                                        multilabel=multilabel,
                                                        augmentation=augment_transform,
                                                        ),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            num_workers=0
                                            ) # 순서가 암기되는것을 막기위해.

    return loader

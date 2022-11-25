from torchvision import transforms
import albumentations as A

def get_augementation(augmentation):
    augment_dict = dict() # torch, album 두 키로 작동 한다.

    
    if augmentation == 'Base':
        augment_dict['album'] = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            ])
        augment_dict['torch'] = transforms.compose([
            transforms.RandomErasing(p=0.4),
        ])
    elif augmentation == 'Erase':
        augment_dict['torch'] = transforms.compose([
            transforms.RandomErasing(p=0.4),
        ])
        augment_dict['album'] = None

    elif augmentation == 'BrightnessContrast':
        augment_dict['album'] = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            ])
        augment_dict['torch'] = None
    else:
        # None
        augment_dict['album'] = None
        augment_dict['torch'] = None
    
    return augment_dict
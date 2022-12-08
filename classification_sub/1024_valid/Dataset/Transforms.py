from torchvision import transforms
import albumentations as A

def get_augementation(augmentation):
    augment_dict = dict() # torch, album 두 키로 작동 한다.

    
    if augmentation == 'Base':
        augment_dict['album'] = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            ])
        augment_dict['torch'] = transforms.Compose([
            transforms.RandomErasing(p=0.4),
        ])
    elif augmentation =='Flip':
        augment_dict['album'] = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            ])
        augment_dict['torch'] = None
    elif augmentation == 'Erase':
        augment_dict['torch'] = transforms.Compose([
            transforms.RandomErasing(p=0.4),
        ])
        augment_dict['album'] = None
    elif augmentation == 'RandomShadow':
        augment_dict['album'] = A.Compose([
                A.RandomShadow(shadow_roi=(0,0,1,1),shadow_dimension=6,p=0.3),
            ])
        augment_dict['torch'] = None        
    elif augmentation == 'BrightnessContrast':
        augment_dict['album'] = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.3),p=0.4),
            ])
        augment_dict['torch'] = None
    elif augmentation == "SunFlare":
        flare_roi=(0, 0, 1, 0.5) # 맺힐 영역
        angle_lower=0 #원의 정도
        angle_upper=1
        num_flare_circles_lower=1 #원의 수
        num_flare_circles_upper=2
        src_radius=300
        src_color=(255, 255, 255)
        augment_dict['album'] = A.Compose([A.RandomSunFlare(flare_roi, angle_lower, angle_upper,
                                                                num_flare_circles_lower,num_flare_circles_upper,
                                                                src_radius,
                                                                src_color,
                                                                p=0.5),
            ])
        augment_dict['torch'] = None


    else:
        # None
        augment_dict['album'] = None
        augment_dict['torch'] = None
    
    return augment_dict
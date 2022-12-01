from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil
from collections import defaultdict
from datetime import datetime


from skimage.metrics import structural_similarity as ssim 
import argparse
import cv2
from skimage import io
from itertools import combinations
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw

from torchvision.transforms.functional import to_pil_image


from tqdm import tqdm
VOCROOTDIR = "../data/2022_04"
img_list = []
for i in range(3):
    files = os.listdir(VOCROOTDIR + "/" + str(i) + "/" )
    img_files = list(filter(lambda x: '.jpg' in x, files))
    img_list += img_files


image_types = {"IMG":defaultdict(list), "PANO":defaultdict(list), "ETC":[]}


for file_name in img_list:
    _split = file_name.split("_")
    if 2 < len(_split):
        image_types[_split[0]][_split[1]].append(file_name)
    else:
        image_types["ETC"].append(file_name)

def get_concat_h(im1, im2,score):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    font = ImageFont.truetype("arial", 400)
    draw = ImageDraw.Draw(dst)
    # .text(
    #     (0, 0),  # Coordinates
    #     'Hello world!',  # Text
    #     (0, 0, 0)  # Color
    # )
    draw.text((10, 10),str(score),(0,0,255),font=font)

    return dst


count = 0
score_list = []
types = ["IMG","PANO"]
for idx in types:
    for k,v in tqdm(image_types[idx].items()):
        if(len(v) == 1):
            continue
        else:
            _check = list(combinations(v,2))
            for _img in _check:
                for i in range(3):
                    file = VOCROOTDIR + "/" + str(i) + "/" + _img[0]
                    if os.path.isfile(file):
                        imageA = cv2.imread(file)
                        break
                    else:
                        continue
                for i in range(3):
                    file = VOCROOTDIR + "/" + str(i) + "/" + _img[1]
                    if os.path.isfile(file):
                        imageB = cv2.imread(file)
                        break
                    else:
                        continue
                # print(_img)
                # print(imageA.shape, imageB.shape)
                if imageA.shape == imageB.shape:
                    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                    (score, diff) = ssim(grayA, grayB, full=True)
                elif imageA.shape == (imageB.shape[1],imageB.shape[0]):
                    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

                    imageCCW = cv2.rotate(imageA, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    grayCCW = cv2.cvtColor(imageCCW, cv2.COLOR_BGR2GRAY)
                    (scoreCCW, diffCCW) = ssim(grayCCW, grayB, full=True)

                    imageCW = cv2.rotate(imageA, cv2.ROTATE_90_CLOCKWISE)
                    grayCW = cv2.cvtColor(grayCW, cv2.COLOR_BGR2GRAY)
                    (scoreCW, diffCW) = ssim(grayCW, grayB, full=True)

                    if scoreCCW < scoreCW:
                        score = scoreCW
                        grayA = grayCW
                    else:
                        score = scoreCCW
                        grayA = grayCCW
                else:
                    continue
                score_list.append([_img, score])
                tmp = get_concat_h(to_pil_image(grayA),to_pil_image(grayB),score)
                # plt.imshow(tmp)
                # plt.show()
                plt.imshow(tmp); plt.axis('off'); plt.tight_layout(); #plt.show()
                plt.savefig('./duplicated/%s.png'%(str(int(score*1000))+ "_" + _img[0][:-4] + "_" + _img[1][:-4]))
                plt.close('all')
                plt.clf()

count = 0
score_list = []
types = ["ETC"]
for idx in types:
    _check = list(combinations(image_types[idx],2))
    for _img in tqdm(_check):
        for i in range(3):
            file = VOCROOTDIR + "/" + str(i) + "/" + _img[0]
            if os.path.isfile(file):
                imageA = cv2.imread(file)
                break
            else:
                continue
        for i in range(3):
            file = VOCROOTDIR + "/" + str(i) + "/" + _img[1]
            if os.path.isfile(file):
                imageB = cv2.imread(file)
                break
            else:
                if i == 2:
                    print(file)
                continue
        del file
        # print(_img)
        # print(imageA.shape, imageB.shape)
        if imageA.shape == imageB.shape:
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            (score, diff) = ssim(grayA, grayB, full=True)
        elif imageA.shape == (imageB.shape[1],imageB.shape[0]):
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

            imageCCW = cv2.rotate(imageA, cv2.ROTATE_90_COUNTERCLOCKWISE)
            grayCCW = cv2.cvtColor(imageCCW, cv2.COLOR_BGR2GRAY)
            (scoreCCW, diffCCW) = ssim(grayCCW, grayB, full=True)

            imageCW = cv2.rotate(imageA, cv2.ROTATE_90_CLOCKWISE)
            grayCW = cv2.cvtColor(grayCW, cv2.COLOR_BGR2GRAY)
            (scoreCW, diffCW) = ssim(grayCW, grayB, full=True)

            if scoreCCW < scoreCW:
                score = scoreCW
                grayA = grayCW
            else:
                score = scoreCCW
                grayA = grayCCW
        else:
            continue
        score_list.append([_img, score])
        tmp = get_concat_h(to_pil_image(grayA),to_pil_image(grayB),score)
        # plt.imshow(tmp)
        # plt.show()
        plt.imshow(tmp); plt.axis('off'); plt.tight_layout(); #plt.show()
        plt.savefig('./duplicated/%s.png'%(str(int(score*1000))+ "_" + _img[0][:-4] + "_" + _img[1][:-4]))
        plt.close('all')
        plt.clf()
        del tmp
    del _check

# Bowel_Movement_Cleaness
Classify bowel movement cleanness by photo


# Environment Settings
~~~
conda create -n bmc python=3.9 -y
conda activate bmc

pip install -r requirements.txt
cd mmsegmentation
pip install -r requirements.txt


# fiftyone (pycocotools error)
conda install -c conda-forge pycocotools
~~~


# Dataset
~~~
#labelme to coco

labelme2coco .\data\bmc
~~~

라벨 오류
- PANO_20200726_193219.jpg (0174)
- IMG_20201123_054809 (0209)
- IMG_20210304_174909 (0555)
- IMG_20210317_002710_1 (0569)
- IMG_20210317_002720 (0570)
- IMG_20210408_184102 (0603)
- IMG_20210530_221525 (0646)
- PANO_20200407_221523 (1077)
- PANO_20200407_221527 (1078)
- PANO_20210330_062718 (1080)

~~~
python ./labelme2voc.py C:\Users\mai\Bowel_Movement_Cleaness\data\bmc_label C:\Users\mai\Bowel_Movement_Cleaness\data\bmc_label_voc --labels labels.txt
~~~


import os
import cv2
import numpy as np
import warnings
import SiftUtils

warnings.filterwarnings(action='ignore')

data_path = './dataset/moderate/1035689/axial/'

files = [file for file in os.listdir(data_path) if file.endswith('.tif')]

imgs = []

for file in files:
    imgs.append(cv2.imread(data_path+file, cv2.IMREAD_GRAYSCALE))

imgs = np.array(imgs)

print(f'Shape of Imgs : {imgs.shape}')



for idx, img in enumerate(imgs[90:91]):
    SiftUtils.superm2(img, idx)    

# for idx, img in enumerate(imgs):
#     kp = sift.detect(img, None)
#     keyPointImg = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv2.imwrite(f'./sift/{idx}_kp_len({len(kp)}).png', keyPointImg)


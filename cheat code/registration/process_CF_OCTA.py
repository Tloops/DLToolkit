import cv2
import numpy as np
from tqdm import tqdm
import random
import time

data_bar = tqdm(range(112, 140))
for i in data_bar:
    filename = '%03dOCTA.png' % i
    data_bar.set_description("Processing " + filename)

    img_path = '/data/student/nieqiushi/DiffuseMorph/sample/testB/' + filename
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # height, width = img.shape
    # dst = np.zeros((height, width, 1), np.uint8)
    # for i in range(height):
    #   for j in range(width):
    #       dst[i][j] = 255-img[i][j]

    cv2.imwrite(img_path.replace('testB', 'testB256'), img)

print("Finished!")

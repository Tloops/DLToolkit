import cv2
import numpy as np
from tqdm import tqdm
import glob


def get_data_num():
    file_list = glob.glob('./FIRE/Images/*_1.jpg')
    for i in range(len(file_list)):
        file_list[i] = file_list[i][-9:-6]
    return sorted(file_list)


def process_mask(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    dst = (mask > 127).astype(np.uint8).repeat(3, axis=1).reshape((mask.shape[0], mask.shape[1], 3))
    return dst


# directory = './FIRE/'
# dataset_mask = process_mask(directory + 'mask.png')

# data_num = get_data_num()
# data_bar = tqdm(data_num)
# for filenum in data_bar:
#     data_bar.set_description("Processing " + filenum)

#     warped_img1 = cv2.imread(directory + 'GT_warped/warped_' + filenum + '_1.jpg')
#     warped_mask1 = process_mask(directory + 'GT_warped/warped_' + filenum + '_1_mask.jpg')
#     img2 = cv2.imread(directory + 'Images/' + filenum + '_2.jpg')

#     warped_img1_masked = warped_img1 * warped_mask1 * dataset_mask
#     img2_masked = img2 * warped_mask1 * dataset_mask

#     cv2.imwrite(directory + 'affine_registered/' + filenum + '_1.jpg', warped_img1_masked)
#     cv2.imwrite(directory + 'affine_registered/' + filenum + '_2.jpg', img2_masked)


file_list = glob.glob('FIRE/affine_registered/*.jpg')
data_bar = tqdm(file_list)
for filename in data_bar:
    img = cv2.imread(filename)
    resized_img = cv2.resize(img, (256, 256))
    cv2.imwrite(filename.replace('affine_registered', 'affine_registered256'), resized_img)

print("Finished!")
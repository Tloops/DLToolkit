import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def field_visualizer(field_numpy, imtype=np.float32):
    '''
    field_numpy: (h, w, 2)
    put field_numpy into a 3-channel image using the red and green channels
    the blue channel is set to 0
    '''
    nh, nw, _ = field_numpy.shape
    tmp = np.zeros((nh, nw, 3))
    tmp[:, :, :2] = field_numpy
    field_np = tmp
    field_np -= np.amin(field_np)
    field_np /= np.amax(field_np)
    field_np = field_np * 255
    return field_np.astype(imtype)


def random_elastic_transform(image, alpha, sigma, random_state=None):
    '''
    image: (h, w, ?), can be gray or color image
    alpha: scale of transformation, the bigger the more distortion, the best choice is 2*w or 2*h
    sigma: smoothness of transformation, the bigger the smoother, the best choice is 0.08*w or 0.08*h
    random_state: random seed
    '''
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w, _ = image.shape
    field_shape = (h, w, 1)
    
    dx = gaussian_filter((random_state.rand(*field_shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*field_shape) * 2 - 1), sigma) * alpha
    field = np.concatenate((dx, dy), axis=-1)

    x, y, z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
    print(x.shape, y.shape, z.shape, dx.shape, dy.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    im_t = map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
    return field, im_t


if __name__ == "__main__":
    img_path = 'A01_1.jpg'
    im = cv2.imread(img_path)
    print('im.shape', im.shape)
    field, im_t = random_elastic_transform(im, im.shape[1]*2, im.shape[1]*0.08)
    print('field.shape', field.shape)
    field_viz = field_visualizer(field)
    cv2.imwrite('field.png', field_viz)
    cv2.imwrite('A01_1_t.jpg', im_t)

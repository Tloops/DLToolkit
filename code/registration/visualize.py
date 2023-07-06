import numpy as np
from PIL import Image


# generate checkboard of two images
# n: nxn checkboard
# dim: the dim of input image
def checkboard(I1, I2, n, dim=2):
    assert I1.shape == I2.shape
    if dim == 2:
        height, width = I1.shape
    else:
        height, width, channel = I1.shape

    hi, wi = height/n, width/n
    if dim == 2:
        outshape = (int(hi*n), int(wi*n))
    else:
        outshape = (int(hi*n), int(wi*n), channel)

    out_image = np.zeros(outshape, dtype='uint8')
    for i in range(n):
        h = int(round(hi * i))
        h1 = int(round(h + hi))
        for j in range(n):
            w = int(round(wi * j))
            w1 = int(round(w + wi))

            if (i-j)%2 == 0:
                if dim == 2:
                    out_image[h:h1, w:w1] = I1[h:h1, w:w1]
                else:
                    out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                if dim == 2:
                    out_image[h:h1, w:w1] = I2[h:h1, w:w1]
                else:
                    out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]
    return out_image


# directly merge by transparency
def merge(I1, I2):
    img1 = I1.convert('RGBA')
    img2 = I2.convert('RGBA')
    img = Image.blend(img1, img2, 0.5)
    return img


if __name__ == "__main__":
    I1 = Image.open("img1.jpg")  # .convert("L")
    I2 = Image.open("img2.jpg")  # .convert("L")
    
    out = checkboard(np.array(I1), np.array(I2), n=4, dim=2)
    img = Image.fromarray(out)
    img.save("checkboard.jpg")
            
    img = merge(I1, I2)
    img.save("/merge.png")

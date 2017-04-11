from PIL import Image
import numpy as np
import sys
import os

def getPSNR(noisy, clean, maxVal = 255.):
    Diff = noisy.astype(float) - clean.astype(float)
    RMSE = np.sqrt(np.mean(Diff ** 2))
    PSNR = 20 * np.log10(maxVal / RMSE)

    return PSNR

def get_psnr(noise, clean, dynamic=255.0):
    """
    Calculate the PSNR value between noise and clean image.

    :param noise: numpy array (no < 0 or > dynamic values are allowed)
    :param clean: numpy array without noise
    :param dynamic: the scale (max value) of the image
    :return: PSNR value
    """
    peak = dynamic * dynamic
    mse = np.square(noise.astype(float) - clean.astype(float)).mean()
    return 10 * np.log10(peak / mse)

def image2patches(im, pSz, stride):
    """
    Function:
        Convert images to patches, (instead of columns)
    Parameters:
        im: input image, shape: [channels, h, w]
        pSz: patch Size
        stride: stride
    """
    assert im.ndim == 3, "Image's shape must be [channels, height, weight]"
    c, h, w = im.shape

    range_y = range(0, h + 1 - pSz, stride)
    range_x = range(0, w + 1 - pSz, stride)
    if range_y[-1] != h - pSz:
        range_y.append(h - pSz)
    if range_x[-1] != w - pSz:
        range_x.append(w - pSz)

    res = [im[:, y : y + pSz, x : x + pSz].reshape(1, c, pSz, pSz) for x in range_x for y in range_y]

    # the final result is a 4-dims array: [num_patches, channels, pSz, pSz]
    return np.concatenate(res, 0)

def patches2image(patches, im_sz, stride):
    """
    Function:
        Convert patches to image
    Parameters:
        patches: patch array, 4-dims array: [num_patches, channels, pSz, pSz]
        im_sz: image size, (im_H, im_W)
        stride: stride
    """
    pSz  = patches.shape[2]
    im_c = patches.shape[1]

    im_H, im_W = im_sz

    range_y = range(0, im_H - pSz + 1, stride)
    range_x = range(0, im_W - pSz + 1, stride)
    if range_y[-1] != (im_H - pSz):
        range_y.append(im_H - pSz)
    if range_x[-1] != (im_W - pSz):
        range_x.append(im_W - pSz)

    final_img = np.zeros((im_c, im_H, im_W))
    w = np.zeros((im_c, im_H, im_W))

    idx = 0
    # be careful: in image2patches, "for x in range_x" is firstly executed.
    for x in range_x:
        for y in range_y:
            p = patches[idx, :, :, :]
            final_img[:, y : y + pSz, x : x + pSz] += p
            w[:, y : y + pSz, x : x + pSz] += 1
            idx += 1
    final_img /= w

    return final_img

def save_image(array, path):
    """Turn a numpy array to RGB image"""
    
    array[array < 0] = 0
    array[array > 255] = 255
    image = Image.fromarray(array)
    image.convert('RGB').save(path) 
    return 0

def processImage(array):
    """remove outliers, To get correct PSNR"""
    array[array < 0] = 0
    array[array > 255] = 255

    return array


def data_augmentation(image, mode):
    # careful: rot90 and flipud deal with first two dimensions
    image = image.transpose(2, 3, 0, 1)
    if mode == 1:
        pass 
    elif mode == 2: # flipped
        image = np.flipud(image)
    elif mode == 3: # rotation 90
        image = np.rot90(image, k = 1)
    elif mode == 4: # rotation 90 & flipped
        image = np.flipud(np.rot90(image, k = 1))
    elif mode == 5: # rotation 180
        image = np.rot90(image, k = 2)
    elif mode == 6: # rotation 180 & flipped
        image = np.flipud(np.rot90(image, k = 2))
    elif mode == 7: # rotation 270
        image = np.rot90(image, 3)
    elif mode == 8: # rotation 270 & flipped
        image = np.flipped(np.rot90(image, k = 3))

    return image.transpose(2, 3, 0, 1)

def load_data(filepath='./data/img_clean_pats.npy'):
    assert '.npy' in filepath
    if not os.path.exists(filepath):
        print "[!] Data file not exists"
        sys.exit(1)

    print "[*] Loading data..." 
    data = np.load(filepath)
    np.random.shuffle(data)
    print "[*] Load successfully..." 
    return data.transpose(3, 2, 0, 1)

if __name__ == '__main__':
    # a = np.arange(64).reshape(1, 8, 8)
    # print a
    # patches = image2patches(a, 5, 2)
    # print patches2image(patches, (8, 8), 2)

    a = np.arange(27).reshape(3, 3, 3)
    print a
    patches = image2patches(a, 2, 1)
    print patches2image(patches, (3, 3), 1)
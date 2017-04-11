from PIL import Image
import numpy as np

path = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/data/train/01.png'
im = Image.open(path).convert('L')
print np.array(im)
width, height = im.size
# scales = [1, 0.9]
# for i in range(len(scales)):
#     scale_width = int(np.floor(width * scales[i]))
#     scale_height = int(np.floor(height * scales[i]))
#     scale_size = (scale_width, scale_height)
#     scale_im = im.resize(scale_size, Image.BICUBIC)


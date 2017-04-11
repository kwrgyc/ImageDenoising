import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
from model import DenoiseNet
from ImageProcess import get_psnr, save_image, processImage

seed = 0
sigma = 25
ngpu = 1
batchSize = 1
nc = 1
imgSize = 512
device = 1
loadNet = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/checkpoint/net_epoch_0.pth'
imageResult = './ImageResults'

class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, root, test_set):
        super(DataSet, self).__init__()
        self.root = root
        self.test_set = test_set
        self.test_path = os.path.join(self.root, self.test_set)

    def __iter__(self):
        for filename in os.listdir(self.test_path):
            image_path = os.path.join(self.test_path, filename)
            im = Image.open(image_path).convert('L')
            im = np.array(im)
            im = im.reshape(1, *im.shape)  # add a channel

            yield im.reshape(1, *im.shape)


dset = DataSet(root = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/data', test_set = 'Set12')
# test_images = dset.ToImageArray()
net = DenoiseNet(ngpu)
if loadNet != '':
    net.load_state_dict(torch.load(loadNet))
print net

input = torch.Tensor(batchSize, nc, imgSize, imgSize)

if torch.cuda.is_available():
    net.cuda(device)
    input = input.cuda(device)

input = Variable(input)

def Transform(data):
    """
    Covert numpy arrays to Tensors and Normalization.
    """
    data = torch.from_numpy(data).float().div_(255.)
    return data

def ReverseTrans(data):
    """
    reverse transform
    """
    data = 255. * data
    return data

for index, label in enumerate(dset):
    np.random.seed(seed)
    noise_input = label + sigma * np.random.randn(*label.shape)
    noise_psnr = get_psnr(noise_input, label)
    noise_tensor_input = Transform(noise_input)
    input.data.resize_(noise_tensor_input.size()).copy_(noise_tensor_input)

    res = net(input)
    output = input - res
    output = output.data.cpu().numpy()
    output = ReverseTrans(output)
    output = processImage(output)
    psnr = get_psnr(output, label)


    label = label.reshape(*label.shape[2:])
    noise_input = noise_input.reshape(*noise_input.shape[2:])
    output = output.reshape(*output.shape[2:])

    save_image(label, '%s/clean_index_%d.png' % (imageResult, index))
    save_image(noise_input, '%s/noise_index_%d.png' % (imageResult, index))
    save_image(output, '%s/output_index_%d.png' % (imageResult, index))
    print "Noise_PSNR: %.4f, PSNR: %.4f" % (noise_psnr, psnr)
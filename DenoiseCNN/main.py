import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.parallel
import torchvision.utils as vutils
from model import DenoiseNet
from dataset import DataSet, DataLoader
from ImageProcess import getPSNR, get_psnr, save_image, processImage, image2patches, patches2image, load_data
from PIL import Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import time
import threading
import itchat

lock = threading.Lock()
running = False

imageResult = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/ImageResults'
checkpoint = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/checkpoint'
batchSize = 128
mean = 0.
std = 1.
nc = 1
scales = [1, 0.9, 0.8, 0.7]
pSz = 40
patchStride = 10
seed = 1234
sigma = 25
learningRate = 0.001
beta = 0.9
momentum = 0.2
niter = 50
every_checkpoint = 1
chat_every = 2000
loadNet = ''
ngpu = 1
device = 1

try:
    os.makedirs(imageResult)
    os.makedirs(checkpoint)
except OSError:
    pass

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        value = torch.randn(64, 1, 3, 3) * np.sqrt(2./ (9 * 64))
        m.weight.data.copy_(value)
        m.bias.data.fill_(0)

b_min = 0.025

def clipping(A, b):
    A[(A >= 0) & (A < b)] = b
    A[(A < 0) & (A > -b)] = -b
    return A

def conv_relu_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        value = torch.randn(64, 64, 3, 3) * np.sqrt(2./ (9 * 64))
        m.weight.data.copy_(value)
        # m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        value = np.random.randn(64) * np.sqrt(2./ (9 * 64))
        value = torch.Tensor(clipping(value, b_min))
        m.weight.data.copy_(value)
        m.bias.data.fill_(0)

dset = DataSet(root = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/data', 
              train = True,
              stride = 1,
              scales = scales)
dataloader = DataLoader(dataset = dset, 
                        batchSize = batchSize,
                        pSz = pSz,
                        patchStride = patchStride,
                        sigma = sigma,
                        shuffle = True,
                        seed = seed
                        )

def Transform(data, target, mean = 0., std = 1.):
    """
    Covert numpy arrays to Tensors and Normalization.
    """
    data = torch.from_numpy(data).float()#.div_(255.)
    target = torch.from_numpy(target).float()#.div_(255.)
    data.sub_(mean).div_(std)
    # target.sub_(mean).div_(std)
    return data, target

def ReverseTrans(data, mean = 0., std = 1.):
    """
    reverse transform
    """
    data = torch.from_numpy(data).float()
    data.mul_(std).add_(mean).mul_(255.)
    return data



net = DenoiseNet(ngpu)
net.start_conv.apply(weights_init)
net.conv_relu_layers.apply(conv_relu_weights_init)
net.end_conv.apply(weights_init)

if loadNet != '':
    net.load_state_dict(torch.load(loadNet))
print net

def returnConv(net):
    """
    Function:
        Return the parameters of nn.Conv2d layer in net which can be used in Optim algorithm
        See main.py optim.Adam
    Parameters:
        net: nn.Module
    Return:
        parameters generator
    """
    for module in net.children():
        classname = module.__class__.__name__
        if classname.find('Conv2d') != -1:
            yield next(module.parameters())

def returnBatchNorm(net):
    for module in net.children():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            yield next(module.parameters())

conv_parameters = returnConv(net.conv_relu_layers)
batchnorm_parameters = returnBatchNorm(net.conv_relu_layers)

optimizer = optim.Adam([
        {'params': net.start_conv.parameters()},
        {'params': conv_parameters},
        {'params': batchnorm_parameters},
        {'params': net.end_conv.parameters()}
    ],
    lr = learningRate,
    betas = (beta, 0.999)
    )
    
# optimizer = optim.Adam(net.parameters(), lr = learningRate, betas = (beta, 0.999))
# optimizer = optim.SGD(net.parameters(), lr = learningRate, momentum = momentum)

criterionMSE = nn.MSELoss()

input = torch.Tensor(batchSize, nc, pSz, pSz)
target = torch.Tensor(batchSize, nc, pSz, pSz)

if torch.cuda.is_available():
    net.cuda(device)
    criterionMSE.cuda(device)
    input = input.cuda(device)
    target = target.cuda(device)

input = Variable(input)
target = Variable(target)

def add_noise(data, sigma):
    noise = sigma / 255. * np.random.randn(*data.shape)
    return (data + noise), noise

def nn_train(chat_name, param):
    global lock, running
    with lock:
        running = True

    print "Waiting For Lock"
    with lock:
        run_state = running

    data = load_data()
    numBatch = int(data.shape[0] / batchSize)
    print("[*] Data shape = " + str(data.shape))

    counter = 0
    print("[*] Start training : ")
        

    for epoch in range(niter):
        if run_state:
            # for index, dataInfo in enumerate(dataloader):
            #     batch_noise, residual_labels = dataInfo

            for index in xrange(numBatch):
                batch_images = data[index * batchSize : (index + 1) * batchSize, :, :, :]
                batch_noise, residual_labels = add_noise(batch_images, sigma)

                batch_noise, residual_labels = Transform(batch_noise, residual_labels, mean = mean, std = std)
                input.data.resize_(batch_noise.size()).copy_(batch_noise)
                target.data.resize_(residual_labels.size()).copy_(residual_labels)

                t0 = time.time()
                net.zero_grad()
                output = net(input)
                loss = criterionMSE(output, target)
                loss.backward()
                optimizer.step()

                t1 = time.time() - t0

                # print '[%d/%d][%d/%d] loss: %.5f, time: %.5f' \
                #     % (epoch, niter, index, len(dataloader), loss.data[0], t1)
                print '[%d/%d][%d/%d] loss: %.5f, time: %.5f' \
                    % (epoch, niter, index, numBatch, loss.data[0], t1)

#                if index % chat_every == 0:
#                    itchat.send('[%d/%d][%d/%d] loss: %.5f, time: %.5f' \
#                    % (epoch, niter, index, len(dataloader), loss.data[0], t1), chat_name)

                with lock:
                    run_state = running
                if not run_state:
                    while True:
                        with lock:
                            run_state = running
                        if run_state:
                            break
            
            if (epoch + 1) % every_checkpoint == 0:    
                torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (checkpoint, epoch))
    with lock:
        running = False

@itchat.msg_register([itchat.content.TEXT])
def chat_trigger(msg):
    global lock, running, learningRate, batchSize, print_every
    if msg['Text'] == 'start':
        print('Starting!')
        with lock:
            run_state = running
        if not run_state:
            threading.Thread(target = nn_train, args = ('filehelper', (learningRate, batchSize, chat_every))).start()
    elif msg['Text'] == 'stop':
        print('Stopping')
        with lock:
            running = False
    elif msg['Text'] == 'continue':
        print('Continue')
        with lock:
            running = True

if __name__ == '__main__':
#    itchat.auto_login(hotReload = True)
#    itchat.run()
    nn_train('test', 'test')

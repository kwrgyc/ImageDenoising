import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel 
import numpy as np

# Depth = 45
Depth = 45
class DenoiseNet(nn.Module):
    """docstring for DenoiseNet"""
    def __init__(self, ngpu):
        super(DenoiseNet, self).__init__()
        self.ngpu = ngpu
        self.start_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_relu_layers = nn.ModuleList(
            [nn.Conv2d(64, 64, 3, 1, 1, bias = False) if i % 3 == 0 else nn.BatchNorm2d(64) if i % 3 == 1 else nn.ReLU() for i in range(Depth)]
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, input):
        # self.gpu_ids = None
        # if torch.cuda.is_available() and self.ngpu > 1:
        #     self.gpu_ids = range(self.ngpu)
        x = self.start_conv(input)
        for module in self.conv_relu_layers:
            x = module(x)
        x = self.end_conv(x)
        return x

if __name__ == '__main__':
    input = Variable(torch.randn(3, 1, 32, 32))
    net = DenoiseNet(1)
    # print dir(net.conv_relu_layers)
    # output = net(input)
    # print output.size()
    print "first: with bias"
    for p in net.start_conv.parameters():
        print p.size()
    print "second: without bias"

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
        elif classname.find('BatchNorm') != -1:
            value = np.random.randn(64) * np.sqrt(2./ (9 * 64))
            value = torch.Tensor(clipping(value, b_min))
            m.weight.data.copy_(value)
            m.bias.data.fill_(0)

    net.conv_relu_layers.apply(conv_relu_weights_init)
    print "Success!"


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

    m = returnConv(net.conv_relu_layers)
    for p in m:
        print p.size()
    # for p in net.conv_relu_layers.parameters():
        # print p.__class__
        # break
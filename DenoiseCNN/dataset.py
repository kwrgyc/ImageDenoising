from PIL import Image
import os
import numpy as np 
import random
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from tqdm import tqdm
import math
from ImageProcess import patches2image, image2patches, data_augmentation, load_data

class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, root, train, stride, scales = [1], train_set = 'Train400', test_set = 'Set12'):
        """
        Parameters:
            root: dataset root directory
            train: Bool, training or test set
            scales: scale every image, tuple or list
            stride: every stride crop an image from the original image
            train_set: folder name of training set under root directory
            test_set: folder name of test set under root dirctory
        """
        super(DataSet, self).__init__()
        self.root = root
        self.train = train
        self.scales = scales
        self.stride = stride
        self.train_set = train_set
        self.test_set = test_set
        self.train_path = os.path.join(self.root, self.train_set)
        self.test_path = os.path.join(self.root, self.test_set)


    def ToImageArray(self):
        """
        Convert images to numpy arrays
        """
        if self.train:
            dataDir = self.train_path
        else:
            dataDir = self.test_path

        self.images = []
        for filename in tqdm(os.listdir(dataDir)):
            image = os.path.join(dataDir, filename)
            im = Image.open(image).convert(mode = 'L')
            height, width = im.size # this is wrong, (width, height) is true, but the following codes are all using (h, w)
            
            for scale in self.scales:
                newsize = int(height * scale), int(width * scale)
                newim = im.resize(newsize, resample = Image.BICUBIC)
                newim = np.array(newim)
                newim = newim.reshape(1, *newim.shape)  # add a channel
                self.images.append(newim)
        return self.images

class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, dataset, batchSize, patchStride, pSz, sigma, shuffle, seed):
        """
        Parameters:
            dataset: dataset, constructed from the DataSet Class
            images: images list, returned from class DataSet
            batchSize: batchSize of image cols
            pSz: patch Size
            patchStride: used in image2cols, who has a parameter needs stride
            sigma: standard deviation
            shuffle: Bool, if shuffle, then randomly choose an image
            seed: random seed
        """
        super(DataLoader, self).__init__()
        self.dataset = dataset
        self.batchSize = batchSize
        self.pSz = pSz
        self.patchStride = patchStride
        self.seed = seed
        self.sigma = sigma
        self.shuffle = shuffle

        self.images = self.dataset.ToImageArray()

        """
        Convert images to patches and add noise
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.clean_inputs = []
        # for clean_image in self.images:
        #     clean_inputs = image2patches(clean_image, self.pSz, self.patchStride) # clean_inputs: [num_patches, channels, pSz, pSz]
        #     self.clean_inputs.append(clean_inputs) 
        # self.clean_inputs = np.concatenate(self.clean_inputs, 0)
        self.clean_inputs = load_data() # careful!!! the output of load_data is normalized between [0, 1]

        self.samples_remaining = len(self.clean_inputs)
        if not self.shuffle:
            self.sampler = SequentialSampler(self.clean_inputs)
        else:
            self.sampler = RandomSampler(self.clean_inputs)
        self.sample_iter = iter(self.sampler)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return int(math.ceil(len(self.sampler) / float(self.batchSize)))

class DataLoaderIter(object):
    """docstring for DataLoaderIter"""
    def __init__(self, loader):
        super(DataLoaderIter, self).__init__()
        self.dataset = loader.dataset
        self.batchSize = loader.batchSize
        self.pSz = loader.pSz
        self.patchStride = loader.patchStride
        self.sigma = loader.sigma
        self.shuffle = loader.shuffle

        self.images = loader.images
        self.clean_inputs = loader.clean_inputs
        self.sampler = loader.sampler
        self.sample_iter = iter(self.sampler)
        self.samples_remaining = len(self.sampler)
        

    def __next__(self):
        """
        Return patch cols pair (batch_noise, batch_clean, noise_image, clean_image)
        """
        if self.samples_remaining == 0:
            raise StopIteration
        index = self._next_indices()
        # clean_patches' shape is [num_patches, channels, pSz, pSz]
        clean_patches = self.clean_inputs[index]

        # Data Augmentation
        mode = np.random.permutation(8)
        clean_patches = data_augmentation(clean_patches, mode[0])

        labels = self.sigma / 255. * np.random.randn(*clean_patches.shape)
        noise_patches = clean_patches + labels
        return noise_patches, labels

    next = __next__ # python 2 compatibility

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(len(self.sampler) / float(self.batchSize)))

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batchSize)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch
        
        
class SequentialSampler(object):
    """Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class RandomSampler(object):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(np.random.permutation(self.num_samples))

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    dataset = DataSet(root = '/home/ieric/Codes/ImageDenoising/DenoiseCNN/data', 
                      train = True,
                      scales = [1, 0.9, 0.8, 0.7],
                      stride = 1,
                      train_set = 'train')

    dataloader = DataLoader(dataset = dataset, 
                            batchSize = 128,
                            pSz = 40,
                            patchStride = 10,
                            sigma = 25,
                            shuffle = False,
                            seed = 1234
                            )

    print len(dataloader)
    print next(iter(dataloader))[0].shape
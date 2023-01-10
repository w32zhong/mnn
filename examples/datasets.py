import torch
from torchvision import datasets
from torch.utils.data import Dataset

import sys
sys.path.insert(0, '.')
from mnn.tensor import Tensor

import pickle


class MNIST(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'rb') as fh:
            self.data, self.labels = pickle.load(fh)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def prepare_MNIST_dataset():
    for train in [True, False]:
        print('download/loading dataset ...')
        mnist = datasets.MNIST('./data/MNIST', download=True, train=train)
        mnist_rawdata = [list(img.getdata()) for img, label in mnist]
        mnist_labels = [label.item() for img, label in mnist]
        pixel_colors = 255
        mnist_data = Tensor(mnist_rawdata)
        mnist_data = mnist_data / pixel_colors

        print('normalizing dataset ...')
        if train:
            mean = mnist_data.mean()
            std = mnist_data.std()
            print('dataset mean, std:', mean.item(), std.item())

        mnist_data = (mnist_data - mean) / std
        mnist_data = mnist_data.tolist()

        print('saving dataset ...')
        if train:
            save_filename = './data/MNIST/mnn_train.pickle'
        else:
            save_filename = './data/MNIST/mnn_test.pickle'
        with open(save_filename, 'wb') as fh:
            pickle.dump((mnist_data, mnist_labels), fh)


def peek_dataset(cls_name, dataset_path, batch_size=1):
    dataset = globals()[cls_name](dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, collate_fn=lambda batch: batch)
    sample = next(iter(loader))
    print(sample)


if __name__ == '__main__':
    import fire, os
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'prepare_MNIST_dataset': prepare_MNIST_dataset,
        'peek': peek_dataset
    })

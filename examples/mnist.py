import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from mnn.tensor import Tensor
from mnn.seq_layers import *

import torch
from datasets import MNIST

def train(epochs=10, dryrun=False, debug=False):

    dataset = MNIST('./data/MNIST/mnn_test.pickle')
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=64, shuffle=True, collate_fn=lambda batch: batch)

    net = SequentialLayers([
        LinearLayer(28 * 28, 256),
        ReluLayer(),
        LinearLayer(256, 10),
        CrossEntropyLossLayer()
    ])

    for ep in range(epochs):
        for b, batch in enumerate(loader):
            images = Tensor([data for data, label in batch])
            images = images.unsqueeze(-1)
            labels = Tensor([label for data, label in batch])
            labels = labels.unsqueeze(-1)
            loss = net(images, labels, debug=debug)
            print(f'Epoch#{ep} batch#{b} loss:', loss.item())

            if dryrun:
                plt.imshow(images[0].reshape(28, 28).get())
                plt.show()
                return


if __name__ == '__main__':
    import fire, os
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'train': train
    })

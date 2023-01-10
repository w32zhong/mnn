import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from mnn.tensor import Tensor
from mnn.seq_layers import *

import torch
from datasets import MNIST

import pickle


def train(epochs=10, dryrun=False, debug=False, batch_size=64,
    save_file='data/mnist_model_ckpt.pkl'):

    dataset = MNIST('./data/MNIST/mnn_test.pickle')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, collate_fn=lambda batch: batch)

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

            data_shape = images.shape
            print(f'Epoch#{ep} batch#{b} {data_shape} loss:', loss.item())

            net.zero_grads()
            gradients = net.backward(debug=debug)
            net.step()

            if dryrun:
                plt.imshow(images[0].reshape(28, 28).get())
                plt.show()
                return

    print('saving checkpoint ...')
    with open(save_file, 'wb') as fh:
        save = net.state_dict(), net.config()
        pickle.dump(save, fh)


def test(dryrun=False, debug=False):
    pass


if __name__ == '__main__':
    import fire, os
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'train': train,
        'test': test
    })

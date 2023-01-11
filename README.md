![](https://static.simpsonswiki.com/images/thumb/1/19/Noir_Homer.png/339px-Noir_Homer.png)
# Manual Neural Networks (MNN)
MNN is a bare *minimum* but sufficiently *efficient* neural network implementation running only on GPU.
It serves as an *educational* reference to whoever wants to inspect how neural network works as well as the math behind the scenes.

* *Minimum*: no complicated or bloated frameworks, manual operator differentiation, only a few source files written from scratch in a crystal clear way.
* *Efficient*: although taking a simplistic approach, it expects no worse efficiency thanks to: no low-granularity automatic differentiation (i.e., *auto diff*) thus no large graph of the derivatives, and running (only) on GPU where CuPy underpins all the efficient matrix manipulation.
* *Educational*: focusing on neural network core knowledge, written in simple Python but with detailed in-source-code LaTeX comments that describe the math! (e.g., including derivations of the Jacobian matrix)

## Current state
This project is still under development.
Although it has a complete pipeline for solving a MNIST classification task, many exciting hands-on and well-documented code is yet to come.
Wishfully, expect a Transformer module which can be trained in one or two days on a consumer GPU in the future!

## Quick start
```sh
python examples/datasets.py prepare_MNIST_dataset
python examples/mnist.py train \
    --save_file=./data/mnist_model_ckpt.pkl \
    --batch_size=1024 \
    --epochs 60
python examples/mnist.py test ./data/mnist_model_ckpt.pkl 
```

## Learning steps
Here is a recommended ordered reading list of MNN source code that you can pick up knowledge smoothly by going through the comments with linked code location:

1. [examples/mnist](examples/mnist.py) (only illustrating the pipeline code)
1. [SequentialLayers](docs/mnn.seq_layers.md)
1. [LinearLayer](docs/mnn.layer.LinearLayer.md)
1. [ReluLayer](docs/mnn.layer.ReluLayer.md)
1. [MSELossLayer](docs/mnn.layer.MSELossLayer.md)
1. [SoftmaxLayer](docs/mnn.layer.SoftmaxLayer.md)
1. [LogSoftmaxLayer](docs/mnn.layer.LogSoftmaxLayer.md)
1. [NllLossLayer](docs/mnn.layer.NllLossLayer.md)
1. [CrossEntropyLossLayer](docs/mnn.layer.CrossEntropyLossLayer.md)

## Credits
This project is inspired by [pytorch](https://github.com/pytorch/pytorch) and [tinynn](https://github.com/borgwang/tinynn).
Further inspiration may also taken from other projects (will be listed).

## License
MIT

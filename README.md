# fashion-mnist-classifier

These are the results of a fun Kaggle competition I competed in for my machine learning class. The assignment was to correctly classify images from the [Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset, which is comprised of various articles of clothing.

<img src=images/mnist-image.png width="500">

## Methodology

We used [convolution neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) to build the classifier. While there are an abundance of ways to implement a CNN, I wanted to keep my model simple and go with some of the tried-and-true architectures we read about in class. The Stanford Computer Vision staff recommends a ["CONV-RELU" layer stacking architecture](http://cs231n.github.io/convolutional-networks/#layerpat), with multiple POOL layers. I decided to implement something similar, and as they stress in the literature, not reinvent the wheel.

## Training the model

The training dataset contained 60,000 samples with 10 different classes. I used [PyTorch](https://pytorch.org/docs/stable/index.html) to implement my architecture and then focused on fine-tuning the "learning rate" and "momentum" parameters. After some empirical testing, I found their optimal values by observing gradual minimization in the loss function over 16 epochs. See my full script in [fashionMNIST.ipynb](./fashionMNIST.ipynb).

<img src=images/loss-plot.jpg width="500">

## Results

Over the 10,000 test samples, I achieved 88% accuracy with my model. This score was high enough to put me in the top 10 of the class!

I only let my model run for 20 epochs, but if I had more time (or more computing power e.g. access to a CUDA device), I think I could have improved my model even further.

# mnist-adversary
A live repository following research in adversarial image generation to fool convolutional (and other) neural networks. Currently testing on MNIST. 

## Thread Models
There are a few different ways to generate adversarial examples for image recognition networks (and neural networks by extension). These methods differ in that they have different levels of knowledge about the model architecture and its parameters, and use differnt optimization techniques to generate perturbations. Perturbations are then added to input sample features with the goal of causing mis-classification.

Much progress has been made in generating and circumventing white-box attacks (through techniques that try to obfuscate, or hide network gradients). However, the problem of circumventing adversarial attacks is similar to an arms race. Shortly after the demonstration of new defenses, new adversaries are designed in response. This makes this area of work highly compelling (there is lots to be done).

For the purpose of this project, we will start by performing a simple and standard white-box attack commonly seen in literature. It is our goal to get a hands-on introduction to the task of adversarial attack on neural networks. We will do this by using the MNIST dataset, and generate perturbations that allow us (the adversarial engineers) to choose how a standard convolutional network classifies images. Specifically, we will be trying to force a standard convolutional network to classify images of 2s as 6s. 

Time permitting, we will try two further things:

1) White/Black-box attacks where we are allowed to modify only a handful of pixels. 
2) Black-box attacks (in which the adversary does not have access to the network)

### Iterative Fast Gradient Sign Method 
We start with a white-box method which requires access to network architecture for gradient calculations. This method was first theorized by Goodfellow et al. in 2015 as a simple test of the linearity of a neural network. Starting from the original image that one wishes to generate adversarial examples for, one can iteratively descend in the direction of the gradient of the loss function calculated with respect to that image and the adversarial target class by subtracting a scaled multiple of sign of the gradient from the image. Accumulated over time steps, this cumulative subtracted image constitued the additive perturbation (notice how this process is essentially gradient descent).  

For this experiment, it was found through trial an error that allowing the perturbed image to be at most e < 0.14 away from the original was sufficient to allow consistent misclassification in the target direction. This e value is a per-pixel quantity and equates to the l_infinity distance metric from literature. 

## Analysis 
It is very simple to direct a standard MNIST classifier to misclassify 2s as 6s using the iterative FGSM technique. However, this technique requires white-box access to the network and its gradients. Future work will involve further black-box testing and restrictions on the number of pixels that can be perturbed. 

## Steps to Reproduce Results 
The code for this project is written primarily using Python3 and TensorFlow. To reproduce results, complete the following steps:

Setup: 
1) Python 3 Setup 
2) Run `pip3 install -r requirements.txt` from the root directory in this file to install necessary python packages.
3) Run `python3 train_mnist.py` in order to train a new mnist model. (This only needs to be run once)

Fast Gradient Sign Method
1) Run `python3 ./experiments/iterative_fgsm.py`. This file will generate adversarial examples for 10 random images of 2s and save results under at `./results/iterative_fgsm`

These results are shown here: 


## References and Suggested Readings 
[Ian J. Goodfellow, Jonathon Shlens: “Explaining and Harnessing Adversarial Examples”, 2014; arXiv:1412.6572.](https://www.sri.inf.ethz.ch/riai2017/Explaining%20and%20Harnessing%20Adversarial%20Examples.pdf)

[Anh Nguyen, Jason Yosinski: “Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images”, 2014; arXiv:1412.1897.](https://arxiv.org/pdf/1412.1897.pdf)

[Alexey Kurakin, Ian Goodfellow: “Adversarial Machine Learning at Scale”, 2016; arXiv:1611.01236.](https://arxiv.org/pdf/1611.01236.pdf)

[Nicholas Carlini: “Towards Evaluating the Robustness of Neural Networks”, 2016; arXiv:1608.04644.](https://arxiv.org/pdf/1608.04644.pdf)

[Andrew Ilyas, Logan Engstrom, Anish Athalye: “Black-box Adversarial Attacks with Limited Queries and Information”, 2018; arXiv:1804.08598.](https://arxiv.org/pdf/1804.08598.pdf)

[Anish Athalye, Nicholas Carlini: “Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples”, 2018; arXiv:1802.00420.](https://arxiv.org/pdf/1802.00420.pdf)

[Chaowei Xiao, Bo Li, Jun-Yan Zhu, Warren He, Mingyan Liu: “Generating Adversarial Examples with Adversarial Networks”, 2018; arXiv:1801.02610.](https://arxiv.org/pdf/1801.02610.pdf)

[Gamaleldin F. Elsayed, Ian Goodfellow: “Adversarial Reprogramming of Neural Networks”, 2018; arXiv:1806.11146.](https://arxiv.org/pdf/1806.11146.pdf)
# mnist-adversary

## Attack Methods 
There are a few different ways to generate adversarial examples for image recognition networks like the one operating on mnist data. These methods differ in that they have different levels of knowledge about the model architecture and its parameters. 

### Fast Gradient Sign Method 
We start with a white box method which requires access to network architecture for gradient calculations. This method was first theorized by Goodfellow et al. as a simple test of the linearity of a neural network. Starting from the original image that one wishes to generate adversarial examples for, one can iteratively descend in the direction of the gradient of the loss function calculated with respect to that image and the adversarial target class by subtracting a scaled multiple of sign of the gradient from the image. Accumulated over time steps, this cumulative subtracted image constitued the additive perturbation (reminiscent to gradient descent).  

## Steps to Reproduce Results 

## Papers Referenced 
https://www.sri.inf.ethz.ch/riai2017/Explaining%20and%20Harnessing%20Adversarial%20Examples.pdf

https://arxiv.org/pdf/1611.01236.pdf

https://arxiv.org/pdf/1608.04644.pdf

https://arxiv.org/pdf/1804.08598.pdf

https://arxiv.org/pdf/1802.00420.pdf

https://arxiv.org/pdf/1412.1897.pdf

https://arxiv.org/pdf/1801.02610.pdf
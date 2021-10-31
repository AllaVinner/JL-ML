# JL-ML - Investigating Autoencoders

This repository contains a number of pre made [autoencoder](https://en.wikipedia.org/wiki/Autoencoder 'Autoencoder Wiki') models and tools to visualize their results and inner workings. Several different architectures and loss functions are investigated through handwritten digit generation (MNIST). Below some interesting results and comparisons are shown! 

## 

The basic principle behind an autoencoder is compression. In the case of images we first feed the model an image, which it then is forced to represent more sparsely by design. You can think of this as a sort of bottleneck through which the image is sqeezed through. The figure below shows a rough sketch of an autoencoder, where the yellow part represents the bottleneck.

<p align="center">
  <img src="https://github.com/AllaVinner/JL-ML/blob/main/images/autoencoder_arch.png" alt="drawing" width="600"/>
</p>

([Image source](https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac 'Source'))

We call this the encoding step. The space in which this sparse encoding exists is called the _latent space_, since it is "hidden" inside the model. The model then tries to recreate the image using the latent representation, resulting in a similar but often blurrier image, since it has been subject to lossy compression. We call this the decoding step. The result is then compared to the original in order to improve the model.

You can read more in this [blog post](https://www.jeremyjordan.me/autoencoders/) and in [Goodfellow el. al](https://www.deeplearningbook.org/).

##

Our first example consists of ten recreated digits from a simple autoencoder. The upper row contains original images, and the bottom row the recreations.

<p align="center">
  <img src="https://github.com/AllaVinner/JL-ML/blob/main/images/autoencoder_10epochs_32bs_shallow_recon.png" alt="drawing" width="800"/>
</p>

As you can see they look a bit blurry, but we can still determine which digits they are. However the model also made some mistakes, including mistaking two "fours" for "nines", so there seems to be a bit of improvement needed.

Each digit is originally represented as a 28 by 28 image, i.e 784 pixels. In this example we use a latent space of dimension 2, which means that the model is forced to represent each digit using only two numbers. Overall this seems like quite a difficult task, but the nice thing is that it allows us to visualize the space!

In the image below the latent encoding of 10 000 characters can be seen, each one given by a point and a class color. We see that the model has created clusters for each digit class, and interestingly enough we can see that there seems to be a fair bit of overlap between the 4- and 9-clusters. This means that the encodings of 4's and 9's end up close to each other, which might be why our model mistanenly decoded a four as a nine above!

image: scatter plot of encoded images from test set

<p align="center">
  <img src="https://github.com/AllaVinner/JL-ML/blob/main/images/latent_scatter_plot_autoencoder.png" alt="drawing" width="400"/>
</p>
  
To get a better feel for the latent space we'll take a look at a couple of other digits in it. In the gif below we're using `sample_scatter_gui` to click around on different digits.

<p align="center">
  <img src="https://github.com/AllaVinner/JL-ML/blob/main/images/scatter_gui_gif.gif" alt="drawing" width="600"/>
</p>
  
So what might we use this for? One interesting application is generating new images of digits. If we randomly sample a point in the latent space, we might get a new digit, right? Let's try that. In the image below are 16 examples of random points in the latent space that have been decoded.

<p align="center">
  <img src="https://github.com/AllaVinner/JL-ML/blob/main/images/simple_auto_random_sampled.png" alt="drawing" width="600"/>
</p>

As you can see the results are pretty varied, with some images resembling digits, but others not at all. 

what if we choose points inbetween two encodings of other numbers?

TODO image: linear latent interpolation between two encodings

TODO: add motivation for variational autoencoders,

they are quite bad. How can we improve them? To force the autoencoder to create some structure in the latent space, or actually make the areas around the encodings also correspond to real numbers, we introduce a latent loss (explan why)

images: reconstructions and randomly generated images with VAE

image: scatter plot?

image: linear latent interpolation? better than AE?

they look better (hopefully). We can however see that some are quite blurry, and some do not even look like numbers. (Insert: why blurry?)

One way to try to make the images sharper is adjust the loss function. These are the ones we have used so far: 

image: plot of BCE loss and cont bern

why is bernoulli the right way? explanation (wrong to assume pixels discrete)

this results in

image: reconstructions without and with cont bern

![alt-text](https://github.com/AllaVinner/JL-ML/blob/main/images/cont_bern_comparison.png)

image: generative without and with cont bern


do u want to make ur own autoencoder? here is an example:


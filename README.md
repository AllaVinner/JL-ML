# JL-ML

This repository investigates autoencoders.

Introduction of autoencoders

Here below we see an autoencoder trained to reconstruct images from MNIST.

image: regular autoencoder reconstructions on test set

![alt-text](https://github.com/AllaVinner/JL-ML/blob/main/images/autoencoder_10epochs_32bs_shallow_recon.png)

They look okay. Since its latent dimension is two dimensional, we can take a look at the encodings of all the images in the test set.

image: scatter plot of encoded images from test set

![alt-text](https://github.com/AllaVinner/JL-ML/blob/main/images/latent_scatter_plot_autoencoder.png)

![alt-text](https://github.com/AllaVinner/JL-ML/blob/main/images/scatter_gui_gif.gif)

insert some cool insight (7)

One idea is to also use them to generate new images, by choosing points in the latent space. How does that look?

image: reconstructions of randomly sampled points?

what if we choose points inbetween two encodings of other numbers?

image: linear latent interpolation between two encodings

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


import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_comparison(images, figsize=None, row_labels=None):
  # images: (images_1, ..., images_n)
  # images_1.shape: (5, 28, 28, 1)
  
  """
  Example usage:
  
  images_original = x_test[:5, :, :, :]
  images_decoded = autoencoder(images_original)
  
  plot_mnist_comparison((images_original, images_decoded), figsize=(28, 8), row_labels=['original', 'decoded'])
  
  """

  nbr_rows = len(images)

  fig = plt.figure(figsize=figsize)
  fig.subplots_adjust(hspace=0, wspace=0)

  for row, row_images in enumerate(images):

    images_concat = np.concatenate(row_images[:,:,:,0], axis=1)
    
    ax = plt.subplot(nbr_rows, 1, row+1)
    fig_temp = plt.imshow(images_concat, cmap='bone', vmin=0, vmax=1)

    ax.get_xaxis().set_visible(False)

    if row_labels is not None: ax.set_ylabel(row_labels[row])

    ax.set_yticks([])
  
  plt.show()
  

def plot_2d_scatter(points, labels, figsize=(10, 10)):
  # points.shape: (nbr_points, 2)
  # labels.shape: (nbr_points,)
  fig, ax = plt.subplots(figsize=figsize)

  scatter = ax.scatter(points[:,0], points[:,1], c = labels.astype('float32'), cmap='tab10')

  legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
  
  ax.add_artist(legend1)

  plt.show()
# pix2pix for Image Segmentation Recovery: U-Net + PatchGAN

This project implements a **pix2pix** model combining **U-Net** and **PatchGAN** to recover the original image from segmentation labels. The model takes segmentation masks as input and generates the corresponding original image as output.

## Features

- **Input:** Segmentation label maps (segmentation mask)
- **Output:** Reconstructed original image
- **Model Architecture:**
  - **U-Net**: Used for encoding the input image (segmentation map) and decoding to produce the output image.
  - **PatchGAN**: A type of discriminator that classifies image patches as either real or fake, used to improve the quality of the generated image.

## Requirements

This project requires the following Python packages:

- `torch >= 1.12.0`
- `torchvision >= 0.13.0`
- `Pillow >= 8.0.0`
- `opencv-python >= 4.5.0`
- `torchinfo >= 0.0.8`
- `tqdm >= 4.60.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.4.0`
- `tensorboard >= 2.7.0`
- `argparse >= 1.4.0`
- `glob2 >= 0.7`

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
## ModelOverview

### Loss Function

#### Generator Loss Function in Pix2Pix

The generator's goal in Pix2Pix is to create images that are as realistic as possible to "fool" the discriminator, making it unable to distinguish generated images from real ones. The generator's loss function typically consists of two parts:

##### Adversarial Loss
Adversarial loss measures how well the generated image is classified as "real" by the discriminator. The generator's goal is to minimize this loss, thus generating images that the discriminator cannot distinguish from real images.

The adversarial loss is based on the standard GAN loss, assuming the discriminator is D , the generator is G , the real image is x , and the target output image is y. The adversarial loss can be written as:

$$
L_{\text{adv}}(G) = -\mathbb{E}_{y, z}[\log D(G(z))]
$$

where z is the input to the generator (usually noise or a conditional image), and D(G(z)) is the probability that the discriminator classifies the generated image as real. The goal is to make this probability as close to 1 as possible, so the generated image looks as real as possible.

##### L1 Loss
L1 loss ensures that the generated image is close to the real image at the pixel level. The L1 loss is a core part of the generator's objective, minimizing pixel-wise differences between the generated and real images. Assuming y is the target image and G(x) is the generated image, the L1 loss is calculated as:

$$
L_{\text{L1}}(G) = \mathbb{E}_{x,y}[\| y - G(x) \|_1]
$$

L1 loss helps the generator learn finer details and structures, making the generated image appear more natural.

##### Total Generator Loss
The total generator loss L_G is the weighted sum of adversarial loss and L1 loss, where a weight coefficient lambda is used to balance the two components:

$$
L_G = L_{\text{adv}}(G) + \lambda L_{\text{L1}}(G)
$$

This weight coefficient lambda helps balance the generator's performance in "fooling" the discriminator and preserving image structure.

---

#### Discriminator Loss Function in Pix2Pix

The discriminator's goal is to distinguish between real and generated images, i.e., maximize the probability of real images being classified as "real" and generated images being classified as "fake." The discriminator's loss function consists of two parts:

##### Adversarial Loss
The adversarial loss for the discriminator measures its performance. The discriminator aims to maximize the probability of real images being classified as real and minimize the probability of generated images being classified as real. Therefore, the discriminator's loss function is penalized based on the discriminator's output probabilities D(x) for real images and D(G(z)) for generated images:

$$
L_{\text{adv}}(D) = - \mathbb{E} _ {x,y}[\log D(x)] - \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$


Where D(x) is the discriminator's output probability for real images x , and D(G(z)) is the discriminator's output probability for generated images G(z). The goal is to maximize D(x) and minimize D(G(z)) so that the discriminator correctly distinguishes between real and generated images.

##### Total Discriminator Loss
The total discriminator loss is simply the adversarial loss:

$$
L_D = L_{\text{adv}}(D)
$$

---

#### Combined Loss Functions

In Pix2Pix, the generator and discriminator loss functions work together, forming a typical GAN structure. The generator aims to generate images with both low adversarial loss and low L1 loss, while the discriminator learns to better distinguish real from generated images.

The final loss functions are as follows:

- Generator Loss:
  
$$
L_G = L_{\text{adv}}(G) + \lambda L_{\text{L1}}(G)
$$

- Discriminator Loss:
  
$$ 
L_D = L_{\text{adv}}(D) 
$$

Where lambda is a hyperparameter that controls the relative importance of the L1 loss, typically set around 10. By optimizing both the generator and discriminator losses simultaneously, Pix2Pix gradually learns to generate more realistic images during training.


### U-Net Encoder-Decoder
The encoder part of the U-Net learns the features of the input segmentation label, while the decoder reconstructs the original image. Skip connections between corresponding layers in the encoder and decoder help preserve spatial information, ensuring the output is as close to the original image as possible.

### PatchGAN Discriminator
The PatchGAN discriminator evaluates small patches within the generated image and classifies them as either real or fake. This type of discriminator helps improve the quality of the output by forcing the generator to produce more realistic details.


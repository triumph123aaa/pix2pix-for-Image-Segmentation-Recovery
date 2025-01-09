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

## ModelOverview

### U-Net Encoder-Decoder
The encoder part of the U-Net learns the features of the input segmentation label, while the decoder reconstructs the original image. Skip connections between corresponding layers in the encoder and decoder help preserve spatial information, ensuring the output is as close to the original image as possible.

### PatchGAN Discriminator
The PatchGAN discriminator evaluates small patches within the generated image and classifies them as either real or fake. This type of discriminator helps improve the quality of the output by forcing the generator to produce more realistic details.


# Convolutional networks (part 6, neural architectures)

Convolution and the networks built on it: the building blocks (convolution, padding, stride, pooling, normalisation) watched on an image, the classic architectures at small scale, dense prediction (segmentation, detection heads), convolution beyond images (1-D on sequences, graphs), and interpreting convolutional networks (filters, feature maps, saliency, class activation maps), as sessions over `nn`, `vision` and the trainer base.

## Capabilities

- A convolution stepped: the kernel sliding over the image, the output pixel computed; options: the kernel (named or learned), padding, stride, dilation.
- Training a small network on `datasets` shapes or digits: the filters and feature maps as probes at the viewed step, the accuracy on the tape.
- Dense prediction: a small U-Net on the geometric scenes, the predicted mask against the truth each epoch.
- Interpretation: saliency and Grad-CAM for a chosen image at the viewed step (with `trustworthy-machine-learning/interpretability`).
- Receptive fields grown layer by layer.

## Examples to make

- The sliding kernel with the output filling in, kernel as a knob (edge, blur, learned).
- First-layer filters becoming edge detectors over training, the feature maps of one image beside them.
- A U-Net's mask sharpening epoch by epoch on a scene (the vision methods' scenes).
- Grad-CAM over training: where the network looks as it learns.

## Rendering

- Images and image steps (`image`, `imageSteps`), grids of filters and feature maps, masks over images (the vision `picture` helper), curves.
- Missing: an image-grid panel kind (many small images at the viewed step with labels).

## Shared

- `shared/`: the image data slot with truth (from `datasets` and `vision`), the feature-map probes; the trainer base from `feedforward-networks`.

## Problems

- Shares the scene data of `computer-vision`; convolution on the main thread is slow: tiny images, and the worker.

# Self-Driving Car Technology - Transfer Learning

## Overview

In other autonomous vehicle software stacks, I have built, trained, and operated various deep neural networks from scratch for image classification tasks, using training data I have either obtained from others or generated myself ([traffic sign classification](https://github.com/dinoboy197/CarND-Traffic-Sign-Classifier-Project), [vehicle detection and tracking](https://github.com/dinoboy197/CarND-Vehicle-Detection), etc). However, many deep learning tasks can use pre-existing trained neural networks from some other similar task, and with some tweaks to the network itself, can significantly reduce the effort and shorten the time to production. **Transfer learning** is the technique of modifying and re-purposing an existing network for a new task.

Some popular high-performance networks include [VGG](https://arxiv.org/pdf/1409.1556.pdf), [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf), and [ResNet](https://arxiv.org/pdf/1512.03385.pdf). Models for these networks were previously trained for days or weeks on the [ImageNet dataset](http://www.image-net.org/). The trained weights encapsulate higher-level features learned from training on thousands of classes, yet they can be adapted to be used for other datasets as well.

## Example pre-trained networks

Some existing networks which can be used for new tasks using transfer learning include:

* **VGG** - A great starting point for new tasks due to its simplicity and flexibility.
* **GoogLeNet** - Uses an inception module to shrink the number of parameters of the model, offering improved accuracy and inference speed over VGG.
* **ResNet** - Order of magnitude more layers than other networks; even better (lower error rate) than normal humans at image classification.

## Transfer learning details

Depending on the size of the new dataset, and the similarity of the new dataset to the old, different approaches are typical when applying transfer learning to repurpose a pre-existing network.

### Small dataset, similar to existing

* Remove last fully connected layer from network (most other layers encode good information)
* Add a new fully connected layer with number of classes in new dataset
* Randomize weights of new fully connected layer, keeping other weights frozen (don't overfit new data)
* Train network on new data

### Small dataset, different from existing

* Remove fully connected layers and most convolutional layers towards the end of the network (most layers encode different information)
* Add a new fully connected layer with number of classes in new dataset
* Randomize weights of new fully connected layer, keeping other weights frozen (don't overfit new data)
* Train network on new data

### Large dataset, similar to existing

* Remove last fully connected layer from network (most other layers encode good information)
* Add a new fully connected layer with number of classes in new dataset
* Randomize weights of new fully connected layer, and initialize other layers with previous weights (don't freeze)
* Train network on new data

### Large dataset, different from existing

* Remove last fully connected layer from network (most other layers encode good information)
* Add a new fully connected layer with number of classes in new dataset
* Randomize weights on all layers
* Train network on new data
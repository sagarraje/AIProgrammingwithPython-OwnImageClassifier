# AIProgrammingwithPython-OwnImageClassifier

![alt text](https://github.com/sagarraje/ImageClassifier/blob/main/certification.png?raw=true)

===================================================================
Python, NumPy, Pandas, Matplotlib, PyTorch, and Linear Algebra—the foundations for building your own neural network. 
===================================================================


Meets Specifications
Congratulations :trophy: for completing this project!
I certainly enjoyed walking through your code. It's very clean and very well commented. I can clearly see the effort that has been put into this.
I tried the command line utility too and it is working correctly.
Also, I have provided some notes and resources related to this project. Kindly make sure to check them out.

Keep learning :udacious: and All the best! :smiley:



Here are some additional links that might help you further your understanding-

Pytorch Notes
PyTorch Recipes
Data Augmentation
Specific to this flower classification problem:

Flower Categorization using Deep Convolutional Neural Networks
Recognition between a Large Number of Flower Species
Files Submitted
The submission includes all required files. (Model checkpoints not required.)

All the required files have been included.

Part 1 - Development Notebook
All the necessary packages and modules are imported in the first cell of the notebook

All the necessary packages and modules have been imported in the first cell of the notebook.
Moving all the imports to the top is a good practice as it helps in looking at the dependencies and the import requirements of the project in one go.

torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

Good job :thumbsup: randomly cropping, rotating, and mirroring the images. This allows the model to generalize better.
For a list of more transformations possible, you can check out- Transformations in PyTorch.

:memo: Notes:

Data augmentation artificially boosts the diversity and number of training examples by performing random transformations to existing images to create a set of new variants.
This helps the model to fit in a better way.
Data augmentation is especially useful when the original training data set is relatively small.
The training, validation, and testing data is appropriately cropped and normalized

Data is properly resized, cropped, and normalized.

:memo: Resource:

There are various ways to accelerate training and inference of a deep learning model. Examples-
Enabling async data loading and augmentation.
Disabling gradient calculation for validation and inference.
You can visit the performance optimization guide to find ways to accelerate training and inference of deep learning models in PyTorch.
The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

The data for each set is loaded with torchvision's DataLoader

The data for each set is loaded with torchvision's DataLoader.

:memo: Note:

Generally, models trained using smaller batch sizes generalize better. For more, you can visit- ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA.
A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen

Good job :thumbsup: using vgg16 as the pre-trained model.
For a list of other models that we can use, you can check out- PyTorch pre-trained classifiers.

A new feedforward network is defined for use as a classifier using the features as input

Good job using ReLU as activation function and Dropout layers to better regularize the model.

:memo: Notes:

Typically, dropout will improve generalization at a dropout rate of between 10% and 50% of neurons.
To see how dropout impacts various aspects like test error, features, sparsity, etc., you can visit the following research paper- Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

You are only training the classifier layers and not the layers from the pre-trained models.

:memo: Resource:

Finding good values for hyperparameters can be challenging.
You can learn about hyperparameter tuning here. This can help us optimize our hyperparameters.
During training, the validation loss and accuracy are displayed

Training loss, validation loss, etc. are correctly displayed during the training.

:memo: Notes:

We can plot our training and validation losses. This can help us better visualize the model. Also, it can help us debug issues related to training by interpreting the loss curves. For more, you can visit- interpreting loss curves.
The network's accuracy is measured on the test data

Good job :thumbsup: achieving a test accuracy of 81%.

There is a function that successfully loads a checkpoint and rebuilds the model

The model is correctly recreated from the checkpoint.

The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

The model, associated hyperparameters, and class_to_idx dictionary are correctly saved in the checkpoint.

The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

Part 2 - Command Line Application
train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint

Good job with the command line utility. The arguments are correctly parsed and the model is correctly trained and saved as a checkpoint.

The training loss, validation loss, and validation accuracy are printed out as a network trains

Training loss, validation loss, etc. are correctly printed during the training.

The training script allows users to choose from at least two different architectures available from torchvision.models

Good job :thumbsup: allowing users to choose between alexnet and vgg16 as the model architecture.

The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

The training script allows users to choose training the model on a GPU

The user is able to train the model correctly on the CPU or the GPU.

The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability

predict.py correctly makes predictions for the given image and the checkpoint.

The predict.py script allows users to print out the top K classes along with associated probabilities

The predict.py script allows users to load a JSON file that maps the class values to other category names

The predict.py script allows users to use the GPU to calculate the predictions

The user is able to use either the CPU or the GPU for predictions.
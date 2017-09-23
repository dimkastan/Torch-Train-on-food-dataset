# Using Torch to train a Deep CNN- an example case on a food dataset
Torch demo for training squeezenet from scratch to recognize foods.

## Note: why I am using food-101
- Is a large enough dataset, however better manageable as compared to ImageNet
- Is better than MNIST
- Is bettern than cats-vs-dogs
- It is worth studying the efficiency wrt the color space
- It is very interesting to study week localization and transfer learning.


## Setup 
 <br />
Before begin you have to setup your environment. First of all you must have allready installed, Torch and the image, optim, nn, cudnn and cunn packages. This can be done easilly via luarocks install <package_name>
 <br />
After installing all requirements follow these steps:
<br />
Get this repo:
<br />
    git clone https://github.com/dimkastan/Torch-Train-on-food-dataset
<br />
<br />
Download and extract data
<br />
All steps are included into setup.sh . Call it using:
<br />
 
    ./setup.sh
 
 <br />
If you are under Windows 10 OS you can use Windows 10 Ubuntu bash.
<br />
Now everything ready to start training. <br />
Currently, only "Train from scratch" is supported. In a few weeks I will add two very basic "Transfer Learning" scenarios.
<br />

## Train from scratch
<br />
After extracting the data you can easily start training your model by calling the train.lua as follows: <br />
 
    th main.lua
 
<br />
The program will draw the Classification accuracy as well as the loss per epoch.
<br />
According to your GPU's memory you can modify the batch size. Also I would advise you to work on a SSD <br />
<br />

<br />


Here are the accuracy plots for both train and test set (single crop) <br />
<p align="center">
  <img src="https://github.com/dimkastan/Torch-Train-on-food-dataset/blob/master/train_sqres_log.jpg" alt="Train Set">
</p>
 <br />
<p align="center">
  <img src="https://github.com/dimkastan/Torch-Train-on-food-dataset/blob/master/test_sqres_log.jpg" alt="Test Set">
</p>
 


### Upcoming features
<br />
In a few weeks I plan to add one or more of the following features:
<br />
- Perform data augmentation <br />
- Operate in different color spaces <br />
- Incorporate food localization <br />
- Incorporate semantic image segmentation <br />
 


## Test a pretrained model
<br />
For convinience here, I also provide a pretrained model (squeezenet_v1) as well as sample code demonstrating how to load and classify images.
<br />
In order to test the model run the following:
<br />

    th demo_food.lua

This will crop one rectangle from apple_pie.jpg and evaluate it. Run multiple times to see how the output is affected by random crops.

<br />









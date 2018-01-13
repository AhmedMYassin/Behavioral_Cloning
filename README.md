## Behavioral Cloning Project

In this project, I have a simulation environment created by Udacity and I am supposed to train a model that can use this simulation to drive a vehicle in autonomous mode around the track without leaving the road.

### Model Architecture and Training Strategy

#### 1. Model architecture 

I have used a model similar to the nividia model architecture to train the data. The model is consisted of 5 convolutional layers using 5x5 kernel then 3 layers of fully connected neural network. 

#### 2. Attempts to reduce overfitting in the model

To overcome overfitting issue, I get rid of 70% of images that have 0 steering angle (I used these images later for testing). This trick changed the behavior of the vehicle completely. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I used also 0.15 correction factor for left and right images. I used dropout (0.8) to overcome overfitting

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving counter clock_wise, recovering from the left and right sides of the road and flipped center images with adjusted steering angle.
I got rid of 70% of images that have 0 steering angle to avoid the model being biased.


### Architecture and Training Documentation

#### 1. Solution Design Approach

First, I started by following the tutorial and implementing that simple neural network to make sure the model can be trained to be used for simulation regardless the accuracy. 

After that, I used the proposed architecture of nividia but the accuracy in the simulation was very bad. I realized that I have a problem in training data normalization. I also filtered the training data and ignored 70% of center images that have steering angle less than 0.05 to prevent the model from being biased.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 33x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 15x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 11x73x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 9x71x64 	|
| RELU					|												| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 7x69x64 	|
| RELU					|												| 
| Flatten     	| 7x69x64 to 30912 neurons 	|
| Dropout					|	0.8											| 
| Fully Connected     	| input 30912, output 100 	|
| Fully Connected     	| input 100, output 50 	|
| Fully Connected     	| input 50, output 10 	|
| Fully Connected     	| input 10, output 1 	|


#### 3. Creation of the Training Set & Training Process

I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

<img src="https://github.com/AhmedMYassin/Behavioral_Cloning/blob/master/examples/center_image.jpg"> 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center. These images show what a recovery looks like:

<p float="left">
<img src="https://github.com/AhmedMYassin/Behavioral_Cloning/blob/master/examples/Back_to_Center_1.jpg" width="280"> 
<img src="https://github.com/AhmedMYassin/Behavioral_Cloning/blob/master/examples/Back_to_Center_2.jpg" width="280"> 
<img src="https://github.com/AhmedMYassin/Behavioral_Cloning/blob/master/examples/Back_to_Center_3.jpg" width="280">
</p>

Then I recorded 2 laps on track one by driving counter clockwise.

As most of center images has steering angle less than 0.05, I only used 30% of these center images.

After the collection process, I had 9729 images and to augment the data while training I flipped center images and the steering angles to prevent the model being biased.

While training the data I used left and right images after adjusting the steering angle with a correction factor 0.15. I also cropped the input data for the model to focus only on the road shape.

For training, I used 2 epochs to train the model, 3 epochs also was ok but the difference in training loss and validation loss was very small. Higher number of epochs causes overfitting in the model. Here the training and validation loss for each epoch while training:

1st epoch: training loss = 0.11 ,validation loss = 0.097

2nd epoch: training loss = 0.0198,validation loss = 0.023


### Simulation

[Here](https://github.com/AhmedMYassin/Behavioral_Cloning/blob/master/video.mp4) you can check a video for the simulation in autonomous driving mode.

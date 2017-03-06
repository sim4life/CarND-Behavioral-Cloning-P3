#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:  
* Use the simulator to collect data of good driving behavior  
* Build, a convolution neural network in Keras that predicts steering angles from images  
* Train and validate the model with a training and validation set  
* Test that the model successfully drives around track one without leaving the road  
* Summarize the results with a written report  


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/training-624x291.png "Solution Design"
[image3]: ./examples/center_2017_02_24_22_48_25_534.jpg "Centre lane driving"
[image4]: ./examples/center_2017_02_25_15_07_08_004.jpg "Reverse driving"
[image5]: ./examples/center_2017_02_25_15_00_25_816.jpg "Recovery Image"
[image6]: ./examples/center_2017_02_25_15_01_13_151.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:  
* `model.py` containing the script to create and train the model  
* `drive.py` for driving the car in autonomous mode  
* `model.h5` containing a trained convolution neural network   
* `video.mp4` containing the video of driving the car in autonomous mode
* `writeup.md` summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I made use of the convolutional network presented by nVidia for autonomous cars.

The model first normalizes the inputs with `(x/127.5) - 1.0` formula using Keras Lambda layer (model.py line120).  
After that, my model crops the inputs from top by 70 pixels and bottom by 25 pixels. This vastly improves training the model as the top contains irrelevant scenery and bottom contains the hood of the car (model.py line 121).
Next, the model consists of 3 convolution neural network layers with 5x5 filter sizes and depths between 24 and 48 (model.py lines 122-124). These layers have a stride of 2x2 and an activation function of ReLU.  
After that, the model consists of 2 convolution neural network layers with 3x3 filter size and depth of 64 each (model.py lines 125-126). These layers also have an activation function of ReLU.  
After that, the model flattens the inputs (model.py line 127). Lastly, the model runs 4 fully connected layers giving output depth of 100, 50, 10 and 1 (model.py lines 128-131)

####2. Attempts to reduce overfitting in the model

The model contains random adjustments to steering angles outputs in order to reduce overfitting (model.py lines 89-91). Without these adjustments, the model was getting trained leaning more towards straight driving because of the bias of heavy use of straight driving in training data.  
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 133).

####4. Appropriate training data

Significant effort was done to select training data to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, reverse driving and zig-zag driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a basic LeNet-5 and train the model on Udacity provided sample dataset to consider it as a baseline to judge improvements.  
Later on, I employed various models, including LeNet-5, nVidia CNN, nVidia CNN with dropouts etc. In all the models, I found both the training error and validation error to be low.  However I found out that classic nVidia model performed better than others in driving the car in autonomous mode.  
The final step was to run the simulator to see how well the car was driving around track 1. There was one particular spot, the left turn right after the bridge, where the vehicle fell off the track. To improve the driving behavior in this and other cases, I put more emphasis on capturing quality data, doing dataset correction and dataset augmentation. These techniques made the trained model drive the car far better.  
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here is a diagram of that process

![alt text][image2]

####2. Final Model Architecture

The final model architecture (model.py lines 119-131) consisted of a convolution neural network with the following layers and layer sizes in the diagram below:

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to centre after steering close to the edges. These images show what a recovery looks like starting from:

![alt text][image5]
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would remove the counter-clockwise driving bias for the track 1. For example, here is an image that has then been flipped:

![alt text][image4]

After the collection process, I had 27996 number of data points. I then preprocessed this data by inflating (by a factor of 2) 40% of the steering angles of straight driving (-0.5 < angle < 0.5), adding adjusted left and right camera angles, augmenting the data by flipping the images and their corresponding steering angles.  
I finally randomly shuffled the data set and put 20% of the data into a validation set.  
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

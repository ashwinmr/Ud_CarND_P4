# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/recovery_before.png "Recovery Image"
[image4]: ./examples/recovery_during.png "Recovery Image"
[image5]: ./examples/recovery_after.png "Recovery Image"
[image6]: ./examples/image_org.png "Normal Image"
[image7]: ./examples/image_flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* output.mp4 video showing the drive

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depth of 32 (code lines 60).

The model includes RELU layers to introduce nonlinearity (code line 63), and the data is normalized in the model using a Keras lambda layer (code line 57).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (code line 62). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 69).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and shifted camera angles with correction to learn recovery. 

I also used image cropping to get rid of unecessary image features (code line 59)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very basic flat model that first validated the pipeline.

My first step was to augment the data with flipped images to prevent bias for steering towards one side.
I then augmented it further with camera shifted images to allow the model to learn how to correct over steering.
I also cropped the images to get rid of unecessary image features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

I noticed that the training and validation loss was very high, which indicated that the model was underfitting. So I increased complexity by adding a convolution layer and max pooling to pick out small features in the image. This would pick out the road curvature to determine steering angle.
I added non-linearity using a relu.

This decreased the training loss but caused overfitting. So I used a dropout layer to prevent overfitting the training data.

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-66) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        	 | 
|:---------------------:|:------------------------------:| 
| Input         		| 160x320x3 color image   	     | 
| Lambda             	| Normalization layer	         |
| Cropping			    | Cropping layer 65x320x3		 |
| Convolution 3x3	    | 1x1 stride,  outputs 63x318x32 |
| Max pooling           | 2x2 stride, outputs 31x159x32  |
| Dropout               |                                |
| Relu                  |                                |
| Fully connected		| output 1       			     |

#### 3. Creation of the Training Set & Training Process

I used the provided training data with center and side camera images.

I used the side camera images with steering correction to teach the car how to recover. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would teach the car how to turn in the opposite direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I then preprocessed this data by cropping it.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 since the loss stopped decreasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
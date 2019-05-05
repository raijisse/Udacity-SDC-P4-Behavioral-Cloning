# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./assets/nvidia-architecture.png "Model Visualization"
[image2]: ./assets/center_track_one.jpg "Center"
[image3]: ./assets/center_from_right.jpg "Recovery Image"
[image4]: ./assets/center_track_two.jpg "Track 2"
[image5]: ./assets/center_track_two_flipped.jpg "Flipped"
[image6]: ./assets/center_cropped.jpg "Cropped"



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


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Be careful that the paths are relative to my workspace (a `/data` folder) that is not included in the git repository. For the code to run successfully, either put your valid training data in a `data` folder or simply change the path to your data.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As suggested in the lesson, I used the Nvidia architecture as a starting point. This architecture uses five convolutional layers with 5x5 filters for the first three layers followed by 3x3 filters for the last two. The depths for the five layers are respectively 24,36,48,64,64 (see `model.py` lines 98-102). The convolutional layers are followed by four fully connected layers.

The model includes ReLU activation layers to introduce nonlinearity (`model.py`, lines 95-110) and the data is normalized in the model using a Keras lambda layer (`model.py`, line 96).

#### 2. Attempts to reduce overfitting in the model

With regard to the original architecture, I added three dropout layers in between the fully connected layers (except before the last one; see `model.py`, lines 104-108) to reduce overfitting.

Moreover, the model was trained and validated on different data sets to ensure that the model was not overfitting (see `model.py`, lines 38, 88-89, 119-123). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py`, line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Indeed, I used a combination of center lane driving, recovering from the left and right sides of the road as well as some driving from test track 2 to add variety to the data set.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

By using the Nvidia architecture as baseline, I knew that I would not allow much room for improvements. Indeed, the Nvidia model was developped especially to do what was asked in the project, so this was the most appropriate model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. By running a few experiments, I figured that the training loss was always decreasing while the validation loss was going back up at some point.
This is the main symptom of overfitting, so I added dropout layers in between the fully connected layers to counter this effect.

As suggested in the lesson I normalized and cropped the input for the model to focus on the road and not be disturbed by the environment.

On a side note, I would like to stress out the importance of using generators in the design process. Using generators on my local machine enabled me to speed-up the process (107s vs 152s per epochs), while reducing by a factor of 3 RAM use. This is great since it enables to experiment more, or keep working on others tasks while training a model !

Finally, I ran the simulator to see how well the car was driving around track one. As I had already included the original sample data provided, some more center lane driving around track one of my own, a few minutes of driving around track two, and data augmentation, my model had everything it needed to learn efficiently to drive around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I even was able to increase the driving speed from the `drive.py` to 30 MPH without problems.

However, I tested the model on track two. It could manage to go around the first curves, even with dark shadow but got on the wrong lane at some point and could not manage to make one of the very steep curves. In the end, I believe the model still needs to see some more variety and adding a few more minutes of driving around track two would probably help.

#### 2. Final Model Architecture

The final model architecture (`model.py`, lines 95-110) consisted of five convolutional layers with 5x5 filters for the first three layers followed by 3x3 filters for the last two. The depths for the five layers are respectively 24,36,48,64,64 (see `model.py` lines 98-102). The output of the convolutional layers is flattened to a vector of size 2112 (because of the dimensions of our input frames) which is connected to four fully connected layers with respective sizes of 100, 50, 10 and 1 for the output.

Here is a visualization of the architecture done with [NN-SVG](http://alexlenail.me/NN-SVG/AlexNet.html). Note that this visualization does not include the dropout layers in between the fully connected layers.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the sample data provided with the project. I then recorded a bit more than two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to center from one side of the road. This image show what a recovery looks like starting from the right side :

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would add variety and prevent the model from learning only on one side (left in track one for instance). For example, here is an image that has then been flipped (it is the same picture than above):

![alt text][image5]

After the collection process, I had 26 694 data points : 4 449 frames that are captured by the center, the left and right cameras (4 449*3 = 13 347 frames) that were flipped (resulting in 13 347*2 = 26 694 frames).

I then normalized the input images using a `Lambda` layer and cropped the 70 top pixels and 25 bottom pixels using the `Cropping2D` layer from Keras. The resulting image passed to the model looks like this (before normalization):

![alt text][image6]

I finally randomly shuffled the data set and put 25% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I stopped the number of epochs at 7 as training loss and validation loss were almost identical and that the loss started plateauing.

# TrafficSignClassifier
CNN to classify German traffic signs for SDCND program

---
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/brummell/TrafficSignClassifier/blob/master/Updated_Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

*see cell 5 of ipython notebook

I used simply the len() method with the set ds to calculate the required summary statistics:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

*see cell 6 of ipython notebook

I first did the obvious and created a histogram of the label counts in the overall data set. This is when I realized augmentation of the data set would not be optional if a high score was required, especially using my fairly bare-bones model. See below:

![alt text][labelhisto]

I also took the equally obvious step of printing a tiling of random images a few times, just to get a sense of the quality, centering, shading, perspective, etc. of the data set.

![alt text][randoimgtiling]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

*see cell 8 of ipython notebook

My initial preprocessing pipeline was extremely simple, and consisted only of rescaling and centering the pixel values to a range of \[-0.5, 0.5\] (ensuring an even start across all feature in the images) (in fact, in the process of copying my code from an old notebook to a new, I left this out, and was reminded of how painful --or in some cases, seemingly intractable- training can be without providing the net with a narrowed, centered range of values for the data). I chose this route over a channel-wise mean/std normalization primarily because I saw no noticeable improvements using these, and they both require fairly large array calculations, as well as the persistance of the mean and std values for later use in testing and prediction. I also chose to forgo anything like PCA-based decorrelation for the sake of time, though I heard little of any major successes with it among fellow students.

In terms of color, I took the advice of Vivek and, rather than grayscaling my images, used a layer of three 1x1 kernels in order to learn an optimal color mapping. In retrospect, looking at some of the failures in my from-the-web prediction set, I'm not entirely convinced retaining the color channels wasn't distracting to the final classifier, as it often seemed to prioritize color similarities even over drastically different geometries.

See the next point below for a discussion of my expansion of the preprocessing via random augmentation, as well as example images from the same.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

*see cell 8 of ipython notebook

For the basic division of data, after the (below-described) augmentation steps, I simply shuffled all of the data and then, skimmed 20% off of the training set to use for validation. The final breakdown was 77400 training examples, 19350 validation examples, and 12630 (unaugmented) testing examples. 

For better of worse, I wound up finishing the behavioral cloning project ahead of this one (who can turn down video games?! ;-)) and so I decided to improve upon my augmentation pipeline from that project and then fold it back into this project --as I was essentially stuck at ~96.2 accuracy without it, and began to figure the above-mentioned class imbalance more seriously. My augmentation set again drew considerable inspiration from Vivek, who himself mentioned recieving some help from a fellow student of ours, but whose name I can't remember. For this particular project, I wound up using only brightness augmentation, random linear shadow insertion, and affine transforms (well, only translation in this case). I applied them, randomly, and sometimes overlapping (multiple augmentations to one image at one time), and the augmentation methods themselves contain randomized parameters for brightness change, shadow insertion coordinates, etc. I also experimented with how much I augemented deficient sets, but in the end found that matching them to the cardinality of the most populus label worked out just fine. See below:

* orignial image
* ![alt text][augmentedimages0]

* translated image
* ![alt text][augmentedimages1]

* brightened image
* ![alt text][augmentedimages2]

* shadow image
* ![alt text][augmentedimages3]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 








I used http://www.gettingaroundgermany.info/zeichen.shtml as a primary reference for signs. It seems fairly comprehensive and up-to-date with deprecations.

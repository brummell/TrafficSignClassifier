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

*see cell 9 of ipython notebook

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution     | 3 1x1 kernels; 1x1 stride; same padding; relu activation	|
| Convolution     | 32 5x5 kernels; 1x1 stride; same padding; relu activation; no regularization; no downsampling 	|
| Convolution     | 32 5x5 kernels; 1x1 stride; same padding; relu activation; no regularization; 2x2 max-pooling 	|
| Convolution     | 64 3x3 kernels; 1x1 stride; same padding; relu activation; no regularization; no downsampling 	|
| Convolution     | 64 3x3 kernels; 1x1 stride; same padding; relu activation; no regularization; 2x2 max-pooling 	|
| Convolution     | 96 3x3 kernels; 1x1 stride; same padding; relu activation; no regularization; no downsampling 	|
| Convolution     | 96 3x3 kernels; 1x1 stride; same padding; relu activation; no regularization; 2x2 max-pooling 	|
| Merging         | each of the max-pooled conv layer outputs as input; concantenated 	|
| Fully-Connected | 200 dimensional output; relu activation; dropout with probability of 0.5; no downsampling 	|
| Fully-Connected | 100 dimensional output; relu activation; dropout with probability of 0.5; no downsampling 	|
| Fully-Connected | 43 dimensional output; relu activation; no regularization; no downsampling 	|
| Softmax         | |
| Loss            | cross-entropy loss function |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

*see cell 10 of ipython notebook

I used the Adam optimizer exclusively, enjoying it's largely "set-and-forget" capabilities and never felt that the optimization was dragging. Though I experimented an order of magnitude in either direction, I found that the learning rate --the only parameter that I tuned- that yielded the best results was 1E-3. I gradually scaled my batch size up in integral multiples of 128 until I had come comfortably close to saturating my GPU. I experimented with epoch counts ranging from 10 up to 40, but found that there was little gain after epoch 20 in most cases, and in fact, there frequently was overfitting.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

*see cell 10 of ipython notebook

The inspiration for my solution came in two forms, and from two major sources. The architecture itself was drawn from Sermanet and LeCun's Traffic Sign Recognition with Multi-Scale Convolutional Networks (it being focused and successful on almost this exact task), most notably, the multi-scale feature feed into the output classifier (concatenating the outputs of each convolutional layer pair, and passing that into the first fully-connected layer, allowing the final classifier the full breadth of feature scales). Beyond that, I also used a fairly similar architecture, with the following expansions: 
* In experimenting, I found gains in 3 each of fc and conv. layers, and particularly in limit 5x5 kernels to the first layer, any expansion in either depth or creep of 5x5 kernels led to early, strong overfitting.
* As mentioned before, I used a color-mapping layer rather than grayscale. They used both and had preferable results with the latter.
* I used a ReLU activation rather than their tanh, this was merely out of habit, and I can't comment on the performance of my architecture with the latter.
* I used RGB throughout, with temporary conversions to HSV in the augmentations only.
* I only experimented with perpetually trainable parameters throughout all of my experimenting.
* Though I got the inspiration for augmentation from the paper originally, I added brightness-related augmentations and used only the translational affine transforms.  
* My decision to use same padding was initially easy dimensionality calculation and tracking, but experimenting with valid padding showed no improvement in the architectures in which it was employed.

My best run model results were:
* validation set accuracy of 0.999 
* test set accuracy of 0.973
which were on par or exceeded with the lower of the paper's best performing six competition results. Still, their top model managed just over 99% accuracy, obviously leaving room for me to improve. Namely, I'm interested in --in the future, experimenting with inception models and transfer learning generally. I expect there is also still a considerable number of augmentation functions that could be added to increase the robustness of the system.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I pulled fifty or so images of various German traffic signs from Google Images. I also found it all but necessary to crop them to something akin to the training set. Other, less-addressable issues with the images in terms of classification included: examples from a class of sign in the dataset, but with inner symbols not seen in the dataset; considerably higher amounts of distracting imagery, often including other, smaller signs directly adjacent to the primary sign; numeric portions (both in the sense of not existing outside the dataset, and in that getting the broader sign class but not the number (e.g. speed) was a very common issue. 

The images, as well as histograms of the K top softmax outputs for the images are displayed at the bottom of this report. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:



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

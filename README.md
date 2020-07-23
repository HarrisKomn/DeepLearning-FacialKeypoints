# Aim 
The aim of this project is to predict keypoint positions on face images. This can be used as a building block in several applications, like:
face recognition, tracking faces in images and video, analysing facial expressions, detecting dysmorphic facial signs for medical diagnosis. The dataset is provided by Kaggle Community
and contains a list of 7049 training images and 1783 test images in training.csv and test.csv file respectively. Each raw of training.csv contains the (x,y) coordinates for 15 keypoints, and image data as 
row-ordered list of pixels.


# Model 
A CNN model is developed using Tensorflow. The network consists of the repeated application of two 3x3 convolutions , each followed by
LeakyReLU and a 2x2 max pooling operation with stride 2 for downsampling. Almost at each downsampling step we double the number of feature
channels.  The Convolutions are followed by 1 Dense layer at the final stage of the newtork. The following image shows the predicted facial key points (green) and the 
given facial key points from the dataset (red): 

![image](https://user-images.githubusercontent.com/43147324/88334071-de9ecb00-cd39-11ea-9a34-7b581643d61c.png)







# Data
 National Library of Medicine
 (https://lhncbc.nlm.nih.gov/publication/pub9932)

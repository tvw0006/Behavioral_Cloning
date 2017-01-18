# Behavioral_Cloning
Behavioral Cloning Project

**Scope** : The purpose of this project is to guide a simulated car through lanes using a convoluted neural network.

**Process** : The simulator records images from 3 cameras as shown below. The images are from various portions of the track and not taken simultaneously. Each image is of size 320 x 160 x 3.

![](https://github.com/tvw0006/Behavioral_Cloning/blob/master/right.jpg)

Right Camera Image

![](https://github.com/tvw0006/Behavioral_Cloning/blob/master/left.jpg)

Left Camera Image

![](https://github.com/tvw0006/Behavioral_Cloning/blob/master/center.jpg)

Center Camera Image

The images are imported into a single array, converted to floats, resized and then cropped to size 46 x 200 x 3. The images are then shuffled and 20 percent of the data is used for validation. The images are normalized and preprocessed using an image data generator. A cropped image is shown below.

![](https://github.com/tvw0006/Behavioral_Cloning/blob/master/cropped.png)

Cropped Image




**Architecture** :

Classification of lane lines is more simplistic than traffic signs, therefore a less complex network is needed. However, I wanted to generate a network that might work similarly to a real-world problem.

The model consists of 4 convolutional and 4 fully connected layers. Subsampling is used in lieu of max pooling layers. For each layer, ELU activation is used due to the superior speed and performance. I relied heavily on NVIDIA&#39;s architecture for my model ( [http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). However, due to the nature of the simulation, less convolutional layers were used.

An Adam optimizer is used for compilation with mean squared error loss, due to the model using regression, not classification. The learning rate was tested at several rates, but the initial rate of 0.001 provided satisfactory results. The neural network architecture is shown below.

![](https://github.com/tvw0006/Behavioral_Cloning/blob/master/Architecture.JPG)

**Training** :

The model generates batches of 10000 images for training and 2000 images for validation for 15 epochs. Training loss converges at 0.017 and validation loss converges at around 0.02.  Using these parameters, the car was able to safely navigate both tracks successfully using only images from the first track.

Although the car successfully navigated the track, I will continue to refine the model until the car is able to successfully navigate with less swerving.

**Testing:**

The model is saved and tested on the simulator. Because the images are cropped, the python output program, drive.py has to be modified to also crop the output images.

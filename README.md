# Bird Sound Recognition
A Python app that identifies Indian Bird sounds/songs.
The motivation for building this app came from a personal experience in Covid19 pandemic lockdown. I used to stay a lot in home and used to see and hear a lot of different birds in my suburb, and slowly became fond of them. I had an epiphany and thought of building a sound classifier of bird sounds using machine learning techniques.
## Data Collection
I collected the sounds of 10 individual birds most commonly found in India from a great website https://www.xeno-canto.org/ using a very useful script found at https://github.com/AgaMiko/xeno-canto-download.<br>
<br>
Initially I collected all bird sounds based on the region India, but had to later limit my collection to 10 most common birds as learning was difficult with 997 different classes of birds with a very imbalanced ratio of classes in the dataset.<br>
<br>
## Data Preprocessing
I used the librosa library a lot in my data wrangling and preprocessing steps.<br>
<br>
I performed the following preprocessing steps:<br>
1. Converted all the mp3 files to wav format
2. Removed silent parts from the audio
3. Created chunks of 10 seconds each for audio with large length
4. Computed Mel-Frequency Cepstral Coefficients(MFCC) which are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope to be used with Artificial Neural Network
5. Generated Mel-Spectograms which is a visualization of the frequency spectrum of a signal, where the frequency spectrum of a signal is the frequency range that is contained by the signal, but is then converted to a mel scale. The images were then used as inputs for Convolutional Neural Network.
6. I performed Data Augmentation on wav files using nlpaug library. I performed augmentation by increasing loudness, shifting the signal, and adding white noise to the audio to generate more inputs.<br>
![download](https://user-images.githubusercontent.com/60923559/137144026-89a23b5e-8130-4186-aee4-178cd1fcdb05.png)
## Modelling using Artificial Neural Network
I performed manual stratified sampling of data points to handle the imbalance in class ratio.<br>
* I used the Sequential layer stack from Keras library to build the network.
* 1 Input Layer - 2 Hidden Layers - 1 Output Layer
* ReLU activation function was used for all the layers except for the output layer where softmax function was used to perform multi-class classification
* Dropout regularization was used to prevent overfitting with keep_prob set at 0.5 for every layer.
* Adam optimizer was used with the loss function used as "categorical_crossentropy"<br>
__Unfortunately I couldn't get a really good accuracy using the MFCC features and hence decided to go for Convolutional Neural Network using Mel-Spectogram images as input.__
## Modelling using Convolutional Neural Network
I used transfer learning to train existing VGG-16 architecture for which you can checkout the architecture at https://www.geeksforgeeks.org/vgg-16-cnn-model/<br>
The basic idea is to take advantage of pre-trained models which have learned a lot of features on a big dataset in a Computer Vision application and remove the final fully-connected layers in the pre-trained model and add our own small network with a softmax layer as the output layer to get multi-class classification. This way we get features from the VGG-16 called bottleneck features that we can use for our classifcation task.
<br>
* Split the images with 70-25-5 split in seperate train-valid-test folders with each folder containing folders for each class
* Generate bottleneck features from the pre-trained VGG-16 model and save them. 
* Load the train-valid-test data from the saved bottleneck features
* Define your own network to which you will feed the bottleneck featrues
* I used a similar architecture as before(in ANN) with some changes to Dropout keep_prob values and size of layers.
* Adam optimizer was used with the loss function used as "categorical_crossentropy"<br>
__I was able to achieve 90% accuracy even though I had few inputs for each class (around 600-700), thanks to transfer learning.__

## Future Scope
Currently, I'm working on creating an Android application which will record audio in real-time and predict the bird sound.<br>
I have already taken steps to save the keras model from HDF5 format to TFLite format for use in the app.<br>
__Thanks for reading and stay tuned for further updates.__

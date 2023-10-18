# Dolphin: Because You Can't Stop Tracking Everything (Even Your Shower Time)

![Dolphin](/static/icon.png?w=100&h=100)

In today's world, tracking everything has become the norm. From our steps to our sleep, we want to know it all! So, why not track our showering time too? Introducing Dolphin - a project that uses environmental sound classification to measure how much time you spend under the shower. With Dolphin, you'll never have to guess how long you've been showering again! 🐬🚿

But why? you may ask. Well, why the hell not? :^) As it turns out, experimenting with Convolutional Neural Networks (CNNs) on real-world problems can be a fun and rewarding experience. Plus, it may even help you save water (or not, but that's a bonus!). 

## Data Aquisition

Gathering the audio data for your shower sound classification model can be a bit of a challenge. You see, there is only so much showering you can do yourself, and asking other people to record themselves showering is just plain weird.
Thankfully, there is a big trend around Autonomous Sensory Meridian Response (ASMR) these days. ASMR is a tingling sensation that some people experience in response to certain auditory or visual stimuli, like whispering, tapping, or crinkling sounds. Apparently there is quite a demand for showering sounds - weird but helpful 😁. Now, you might be thinking, "Wait, can I use these ASMR videos to train my shower sound classification model?" Well, you can, but there's a catch. Most of the time, ASMR videos are just 3 minutes long, and they're often looped over and over again. As with any machine learning project, the quality of your data is crucial. The GIGO principle (Garbage In, Garbage Out) always applies, so make sure you're using high-quality, diverse data to train your model.

Another way to add some data to your training set apart from scouring the interwebs for sounds of people showering (which is just a marvelous pastime) you can also use augmentation. The project uses the excellent audiomentations library to apply noise and change the gain of captured samples to double your training data. This should make the model more robust to different recording environments. 

## Model

The project initially attempted to use a normal deep neural network and combine various features offered by the librosa library to classify the sounds. The prediction results for each single extraction were then averaged to reach the final confidence levels for the classes. However, this approach was prone to issues with other sounds overlaying the target sounds, such as music playing in the background, which could significantly affect the accuracy of the predictions.

CNNs are commonly used for image classification tasks, but they can also be applied to sound classification. The key to using CNNs for sound classification is to convert the audio signals into a "visual" representation, such as a spectrogram. A spectrogram is a 2D image that shows the distribution of energy in a sound signal over time and frequency. By converting the audio signals into a spectrogram, we can leverage the power of CNNs to learn features and patterns in the sound data.

Sample training image/Augmentation of said image (Noise & Gain reduction):

![Training Data](/static/110007450.jpg) ![Training Data Augmented](/static/110007450_aug.jpg)

The architecture of the CNN is pretty mundane in that it uses multiple convolutional layers with pooling layers and some dense layers at the end:
![Architecture](/static/architecture.png)

## Results

I trained the model for ~15 epochs and this is those are the results (orange = training, blue = validation):

Epoch Accurancy:

![Training Accurancy](/static/epoch_accuracy.svg)

Epoch Loss:

![Training Loss](/static/epoch_loss.svg)

Validation Accurancy vs iterations:

![Validation Accurancy](/static/evaluation_accuracy_vs_iterations.svg)

Validation Loss vs iterations:

![Validation Loss](/static/evaluation_loss_vs_iterations.svg)

Learning Rate:

![Learning Rate](/static/epoch_learning_rate.svg)

To check the performance of the model, I created a script that calculates the shower time of a given audio file and tried it for the model at each training epoch.

Audio file which has 438 seconds of showering in it:

![Test Performance Showering](/static/test_perf_showering.png)

Audio file which has 0 seconds of showering in it:

![Test Performance No Showering](/static/test_perf_none.png)

The best results are reached after 11 epochs of training, pretty much spot on in regards to the sample with showering and some false positive detection of 3.1% (detected 6 seconds of showering in a 190-second clip of audio without any showering).

Surely there is still some performance to be squeezed from this with some hyperparameter search, but I am content with the results for now.

## Things that I learned

- When building a desktop computer, never undersize your power supply. When training the model, the power consumption of my desktop would sharply increase, leading to a sudden shutdown of the computer.

- Picking up from the previous point: While you can do some trickery with saving your model at every epoch to deal with your computer randomly throwing fits during training, you have to be really careful. When picking up the training with one of the intermediate models, your training/validation split is completely mixed up. This will result in a horrible test set performance since now you are also overfitting on the validation data.

- The python package visualkeras is pretty neat for generating illustrations.

- Using tensorboard for gathering insights in the training process.

- To help the model converge you can use a learning rate scheduler. In this case I used one with exponential decay.

- Model validation accuracy does not necessarily translate to real-world performance. Your model preference also depends on your desired properties (false positive rate, general accuracy).

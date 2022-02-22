# MBTI Predictor
Predicting someone's MBTI type based on their online posts, using AI.

### Approach
I originally attempted to use a simple LSTM, using nltk and Keras. However, this approach led to the model badly overfitting.

I then decided to re-write the program and use a bidirectional LSTM. This was slightly better and, somewhat, prevented overfitting. However, the model trained slowly due to a low learning rate.

I then decided to use a TF BERT Model as a layer within a Keras model. I thought this would improve the performance of the predictor but my GPU (NVIDIA GTX 1650) couldn't run it. I decided to publish this project as a package so that others could attempt to train the BERT model.

### Findings
I could not find a significant correlation between a post and the MBTI type of its poster.
Feel free to download the source code and try for yourself.

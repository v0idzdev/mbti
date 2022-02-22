# MBTI Predictor
Predicting someone's MBTI type based on their online posts, using AI.

### Approach
I originally attempted to use a simple LSTM, using nltk and Keras. However, this approach led to the model badly overfitting.

I opted for a bidirectional LSTM, treating the data differently with techniques such as lemmatization. This approach led to significantly slower learning - the model improved less and less per epoch. However, the validation accuracy did increase per epoch, as opposed to decreasing or remaining static. 

As a last resort, I chose to use the BERT transformer with the AutoTokenizer from 'transformers.' In theory, it would have led to significantly better results - however, the large number of parameters meant it couldn't run on my GTX 1050. 

Feel free to download the source code, if your hardware resources could accommodate training the BERT model. 

### Findings
I could not find a significant correlation between a post and the MBTI type of its poster, using both a standard LSTM and its bidirectional variant. 

The dataset was unbalanced, with the number of posts per MBTI type differing greatly. I decided not to rectify this issue initially because the training data was already small at around 9000 entries. I may revisit this project if I find a more balanced and extensive dataset.

Feel free to download the source code and try for yourself.

## Project Summary: Sentiment Analysis with RNN (Movie Reviews)

This project implements a **Recurrent Neural Network (RNN)** to perform **sentiment analysis** on movie reviews. Using a dataset of text reviews paired with positive/negative labels, the model learns to classify whether a given review expresses a favorable or unfavorable sentiment.

### Key Features

* **Text preprocessing** including tokenization, vocabulary creation, and sequence padding.
* **Neural network architecture** built using an embedding layer, recurrent layers (SimpleRNN/LSTM/GRU depending on implementation), and a final dense classification layer.
* **Training and evaluation** of the model using accuracy and loss metrics.
* **Prediction capability** allowing users to input any custom movie review and receive a sentiment score.

### Files Included

* `Sentiment_RNN.ipynb` — main notebook containing preprocessing steps, model building, training, and prediction.
* `Files` word document which includes links to `reviews.txt` & `labels.txt` — dataset of movie reviews and corresponding sentiment labels.

### How to Use

1. Upload the dataset (`reviews.txt`, `labels.txt`) to your Colab environment.
2. Run the notebook to preprocess data and train the RNN model.
3. Enter your own movie review at the end of the notebook to get the predicted sentiment (positive/negative).

### Learning Outcomes

* Understanding of RNN-based natural language processing (NLP).
* Hands-on experience in training deep learning models for text classification.
* Ability to evaluate model performance and generate predictions on real-world text.

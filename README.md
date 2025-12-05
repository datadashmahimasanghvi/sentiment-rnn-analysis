# Sentiment Analysis using RNN (PyTorch)

This repository contains an end-to-end implementation of sentiment analysis using a Recurrent Neural Network (RNN) built with PyTorch. The notebook demonstrates data preprocessing, tokenization, vocabulary construction, batching, model development, training, and evaluation. It is based on a lecture exercise and serves as an academic and practical example of sequence modeling for sentiment classification.

---

## Project Overview

The project covers the complete workflow for training a sentiment classifier:

* Loading and preprocessing text data
* Tokenization and numericalization
* Vocabulary construction and handling unknown tokens
* Sequence padding and batching
* Building an RNN-based sentiment classifier
* Training loop with loss and accuracy tracking
* Evaluating model performance
* Visualizing insights and predictions

This repository is useful for anyone learning NLP, sequence models, or PyTorch fundamentals.

---

## Repository Structure

```
sentiment-rnn-analysis/
│
├── notebooks/
│   └── Lecture_8_Sentiment_RNN_(Exercise).ipynb   # Main notebook
│
├── requirements.txt                                # Python dependencies
│
└── README.md                                        # Project documentation
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

The project uses:

* pandas
* numpy
* torch
* torchtext
* scikit-learn
* matplotlib
* tqdm

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/sentiment-rnn-analysis.git
cd sentiment-rnn-analysis
```

2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebooks/Lecture_8_Sentiment_RNN_(Exercise).ipynb
```

Alternatively, upload the notebook to Google Colab and run it there.

---

## Model Architecture

The sentiment classifier consists of:

* An Embedding Layer
* A Recurrent Neural Network (RNN or LSTM depending on the exercise variant)
* A fully connected output layer
* CrossEntropyLoss for classification
* Adam optimizer for training

The RNN processes sequences of token embeddings and outputs a prediction for the sentiment class.

---

## Example Output

The notebook includes:

* Training accuracy and loss per epoch
* Validation accuracy
* Example prediction results using trained model

---

## Future Enhancements

Potential improvements include:

* Switching to LSTM or GRU for better sequence modeling
* Adding attention mechanisms
* Hyperparameter tuning
* Expanding dataset and improving preprocessing
* Exporting the trained model for deployment
* Creating a Python script version of the training pipeline

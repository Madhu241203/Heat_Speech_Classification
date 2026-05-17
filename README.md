#  Tweet Classification using RNN (Handling Imbalanced Data)

## Problem Statement

The goal of this project is to build a deep learning model to classify tweets into two categories (e.g., positive/negative or relevant/irrelevant).
The dataset is **highly imbalanced**, making it challenging to correctly predict the minority class.
The objective is to develop a model that not only achieves good accuracy but also performs well on the minority class using appropriate techniques.

---

##  Data Preprocessing Steps

The following preprocessing steps were applied to clean and prepare the text data:

* Converted all text to **lowercase**
* Removed:

  * **@mentions**
  * **URLs**
  * **Special characters and numbers**
* Removed **extra spaces**
* Tokenized tweets into individual words
* Removed empty or invalid entries after cleaning

---

##  Vocabulary Creation Approach

* Tokenized tweets were used to build a **word frequency dictionary**
* Selected **top 5000 most frequent words** to limit vocabulary size
* Created a **word-to-index mapping**
* Added special tokens:

  * `<PAD>` = 0 (for padding sequences)
  * `<OOV>` = 1 (for unknown words)
* Converted each tweet into a sequence of integers based on this mapping
* Applied **padding** to ensure uniform sequence length

---

##  Model Architecture

A deep learning model was built using TensorFlow/Keras with the following layers:

* **Embedding Layer**

  * Converts word indices into dense vector representations
* **LSTM Layer**

  * Captures sequential dependencies in text data
* **Dropout Layer (0.5)**

  * Reduces overfitting
* **Dense Output Layer**

  * Sigmoid activation for binary classification

---

##  Training Details

* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Epochs:** 10
* **Batch Size:** 32
* **Train-Test Split:** 80/20 (Stratified Sampling)
* **Class Imbalance Handling:**

  * Used **class weights** to give higher importance to minority class

---

##  Final Results

* **Test Accuracy:** ~88%
* **F1-Score (Minority Class):** ~0.41

###  Observations:

* Model performs well on the **majority class**
* Performance on **minority class is moderate**
* Indicates need for further improvements such as:

  * Threshold tuning
  * Advanced models (BiLSTM, GRU)
  * Data balancing techniques (SMOTE)

---

##  Conclusion

This project demonstrates an end-to-end NLP pipeline:

* Text preprocessing
* Vocabulary building
* Sequence modeling using RNN/LSTM
* Handling imbalanced datasets

It highlights the importance of evaluating models using **F1-score and recall**, rather than relying solely on accuracy.

---

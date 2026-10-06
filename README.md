# Sentiment-Analysis-using-NLP

This project focuses on **Sentiment Analysis using Natural Language Processing (NLP)**.

The goal is to train machine learning models to predict the sentiment of tweets posted about US Airlines. Each tweet is classified into one of three sentiment categories:

* 😊 Positive
* 😐 Neutral
* 😞 Negative

The project was completed using **Python and Google Colab**.

---

## 📊 Dataset

The dataset used for this project is the **Twitter US Airline Sentiment Dataset** from Kaggle.

**Dataset:** Twitter US Airline Sentiment
**Source:** Kaggle
**Link:** https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

The dataset contains tweets about different US airlines along with their corresponding sentiment labels.

For this project, only the following columns were used:

* `airline_sentiment` – Sentiment label
* `text` – Original tweet

---

## 🛠️ Technologies and Libraries

* Python
* Google Colab
* Pandas
* NLTK
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Random Forest

---

## 🔄 Project Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Selection
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Train-Test Split
   ↓
Machine Learning Models
   ↓
Sentiment Prediction
   ↓
Accuracy Evaluation
```

---

## 🧹 Text Preprocessing

The tweets were cleaned using several NLP preprocessing techniques.

### 1. Convert to Lowercase

All text is converted to lowercase.

```text
"I LOVE this airline"
```

becomes:

```text
"i love this airline"
```

### 2. Remove URLs

URLs contained in tweets are removed.

### 3. Tokenization

Tweets are separated into individual words using NLTK.

### 4. Stopword Removal

Common English stopwords are removed.

Examples:

```text
the
is
a
an
and
to
```

### 5. Stemming

The **Porter Stemmer** is used to reduce words to their stem.

For example:

```text
amazing → amaz
flying → fli
```

---

## 🔢 Feature Extraction

Machine learning models cannot directly process raw text, so the cleaned tweets are converted into numerical features.

### TF-IDF

`TfidfVectorizer` from Scikit-learn is used for feature extraction.

The maximum number of features is set to **3000**.

```python
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["text_cleaned"]).toarray()
Y = df["airline_sentiment"].values
```

Where:

* `X` = TF-IDF feature matrix
* `Y` = Sentiment labels

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using `train_test_split()`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)
```

### Parameters

* **Test size:** 20%
* **Random state:** 2
* **Training data:** 80%
* **Testing data:** 20%

---

## 🤖 Machine Learning Models

Two machine learning algorithms were trained and evaluated.

### 1. Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

model_nb = MultinomialNB()

model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)
```

The model accuracy was calculated using:

```python
accuracy_score(y_test, y_pred_nb)
```

---

### 2. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
```

The accuracy was calculated using:

```python
accuracy_score(y_test, y_pred_rf)
```

---

## 📈 Model Evaluation

The performance of both models was evaluated using **accuracy**.

```python
print("Multinomial Naive Bayes Accuracy:", accuracy_nb)
print("Random Forest Accuracy:", accuracy_rf)
```

### Results

| Model                   |        Accuracy |
| ----------------------- | --------------: |
| Multinomial Naive Bayes | 0.7219945355191257 |
| Random Forest           | 0.7482923497267759 |

---

## 💡 Example Prediction

After training the model, a new tweet can be processed and classified.

Example:

```text
"I had an amazing flight and the staff were very friendly!"
```

The model may predict:

```text
Positive
```

---

## 📁 Project Structure

```text
Lesson-6-Sentiment-Analysis/
│
├── Tweets.csv
├── Sentiment_Analysis.ipynb
└── README.md
```

---

## 🎯 Learning Outcomes

Through this task, I learned how to:

* Work with a real-world text dataset
* Perform basic NLP preprocessing
* Remove stopwords and URLs
* Apply stemming using Porter Stemmer
* Convert text into numerical features using TF-IDF
* Split data into training and testing sets
* Train a Multinomial Naive Bayes classifier
* Train a Random Forest classifier
* Evaluate machine learning models using accuracy
* Perform basic sentiment classification

---

## 🚀 Conclusion

- This project demonstrates a basic **NLP sentiment analysis pipeline** using tweets about US Airlines.

- By combining **text preprocessing, TF-IDF feature extraction, and machine learning**, the system can classify tweets as **positive, negative, or neutral**.

- The performance of **Multinomial Naive Bayes** and **Random Forest** can then be compared using their test-set accuracy.


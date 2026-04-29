'''
This is Levi Stoner's source code for the DATA539 class competition. 

The task is to write a program that can determine whether or not a piece of text is a movie/TV show review, and then
whether or not each review is positive or negative. This means that there will be data that is not a movie or tv review, 
and this must be flagged and not accounted for. If the data is identified as a movie/tv review, then we must identify the rating
of that review. This leaves us to create the following classification groups:

    1) Not a movie/tv review
    2) Positive movie/tv review
    3) Negative movie/tv review


The training data is structured into 3 columns: ID, Text, Label.

    1) ID column represents that review's specific ID/classification number. Every single review should have a unique ID, even for
    non-movie/tv reviews.
    2) Text column represents the actual review quote.
    3) Label column represents whether the review is positive/negative(represented by 2 and 1 respectively) or a 
    non-movie/tv review (represented by 0)

Essentially, we are parsing through each review and trying to identify key-words/trends that can predict a review score, as the test 
data is structured into 2 columns: ID, Text.
'''


'''
The first code chunk is dedicated to bringing the training data into the environment. To do this, we implemented a pandas
approach in order to read the CSV file and create a dataframe. We want to use the quoting option here because this allows us
to properly record each review without confusion in regards to quotation marks. Finally, we check whether the data was correctly 
brought in by looking at the identification numbers at the beginning and end of the training file.

Note that bringing the data in via pandas in the following way requires that the training data is stored in the same folder as the 
code file.
'''

import pandas as pd

# Create the dataframe.
df = pd.read_csv("train.csv",  dtype={"ID": str}, quoting=1)

# Check whether identification numbers match training file.
print(df.head())
print(df.tail())


'''
This code chunk is dedicated to getting familier with the dataset. For example, we check how many values in the training set have a label
of 0,1, or 2. Next, we check for any missing values in the dataset, and fill them in if they exist. Finally, we create training variables, x and y, 
using the column data of"TEXT" and "LABEL".
'''

# Dataset check.
print(df.info())
print(df["LABEL"].value_counts())

# Check for any missing values in the dataset and replace missing text values if they exist.
print(df.isnull().sum())
df["TEXT"] = df["TEXT"].fillna("")
df["ID"] = df["ID"].fillna("")
df["LABEL"] = df["LABEL"].fillna("")

# Separate into training variables.
X = df["TEXT"]
y = df["LABEL"]


'''
This code chunk is dedicated to spliting the training data into a training set and validation set. This will allow us to test the
model on examples it has not seen yet, to verify whether the model is prediciting well.
'''

from sklearn.model_selection import train_test_split

# Create training dataset and validation dataset. The parameter "test_size" refers to how much of the set will be trained vs how 
# much will be tested, so the value of "0.2" refers to 80% of the data being trained on, and 20% of the data being tested on. The 
# parameter "random_state" allows reproducability, as it will give the same split each time. The parameter "stratify" refers to the 
# proportion of each label that will be used, as an attempt to keep consistency. We want to use this parameter with respect to y, as 
# this is the variable being predicted.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Length of training and validation sets.
print("Training size:", len(X_train))
print("Validation size:", len(X_val))


'''
This code chunk is dedicated to building a classification model. To accomplish this, we use scikit‑lear packages 
that translates reviews into Term Frequency-Inverse Document frequency (TF-IDF) features and then fits a Logistic Regression 
classifier. 
'''


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Build a pipeline that converts reviews into TF‑IDF features. It then trains a Logistic Regression classifier with 
# a high iteration limit and class balancing to deal with label imbalance.

model = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=3000,class_weight="balanced"))])

# Fit model on training split
model.fit(X_train, y_train)

# Create a prediction for classification
y_pred = model.predict(X_val)

# Check model performance
print("Validation accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


'''
The final code chunk loads in the test data, uses the model from the last chunk to predict labels,and then creates a submission file with the 
ID and predicted label columns.
'''

# Load in the test data. Remember that the "test.csv" file must be in the same folder as this code file.
test_df = pd.read_csv("test.csv", dtype={"ID": str}, quoting=1)

# Check whether identification numbers match training file.
print(test_df.head())
print(test_df.tail())

# Replace missing text values if any exist.
test_df["TEXT"] = test_df["TEXT"].fillna("")

# Predict labels for the test text.
test_predictions = model.predict(test_df["TEXT"])

# Create dataframe for submission.
submission = pd.DataFrame({"ID": test_df["ID"], "LABEL": test_predictions})

# Create a submission CSV file.
submission.to_csv("submission.csv", index=False)


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

df = pd.read_csv('df_train.csv')

df.head()


# In[21]:


# df Preprocessing

# Fill missing values
df['following'] = df['following'].fillna(0)
df['followers'] = df['followers'].fillna(0)
df['actions'] = df['actions'].fillna(0)
df['is_retweet'] = df['is_retweet'].fillna(0)

df['tweet_length'] = df['Tweet'].apply(len)  # Length of the tweet
df['hashtag_count'] = df['Tweet'].apply(lambda x: x.count('#'))  # Number of hashtags
df['mention_count'] = df['Tweet'].apply(lambda x: x.count('@'))  # Number of mentions
df['url_count'] = df['Tweet'].apply(lambda x: x.count('http'))  # Number of URLs
df['capitalized_count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))  # Capitalized words
df['exclamation_count'] = df['Tweet'].apply(lambda x: x.count('!'))  # Exclamation symbols
df['question_mark_count'] = df['Tweet'].apply(lambda x: x.count('?'))  # Question marks


# Encode the target variable (Type)
df['Type'] = df['Type'].map({'Spam': 1, 'Quality': 0})
print(df.head())


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
# Convert the 'Tweet' col to string and handle non-string entries
df['Tweet'] = df['Tweet'].astype(str)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

df['cleaned_tweet'] = df['Tweet'].astype(str).apply(clean_text)
vectorizer = TfidfVectorizer(max_features=500)  
X_text_feature = vectorizer.fit_transform(df['cleaned_tweet']).toarray()  # Convert to array

# Re-extract features now that all entries are strings
df['tweet_length'] = df['Tweet'].apply(len)  # Length of the tweet
df['hashtag_count'] = df['Tweet'].apply(lambda x: x.count('#'))  
df['mention_count'] = df['Tweet'].apply(lambda x: x.count('@'))  
df['url_count'] = df['Tweet'].apply(lambda x: x.count('http'))  
df['capitalized_count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper())) 
df['exclamation_count'] = df['Tweet'].apply(lambda x: x.count('!'))
df['question_mark_count'] = df['Tweet'].apply(lambda x: x.count('?'))  

print(df.head(9))
X_meta_features = df[['following', 'followers', 'actions', 'is_retweet', 'tweet_length', 
                      'hashtag_count', 'mention_count', 'url_count', 
                      'capitalized_count', 'exclamation_count', 'question_mark_count']]

# Step 3: Combine both sets of features (NLP + Metaheuristic)
import numpy as np
X_combined = np.hstack((X_text_feature, X_meta_features.values))
y = df['Type']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# Display the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:# Whale Optimization Algorithm for feature selection
class WhaleOptimizationAlgorithm:
    def __init__(self, X, y, num_whales=30, max_iter=3):
        self.X = X
        self.y = y
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.best_features = None
        self.best_score = 0

    def fitness(self, features):
        # Train a classifier and return the accuracy
        if np.sum(features) == 0:
            return 0  # Avoid empty feature sets
        X_subset = self.X[:, features.astype(bool)]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, self.y, test_size=0.2, random_state=42)
        model = AdaBoostClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def optimize(self):
    # Initialize positions of whales
      positions = np.random.rand(self.num_whales, self.X.shape[1])
      for iteration in range(self.max_iter):
        print(f"Iteration {iteration + 1}/{self.max_iter}")
        for i in range(self.num_whales):
            # Calculate fitness
            score = self.fitness(positions[i])
            if score > self.best_score:
                self.best_score = score
                self.best_features = positions[i]
        # Update positions (simplified version of WOA)
        positions += np.random.rand(self.num_whales, self.X.shape[1]) * 0.1
      return self.best_features

import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Run the Whale Optimization Algorithm
woa = WhaleOptimizationAlgorithm(X_train, y_train)
best_features = woa.optimize()

# Train the AdaBoost classifier on the selected features
X_train_selected = X_train[:, best_features.astype(bool)]
X_test_selected = X_test[:, best_features.astype(bool)]
model = AdaBoostClassifier(algorithm='SAMME')  # Explicitly using SAMME
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
print("Accuracy:", accuracy_score(y_test, y_pred))
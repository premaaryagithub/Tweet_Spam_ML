This project focuses on detecting spam in tweets using a combination of Natural Language Processing (NLP), metaheuristic feature selection via the Whale Optimization Algorithm (WOA), and a machine learning classifier (AdaBoost). The system preprocesses tweet data, extracts relevant features, and uses WOA to optimize feature selection for improved spam detection accuracy.


**Overview**
The project involves the following steps:

**Data Preprocessing: Cleaning and preparing tweet data for analysis.
Feature Extraction: Extracting text-based features (e.g., tweet length, hashtags, mentions) and using TF-IDF for NLP features.
Feature Selection: Applying the Whale Optimization Algorithm (WOA) to select the most relevant features.
Model Training: Using AdaBoost to train a spam detection model on the selected features.
Evaluation: Measuring the accuracy of the model on a test dataset.**

Features::

**Text Preprocessing: Cleans tweets by removing URLs, mentions, hashtags, and stopwords.
Feature Extraction: Extracts meta-features such as tweet length, hashtag count, mention count, etc.
Whale Optimization Algorithm (WOA): Optimizes feature selection to improve model performance.
AdaBoost Classifier: A robust machine learning model for spam detection.
Accuracy Evaluation: Evaluates the model's performance using accuracy metrics.**

Requirements: 

**Python Libraries**
Install the required libraries using the following command:

**pip install pandas numpy scikit-learn nltk**

Dataset:
The dataset (df_train.csv) should contain the following columns:

**Tweet: The text of the tweet.
Type: The label indicating whether the tweet is "Spam" or "Quality".
Additional metadata columns: following, followers, actions, is_retweet.**
Workflow : 

Data Preprocessing:

**Fill missing values in the dataset.
Extract meta-features such as tweet length, hashtag count, mention count, etc.
Clean tweet text by removing URLs, mentions, hashtags, and stopwords.**

Feature Extraction:

**Use TF-IDF to convert cleaned tweet text into numerical features.
Combine TF-IDF features with meta-features.**

Feature Selection:

**Apply the Whale Optimization Algorithm (WOA) to select the most relevant features.**

Model Training:

**Train an AdaBoost classifier on the selected features.**

Evaluation:

**Evaluate the model's accuracy on a test dataset.**

**Code Structure**
The code is structured as follows:

**Data Loading and Preprocessing:
Load the dataset (df_train.csv).
Fill missing values and extract meta-features.
Clean tweet text and apply TF-IDF vectorization.**

Feature Selection:

**Implement the Whale Optimization Algorithm (WOA) to optimize feature selection.
Model Training and Evaluation:
Train an AdaBoost classifier on the selected features.
Evaluate the model's accuracy on a test dataset.**

Results
**The model achieves an accuracy of X% on the test dataset. The Whale Optimization Algorithm (WOA) successfully identifies the most relevant features, improving the model's performance.**

Customization
**Dataset: Replace df_train.csv with your own dataset.
Feature Extraction: Modify the meta-feature extraction logic to suit your needs.
WOA Parameters: Adjust the number of whales (num_whales) and iterations (max_iter) in the WOA.
Classifier: Experiment with other classifiers (e.g., Random Forest, SVM) instead of AdaBoost.**

License
This project is open-source and available under the MIT License. Feel free to modify and distribute it as needed.

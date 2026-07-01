# Tweet Spam Detection with Metaheuristic Feature Selection

Spam classification for tweets, combining NLP feature engineering with a self-implemented Whale Optimization Algorithm (WOA) for feature selection, feeding an AdaBoost classifier.

## Approach

1. **Meta-feature engineering** — tweet length, hashtag count, mention count, URL count, capitalized-character count, exclamation/question mark counts, plus account-level features (`following`, `followers`, `actions`, `is_retweet`).
2. **Text preprocessing** — removed URLs, mentions, hashtags, and non-alphabetic characters; lowercased; removed stopwords (NLTK).
3. **TF-IDF vectorization** — cleaned tweet text converted to 500 TF-IDF features, combined with the 11 meta-features into a single 511-dimension feature matrix.
4. **Feature selection — Whale Optimization Algorithm**: a custom, simplified implementation of WOA (a nature-inspired metaheuristic optimizer) that searches for the best-performing feature subset, using AdaBoost validation accuracy as the fitness function.
5. **Final model** — AdaBoost classifier trained on the WOA-selected feature subset.

## Results (actual run output, 11,968 tweets: 6,153 Quality / 5,815 Spam)

| Model | Features Used | Accuracy |
|---|---|---|
| Baseline AdaBoost | All 511 features | 99.79% |
| WOA-selected AdaBoost | 259 / 511 features (49% fewer) | 99.79% |

**The honest headline result here is not the accuracy number** — it's that WOA cut feature dimensionality roughly in half **with zero accuracy loss**, which is the actual point of a feature-selection algorithm (a smaller, cheaper, less overfit-prone model at equal performance).

## Important caveat — read before quoting this on a resume or in an interview

99.79% accuracy is high enough to be a red flag if stated without context, and an interviewer will likely probe it. The reason: this dataset's `actions` meta-feature is a near-perfect proxy for the label — bot/spam accounts in this dataset average ~11,500 actions vs. ~99 for genuine accounts, which makes the classification problem close to trivially separable once that single feature is included. This isn't a flaw in the code; it's a property of this particular dataset.

**How to talk about this correctly in an interview:** lead with the feature-selection result (49% dimensionality reduction, no accuracy loss), and proactively mention that you identified `actions` as a high-leakage feature — that's a stronger signal of ML maturity than the accuracy number itself. If you want a more realistic difficulty benchmark, re-run with `actions` excluded and report that number too.

## Tech Stack

Python, pandas, NumPy, scikit-learn (AdaBoostClassifier, TfidfVectorizer, train_test_split), NLTK

## How to Run

```bash
pip install pandas numpy scikit-learn nltk
python -c "import nltk; nltk.download('stopwords')"
python tweet_spam_ml.py
```

The dataset (`data_train.csv`) is already committed in this repo.

## Possible Extensions

- Re-run with `actions` excluded to get a more realistic (harder) accuracy benchmark.
- Compare WOA-selected features against a simpler baseline like `SelectKBest` to quantify what the metaheuristic actually buys you over a standard method.
- Visualize which specific features WOA consistently selects across multiple runs (currently non-deterministic since positions are randomly initialized).

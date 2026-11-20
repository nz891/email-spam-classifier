# Email Spam Classifier
### Comparing classical NLP pipelines for spam detection

A binary text classifier that labels emails as spam (1) or not spam (0), comparing how
different feature representations and models affect performance.

## Pipeline
1. **Preprocessing:** contraction expansion, lowercasing, punctuation and number removal,
   stopword filtering, stemming and lemmatization (NLTK)
2. **Features:** Bag of Words and TF-IDF (30,770-term vocabulary)
3. **Models:** Multinomial Naive Bayes (BoW), Logistic Regression (TF-IDF), SVM (TF-IDF)
4. **Evaluation:** 80/20 split, classification reports, confusion matrices, ROC and
   precision-recall curves

## Results
| Model | Accuracy | Spam precision | Spam recall |
|---|---|---|---|
| Naive Bayes + BoW | 97.1% | 0.95 | 0.95 |
| Logistic Regression + TF-IDF | 98.2% | 0.95 | 0.99 |
| **SVM + TF-IDF** | **98.3%** | **0.96** | **0.99** |

TF-IDF beat raw word counts: weighting words by how distinctive they are gave the models
a clearer signal than frequency alone.

## Failure analysis
Logistic Regression misclassified 13 of 724 validation emails. Most errors were legitimate
business emails with financial or transactional wording ("deal", "price", "billing") that
overlaps with spam vocabulary. TF-IDF picks up keywords but not tone or intent.

## Dataset
3,619 labelled emails (Enron-style corporate mail plus spam).

## Tools
Python, pandas, scikit-learn, NLTK, contractions, matplotlib, seaborn (Google Colab)

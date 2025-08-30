# Review Quality Assessment System

ML-based system to detect spam, advertisements, irrelevant content, and reviews from users who never visited locations. Implements dual-approach architecture combining rule-based detection with transformer models.

## Features

- **Policy Enforcement**: Automatic detection of promotional content, irrelevant reviews, and unvisited location rants
- **Dual Classification**: Two complementary approaches for robust violation detection
- **High Accuracy**: 95%+ performance across violation categories
- **Explainable Results**: Justifications provided for each classification decision

## Approaches

### Approach 1: Hybrid LLM + Rule-Based
- DistilBERT sentiment analysis + keyword pattern matching
- Real-time inference on 1,995+ reviews
- Performance: 91.9% promotional, 98.2% irrelevant, 98.7% not visited

### Approach 2: BERT Embeddings + Ensemble
- 768-dimensional DistilBERT embeddings + Random Forest classifier
- Multi-class classification with comprehensive preprocessing
- Performance: 95.0% overall accuracy, F1-scores 0.919-0.976 per class
- Model persistence: 2 joblib files + 1 pkl file for production deployment

## Installation

```bash
pip install transformers torch scikit-learn pandas numpy nltk matplotlib seaborn wordcloud tqdm
```

## Quick Start

### Approach 1 (Hybrid)
```python
# Load and run classification
reviews_classified = add_classification_columns(reviews)
print(reviews_classified[['reviewContent', 'ispromotional', 'isirrelevant', 'notvisited', 'justification']].head())
```

### Approach 2 (BERT + RF)
```python
# Generate BERT embeddings and classify
embeddings = generate_bert_embeddings(reviews['text'])
predictions = classifier.predict(embeddings)
```

## Data

- **Training**: 1,995 balanced Yelp reviews
- **Validation**: 675 manually labeled reviews
- **Categories**: Clean, Promotional, Irrelevant, Not Visited

## Model Performance

| Approach | Overall | Promotional | Irrelevant | Not Visited |
|----------|---------|-------------|------------|-------------|
| Hybrid   | -       | 91.9%       | 98.2%      | 98.7%       |
| BERT+RF  | 95.0%   | F1: 0.974   | F1: 0.919  | F1: 0.930   |

## Repository Structure

```
├── Techtokers.ipynb          # Approach 1: Hybrid LLM implementation
├── classifier.ipynb          # Approach 2: BERT + Random Forest
├── data/
│   ├── balancedYelpReviews.csv
│   └── reviews_labelled.csv
└── README.md
```

## Usage Examples

### Detect Policy Violations
```python
review = "Great pizza! Visit www.pizzapromo.com for 50% off!"
result = classify_review_with_bert(review)
# Output: {"ispromotional": 1, "justification": "Contains promotional content or links"}
```

### Batch Processing
```python
classified_reviews = add_classification_columns(review_dataframe)
violations = classified_reviews[
    (classified_reviews['ispromotional'] == 1) |
    (classified_reviews['isirrelevant'] == 1) |
    (classified_reviews['notvisited'] == 1)
]
```

## Technical Details

**Models Used**:
- DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Random Forest with class balancing
- Rule-based pattern matching

**Key Libraries**:
- transformers, torch (BERT models)
- scikit-learn (ML algorithms)
- nltk, pandas (preprocessing)

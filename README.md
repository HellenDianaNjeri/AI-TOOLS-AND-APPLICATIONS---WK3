# AI Fundamentals Project

This repository contains three core machine learning and deep learning tasks along with an ethics and optimization review. Each task demonstrates a specific area of AI development using tools like Scikit-learn, TensorFlow, PyTorch, and spaCy.

---

## Project Structure

| File | Description |
|------|-------------|
| `iris_decision_tree.ipynb` | Classical ML: Decision tree classifier on the Iris dataset using Scikit-learn |
| `mnist_cnn_classification.ipynb` | Deep Learning: CNN for digit classification on MNIST using TensorFlow |
| `spacy_ner_sentiment.ipynb` | NLP: Named Entity Recognition and rule-based sentiment analysis using spaCy |
| `ethics_and_debugging.ipynb` | Part 3: Bias discussion and TensorFlow model debugging |

---

## Task Overview

### Task 1: Classical Machine Learning with Scikit-learn
- Dataset: Iris flower dataset
- Model: DecisionTreeClassifier
- Evaluation: Accuracy, Precision, Recall
- Tools: `scikit-learn`, `pandas`, `matplotlib`

### Task 2: Deep Learning with TensorFlow
- Dataset: MNIST handwritten digits
- Model: Convolutional Neural Network (CNN)
- Target: ≥95% test accuracy
- Output: Visual predictions on sample images

### Task 3: Natural Language Processing with spaCy
- Dataset: Amazon-style product reviews
- Task: Named Entity Recognition (NER)
- Sentiment: Rule-based positive/negative classification
- Output: Extracted entities + sentiment labels

### Part 3: Ethics & Optimization
- Bias analysis in MNIST and review datasets
- Mitigation strategies using TensorFlow Fairness Indicators and spaCy
- Debugged faulty TensorFlow model with fixed architecture and loss function

---

##  Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow spacy
python -m spacy download en_core_web_sm

# Iris Classification + LIME (Tabular Data)

A Random Forest classifier trained on the Iris dataset, explained per-prediction using LIME's tabular explainer. Third project in this series, after applying the same explainability approach to text (project 01) and images (project 02) — this one covers structured/tabular data, the third major input type LIME supports.

## Problem

A classifier can hit high accuracy and still be right for the wrong reasons — relying on a feature that happens to correlate with the label in this dataset but wouldn't generalize. Global accuracy alone doesn't surface that. This project checks, at the level of a single prediction, which specific feature values actually pushed the model toward its answer.

## Approach

1. Load the Iris dataset (4 features: sepal length, sepal width, petal length, petal width; 3 target classes: setosa, versicolor, virginica).
2. Split 80/20 train/test.
3. Train a Random Forest (`n_estimators=500`) on the training set.
4. Evaluate on the held-out test set.
5. Pick a single test instance, and use `LimeTabularExplainer` to explain that one prediction — which feature values contributed, and by how much, toward the predicted class.

## Results

- **Test accuracy: 0.97** (29/30 correct on the held-out set)
- **Confusion matrix:**
  ```
  [[11  0  0]
   [ 0 12  1]
   [ 0  0  6]]
  ```
  One misclassification, between the two visually closer classes (versicolor/virginica) — consistent with how these two overlap in petal measurements in the actual dataset.

## What LIME adds here

For a single test instance, LIME doesn't just confirm the predicted class — it shows which of the 4 features, and which value ranges, drove that specific prediction, with a per-feature weight. This is the difference between "the model is 97% accurate" and "here's why the model made *this* call, on *this* input" — the second is what you'd actually need if you were debugging a misclassification or explaining a prediction to someone who isn't going to trust a black box.

## Tech stack

Python, scikit-learn (RandomForestClassifier), LIME (`lime.lime_tabular`), pandas, NumPy

## Notes on scope

This is a deliberately small, classic dataset (150 rows, 4 features) — chosen to isolate and understand the tabular-explainer mechanics cleanly, not as a demonstration of handling real-world scale or messy data. The same `LimeTabularExplainer` pattern applies directly to larger, messier tabular datasets; this project is the minimal version of that pattern.

## Credits

- [LIME](https://github.com/marcotcr/lime) — Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier.
- Iris dataset — via `sklearn.datasets.load_iris`.

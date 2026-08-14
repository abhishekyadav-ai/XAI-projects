# XAI-projects

Hands-on projects exploring explainable AI (XAI) techniques — understanding *why* machine learning models make the predictions they do, not just how accurate they are.

This repo is where I document my journey learning model interpretability, one technique and one dataset at a time. Each project applies a specific XAI method to a real-world problem, walks through the reasoning behind the model's predictions, and honestly reports what worked, what didn't, and what I learned.

## Why this repo exists

Most of my other projects focus on building and training models. This one flips the question: once a model works, how do you trust it? Explainability matters because a model that's accurate but opaque is hard to debug, hard to defend, and easy to trust for the wrong reasons. These projects are my attempt to open that black box, technique by technique.

## Projects

| # | Project | Technique | Dataset |
|---|---------|-----------|---------|
| 01 | [Quora Insincere Questions](./01-Quora-Insincere-Questions-LIME) | LIME | [Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification) |
| 02 | [InceptionV3 Image Classification](./02-InceptionV3-Image-Classification-LIME) | LIME | ImageNet (via pretrained InceptionV3) |
| 03 | [Iris Classification](./03-Iris-Classification-LIME) | LIME | [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (via `sklearn.datasets`) |

More projects will be added here as I explore other techniques (SHAP, permutation importance, etc.).

## Structure

Each project lives in its own numbered folder and includes:
- A Jupyter notebook with the full implementation
- A project-level README explaining the approach, results, and key takeaways
- A `requirements.txt` for reproducibility
- Instructions for obtaining any dataset used (raw data isn't committed to the repo)

## Tech stack

Python, scikit-learn, pandas, LIME, matplotlib/seaborn — expanding as new projects are added.

## About

I'm a CS student learning ML/DL by building projects and documenting the process publicly. Feedback and suggestions are welcome — feel free to open an issue.

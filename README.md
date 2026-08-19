# XAI-projects

Hands-on projects exploring explainable AI (XAI) techniques — understanding *why* machine learning models make the predictions they do, not just how accurate they are.

This repo is where I document my journey learning model interpretability, one technique and one dataset at a time. Each project applies a specific XAI method to a real-world problem, walks through the reasoning behind the model's predictions, and honestly reports what worked, what didn't, and what I learned.

## Why this repo exists

Most of my other projects focus on building and training models. This one flips the question: once a model works, how do you trust it? Explainability matters because a model that's accurate but opaque is hard to debug, hard to defend, and easy to trust for the wrong reasons. These projects are my attempt to open that black box, technique by technique.

## Projects

| # | Project | Model | XAI Technique | Data Type | Dataset |
|---|---------|-------|----------------|-----------|---------|
| 01 | [Quora Insincere Questions](./01-Quora-Insincere-Questions-LIME) | TF-IDF + Logistic Regression | LIME | Text | [Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification) |
| 02 | [InceptionV3 Image Classification](./02-InceptionV3-Image-Classification-LIME) | InceptionV3 (pretrained) | LIME | Image | ImageNet |
| 03 | [Iris Classification](./03-Iris-Classification-LIME) | Random Forest | LIME | Tabular | [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (via `sklearn.datasets`) |
| 04 | [IMDB Sentiment Classification](./04-IMDB-Sentiment-SHAP) | DistilBERT (pretrained sentiment pipeline) | SHAP | Text | [IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) |
| 05 | [ImageNet Classification](./05-ImageNet-Classification-SHAP) | ResNet50 (pretrained) | SHAP | Image | ImageNet (via `shap.datasets.imagenet50`) |
| 06 | [Adult Income Classification](./06-Adult-Income-SHAP) | XGBoost | SHAP | Tabular | [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) (via `shap.datasets.adult`) |
| 07 | [Stroke Prediction Counterfactuals](./07-Stroke-Prediction-Counterfactuals-DiCE) | Random Forest | Counterfactuals (DiCE) | Tabular | [Healthcare Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |

**A quick way to read this table:** each row pairs one *data type* (text, image, or tabular) with one *XAI technique* (LIME or SHAP). Projects 01–03 use LIME, 04–06 use SHAP — same three data types, covered twice, once per technique. That pairing is intentional: it makes it easy to compare how the same method behaves across very different kinds of data, and how the two techniques differ from each other on the same kind of data.

## What's the difference between LIME and SHAP?

Both answer the same question — "why did the model predict this?" — but they get there differently:

- **LIME** explains one prediction at a time by building a small, simple model *around* that specific prediction and seeing what it relies on. It's fast and intuitive, but two runs on the same input can give slightly different explanations, and it doesn't guarantee mathematically consistent results across predictions.
- **SHAP** is grounded in game theory (Shapley values) and calculates each feature's *fair contribution* to a prediction, relative to a baseline. It's more computationally expensive but gives consistent, theoretically-backed explanations, and scales naturally from a single prediction up to whole-dataset patterns (which LIME isn't really built for).

## Repository structure

```
XAI-projects/
├── README.md                                    (this file)
│
├── 01-Quora-Insincere-Questions-LIME/
│   ├── images/
│   ├── README.md
│   ├── quora_insincere_lime.ipynb
│   └── requirements.txt
│
├── 02-InceptionV3-Image-Classification-LIME/
│   ├── images/
│   ├── README.md
│   ├── app.py                                   (Gradio demo app)
│   ├── inceptionv3_image_classification.ipynb
│   └── requirements.txt
│
├── 03-Iris-Classification-LIME/
│   ├── README.md
│   └── iris_lime_tabular.ipynb
│
├── 04-IMDB-Sentiment-SHAP/
│   ├── images/
│   ├── README.md
│   ├── imdb_sentiment_shap.ipynb
│   └── shap_text_plot.png
│
├── 05-ImageNet-Classification-SHAP/
│   ├── README.md
│   ├── imagenet_shap.ipynb
│   ├── shap_image_plot_blur.png
│   └── shap_image_plot_inpaint.png
│
└── 06-Adult-Income-SHAP/
    ├── README.md
    ├── adult_income_shap.ipynb
    ├── shap_dependence_plot.png
    ├── shap_force_single.png
    ├── shap_force_stacked.png
    ├── shap_summary_bar.png
    └── shap_summary_beeswarm.png
│
└── 07-Stroke-Prediction-Counterfactuals/
    ├── README.md
    └── stroke_prediction_counterfactuals.ipynb

```

Each project folder is self-contained — its own notebook, its own README (with the detailed write-up: dataset, model, explainer code, plots, and learnings), and its own images.

## Structure

Each project lives in its own numbered folder and includes:
- A Jupyter notebook with the full implementation
- A project-level README explaining the approach, results, and key takeaways
- Plot images referenced by that README
- (Where applicable) a `requirements.txt` for reproducibility
- Instructions for obtaining any dataset used (raw data isn't committed to the repo)

## Tech stack

Python, scikit-learn, pandas, XGBoost, LIME, SHAP, Transformers, Keras/TensorFlow, matplotlib/seaborn, Gradio — expanding as new projects are added.

## About

I'm a CS student learning ML/DL by building projects and documenting the process publicly. Feedback and suggestions are welcome — feel free to open an issue.

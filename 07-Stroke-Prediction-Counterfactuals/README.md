# 07 — Counterfactual Explanations: "What Would Need to Change?"

## What this project does

The first six projects in this repo answer "why did the model predict this?" by pointing at which features mattered. This project asks a different, more actionable question: **"what would need to change about this person's data for the model to predict differently?"**

I trained a **Random Forest** model to predict stroke risk from patient health data, then used **DiCE (Diverse Counterfactual Explanations)** to generate realistic "what-if" versions of a patient's data that would flip the model's prediction — e.g., "if this patient's average glucose level had been X instead of Y, the model would have predicted no-stroke instead of stroke."

This is a fundamentally different kind of explanation than LIME or SHAP:
- LIME/SHAP tell you **which features mattered** for a prediction that already happened.
- Counterfactuals tell you **what specific change** would produce a different outcome — which is often more directly useful for a real person or decision-maker asking "so what do I do about it?"

## A note on model performance

Before looking at the counterfactuals, it's worth being upfront about the model itself:

- **Accuracy: 94.2%** — looks great at first glance.
- **F1 Score: 0.034** — tells a very different story.

This gap is a textbook symptom of **class imbalance**: strokes are rare events in this dataset, so a model can score ~94% accuracy just by predicting "no stroke" almost every time, while still doing a poor job of actually catching the positive (stroke) cases — which is the class that actually matters here. I did apply **SMOTE oversampling** to the training data specifically to address this, but the F1 score shows it wasn't nearly enough to fix the underlying problem.

I'm including this here because it directly affects how to read the counterfactuals below: they're explaining the behavior of a model that is not yet reliable at its core task, not a validated clinical tool.

## Dataset

- **Healthcare Stroke Prediction dataset** (Kaggle: `healthcare-dataset-stroke-data.csv`), loaded from Google Drive.
- Preprocessing:
  - Dropped rows with missing values (`dropna`)
  - One-hot encoded all categorical columns into integer (0/1) columns via `pd.get_dummies(..., dtype=int)`
- Split 80/20 into train/test (`random_state=42`)
- **SMOTE oversampling** applied to the training set only, to rebalance the rare "stroke" class before training
- After oversampling: 7,542 training rows, 982 test rows, 22 features

## Model

- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`, default hyperparameters)
- Trained on the SMOTE-oversampled training data, evaluated on the untouched test set
- Results: **94.2% accuracy, 0.034 F1 score** (see note above — this model has real problems on the minority class)

## How the counterfactual explanation was generated

```python
import dice_ml

# 1. Tell DiCE about the dataset: which columns are continuous (DiCE needs to
#    know this to generate realistic perturbations), and which column is the
#    target/outcome.
data_dice = dice_ml.Data(
    dataframe=data_loader.data,
    continuous_features=['age', 'avg_glucose_level', 'bmi'],
    outcome_name='stroke'
)

# 2. Wrap the trained model for DiCE
rf_dice = dice_ml.Model(model=rf, backend='sklearn')

# 3. Create the explainer. method='random' generates counterfactuals by
#    randomly perturbing features and checking which combinations flip the
#    model's prediction (other methods exist, e.g. genetic algorithms,
#    gradient-based — 'random' is the simplest to start with).
explainer = dice_ml.Dice(data_dice, rf_dice, method='random')
```

**Unconstrained counterfactuals** — let DiCE vary any feature freely:

```python
input_datapoint = X_test[0:1].copy()

cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,                # generate 3 different counterfactual examples
    desired_class='opposite'    # flip the prediction to the opposite class
)

cf.visualize_as_dataframe(show_only_changes=True)  # only show what changed
```

**Constrained (feasible) counterfactuals** — this is the more realistic and useful version. Without constraints, DiCE is free to suggest changes that don't make practical sense (e.g., "reduce age by 20 years"). Restricting *which* features can vary, and *within what range*, keeps the suggestions actionable:

```python
features_to_vary = ['avg_glucose_level', 'bmi', 'smoking_status_smokes']
permitted_range = {'avg_glucose_level': [10, 300], 'bmi': [15, 45]}

input_datapoint2 = X_test[i:i+1]

cf = explainer.generate_counterfactuals(
    input_datapoint2,
    total_CFs=3,
    desired_class='opposite',
    permitted_range=permitted_range,
    features_to_vary=features_to_vary
)

cf.visualize_as_dataframe(show_only_changes=True)
```

**What the output looks like:** for a given patient (the "query instance"), DiCE returns a small table of alternate versions of that patient — each one showing only the features that changed, and by how much — such that the model would have predicted the opposite outcome for each alternate version.

## What I learned / observations

- **Unconstrained counterfactuals can suggest nonsensical changes.** Without `permitted_range` and `features_to_vary`, DiCE is free to vary any feature by any amount — which can produce technically-valid-but-unrealistic suggestions (e.g., wildly changing `age` or an ID column). Constraining which features can move, and within what bounds, is what makes the explanation actually usable.
- **Counterfactuals are a fundamentally different flavor of explanation** than LIME/SHAP — instead of ranking feature importance, they answer a concrete, individual "what would it take" question. This feels closer to what a non-technical stakeholder (e.g., a patient or doctor) would actually want to know.
- **A high-accuracy model can still be a bad model.** The accuracy/F1 gap here was a good reminder that accuracy alone can hide serious problems on imbalanced datasets, and that explainability tools will faithfully explain a flawed model just as readily as a good one — XAI doesn't validate the model, it just tells you how it behaves.

## How to run

```bash
pip install torch dice-ml imbalanced-learn scikit-learn pandas
```

Then run the notebook `Counterfactuals_in_XAI.ipynb` (inside `07-Stroke-Prediction-Counterfactuals-DiCE/`) top to bottom. You'll need the `healthcare-dataset-stroke-data.csv` file (from Kaggle) available at the path referenced in the notebook, or update the path to wherever you place it locally.

## References / credit

- [DiCE documentation](https://interpret.ml/DiCE/)
- [DiCE paper: "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"](https://arxiv.org/abs/1905.07697)
- [Healthcare Stroke Prediction dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [SMOTE (imbalanced-learn)](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

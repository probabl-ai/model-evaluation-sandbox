"""
Example from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
"""

# %%
import sklearn
from sklearn.datasets import fetch_openml

sklearn.set_config(transform_output="pandas")

german_credit = fetch_openml(data_id=31, as_frame=True, parser="pandas")
X, y = german_credit.data, german_credit.target
# %%

y.value_counts()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# %%
from sklearn.metrics import confusion_matrix

def fpr_score(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr

# %%
from sklearn.metrics import make_scorer, precision_score, recall_score

pos_label, neg_label = "bad", "good"
tpr_score = recall_score  # TPR and recall are the same metric
scoring = {
    "precision": make_scorer(precision_score, pos_label=pos_label),
    "recall": make_scorer(recall_score, pos_label=pos_label),
    "fpr": make_scorer(fpr_score, neg_label=neg_label, pos_label=pos_label),
    "tpr": make_scorer(tpr_score, pos_label=pos_label),
}

# %%
import numpy as np

def credit_gain_score(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label])
    gain_matrix = np.array(
        [
            [0, -1],  # -1 gain for false positives
            [-5, 0],  # -5 gain for false negatives
        ]
    )
    return np.sum(cm * gain_matrix)


scoring["cost_gain"] = make_scorer(
    credit_gain_score, neg_label=neg_label, pos_label=pos_label
)
# %%
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    categorical_features="from_dtype", random_state=0
).fit(X_train, y_train)
model

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

PrecisionRecallDisplay.from_estimator(
    model, X_test, y_test, pos_label=pos_label, ax=axs[0], name="GBDT"
)
axs[0].plot(
    scoring["recall"](model, X_test, y_test),
    scoring["precision"](model, X_test, y_test),
    marker="o",
    markersize=10,
    color="tab:blue",
    label="Default cut-off point at a probability of 0.5",
)
axs[0].set_title("Precision-Recall curve")
axs[0].legend()

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    pos_label=pos_label,
    ax=axs[1],
    name="GBDT",
    plot_chance_level=True,
)
axs[1].plot(
    scoring["fpr"](model, X_test, y_test),
    scoring["tpr"](model, X_test, y_test),
    marker="o",
    markersize=10,
    color="tab:blue",
    label="Default cut-off point at a probability of 0.5",
)
axs[1].set_title("ROC curve")
axs[1].legend()
_ = fig.suptitle("Evaluation of the vanilla GBDT model")

# %%
business_metric = scoring['cost_gain'](model, X_test, y_test)
print(f"Business defined metric: {business_metric}")

# %%

# Run `mlflow ui` in your terminal first
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Binary fraud")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(model.get_params())

    # Log the loss metric
    mlflow.log_metric("business_metric", business_metric)

    # Log the figures
    mlflow.log_figure(fig, "PR_ROC_curve.png")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "HGBT with default threshold")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="binary_fraud",
        signature=signature,
        input_example=X_train,
        registered_model_name="HGBT-default",
    )

# %%

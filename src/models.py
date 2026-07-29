"""Registry of the algorithms compared in this project. Each factory
returns an unfitted estimator; pass use_class_weight=True to apply
class_weight='balanced' for models that support it."""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
    "naive_bayes",
    "svm",
    "xgboost",
    "lightgbm",
    "catboost",
]

# KNN and Naive Bayes have no class_weight concept, so the
# "class_weight" strategy is skipped for them in train.py.
SUPPORTS_CLASS_WEIGHT = {"logistic_regression", "decision_tree", "random_forest", "svm"}


def build_model(name, use_class_weight=False):
    class_weight = "balanced" if use_class_weight else None

    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=class_weight, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "svm":
        # LinearSVC, not a full RBF-kernel SVC: the kernelized version
        # doesn't scale to this many rows, especially once oversampling
        # is applied on top.
        return LinearSVC(class_weight=class_weight, random_state=RANDOM_STATE, max_iter=5000)
    if name == "xgboost":
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    if name == "lightgbm":
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    if name == "catboost":
        return CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)

    raise ValueError(f"Unknown model: {name}")

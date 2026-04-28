"""Configuration constants for ML models."""

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


DEFAULT_PARAM_GRIDS = {
    "LinearRegression": {"fit_intercept": [True, False]},
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    },
    "Lasso": {"alpha": [0.1, 1.0, 10.0], "max_iter": [1000, 5000]},
    "DecisionTreeRegressor": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 10],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 10],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
}

MODEL_MAP = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "SVC": SVC,
    "KNeighborsRegressor": KNeighborsRegressor,
    "KNeighborsClassifier": KNeighborsClassifier,
    "LogisticRegression": LogisticRegression,
}

REGRESSION_MODELS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "SVR",
    "KNeighborsRegressor",
]

CLASSIFICATION_MODELS = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "SVC",
    "KNeighborsClassifier",
]

SCALING_METHODS = ["none", "standard", "minmax", "robust"]

"""PyMLUI Core - GUI-agnostic ML training logic."""

from .config import (
    DEFAULT_PARAM_GRIDS,
    MODEL_MAP,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    SCALING_METHODS,
)
from .trainer import (
    train_model_with_config,
    calculate_feature_importance,
    calculate_shap_values,
    create_scaler,
    SHAP_AVAILABLE,
)
from .visualization import (
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_dependence,
    plot_feature_importance_comparison,
    plot_model_internal_importance,
    plot_regression_results,
    plot_classification_results,
    plot_learning_curve,
    set_plot_blocking,
)

__all__ = [
    # Config
    "DEFAULT_PARAM_GRIDS",
    "MODEL_MAP",
    "REGRESSION_MODELS",
    "CLASSIFICATION_MODELS",
    "SCALING_METHODS",
    # Trainer
    "train_model_with_config",
    "calculate_feature_importance",
    "calculate_shap_values",
    "create_scaler",
    "SHAP_AVAILABLE",
    # Visualization
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_dependence",
    "plot_feature_importance_comparison",
    "plot_model_internal_importance",
    "plot_regression_results",
    "plot_classification_results",
    "plot_learning_curve",
]

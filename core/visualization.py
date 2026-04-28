"""Visualization functions - GUI agnostic."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_PLOT_BLOCK = True


def set_plot_blocking(block):
    """Set whether plt.show() should block. Call with False for non-blocking (Qt apps)."""
    global _PLOT_BLOCK
    _PLOT_BLOCK = block


def _show():
    """Call plt.show() with appropriate blocking behavior."""
    if _PLOT_BLOCK:
        plt.show()
    else:
        plt.show(block=False)


def plot_shap_summary(shap_values, feature_names, title="SHAP Summary"):
    """Plot SHAP summary plot."""
    if shap_values is None or not SHAP_AVAILABLE:
        print("SHAP values not available")
        return False

    try:
        plt.figure(figsize=(12, 8))
        shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(
            shap_values_to_plot,
            feature_names=feature_names,
            show=False,
            max_display=min(20, len(feature_names))
        )
        plt.title(title, fontsize=14)
        plt.tight_layout()
        _show()
        return True
    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
        return False


def plot_shap_bar(shap_importance, title="SHAP Feature Importance"):
    """Plot SHAP feature importance bar chart."""
    if shap_importance is None:
        print("SHAP importance not available")
        return False

    try:
        plt.figure(figsize=(10, 6))
        sorted_items = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        features = [item[0] for item in sorted_items]
        importance = [item[1] for item in sorted_items]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        plt.barh(features, importance, color=colors)
        plt.xlabel('Mean |SHAP value|')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        _show()
        return True
    except Exception as e:
        print(f"Error plotting SHAP bar: {e}")
        return False


def plot_shap_dependence(shap_values, feature_names, feature_index=0, data=None,
                         title="SHAP Dependence Plot"):
    """Plot SHAP dependence plot."""
    if shap_values is None or not SHAP_AVAILABLE:
        print("SHAP values not available")
        return False

    try:
        plt.figure(figsize=(10, 6))
        shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
        if feature_index >= len(feature_names):
            feature_index = 0
        feature_name = feature_names[feature_index]

        shap.dependence_plot(
            feature_name, shap_values_to_plot, data,
            feature_names=feature_names, show=False
        )
        plt.title(title, fontsize=14)
        plt.tight_layout()
        _show()
        return True
    except Exception as e:
        print(f"Error plotting SHAP dependence: {e}")
        return False


def plot_feature_importance_comparison(feature_importance_dicts, title="Feature Importance Comparison"):
    """Compare different feature importance methods."""
    if not feature_importance_dicts:
        print("No feature importance data available")
        return False

    try:
        plt.figure(figsize=(12, 8))
        all_features = set()
        for imp_dict in feature_importance_dicts.values():
            if imp_dict:
                all_features.update(imp_dict.keys())

        features = list(all_features)[:10]
        data = []
        method_names = []

        for method_name, imp_dict in feature_importance_dicts.items():
            if imp_dict:
                method_names.append(method_name)
                row = [imp_dict.get(feature, 0) for feature in features]
                data.append(row)

        if not data:
            return False

        data = np.array(data)
        data_norm = data / (data.sum(axis=1, keepdims=True) + 1e-10)

        x = np.arange(len(features))
        bottom = np.zeros(len(features))
        for i in range(len(method_names)):
            plt.bar(x, data_norm[i], bottom=bottom, label=method_names[i], alpha=0.7)
            bottom += data_norm[i]

        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title(title)
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        _show()
        return True
    except Exception as e:
        print(f"Error plotting feature importance comparison: {e}")
        return False


def plot_model_internal_importance(importance_dict, title="Model Internal Feature Importance"):
    """Plot model's internal feature importance."""
    if not importance_dict:
        print("Importance data not available")
        return False

    try:
        plt.figure(figsize=(10, 6))
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        features = [item[0] for item in sorted_items]
        importance = [item[1] for item in sorted_items]

        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(features)))
        plt.barh(features, importance, color=colors)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        _show()
        return True
    except Exception as e:
        print(f"Error plotting model internal importance: {e}")
        return False


def plot_regression_results(y_true, y_pred, title):
    """Plot regression results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('True vs Predicted')
    axes[0].grid(True, alpha=0.3)

    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    _show()


def plot_classification_results(y_true, y_pred, title):
    """Plot classification results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_true, y_pred)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=axes[0])

    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xticks(range(len(unique_labels)))
    axes[0].set_yticks(range(len(unique_labels)))
    axes[0].set_xticklabels(unique_labels)
    axes[0].set_yticklabels(unique_labels)

    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            axes[0].text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    accuracy = accuracy_score(y_true, y_pred)
    labels = ['Correct', 'Incorrect']
    sizes = [accuracy, 1 - accuracy]
    colors = ['lightgreen', 'lightcoral']

    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].axis('equal')
    axes[1].set_title(f'Accuracy: {accuracy:.2%}')

    plt.suptitle(title)
    plt.tight_layout()
    _show()


def plot_learning_curve(model, X, y, title, problem_type='regression'):
    """Plot learning curve."""
    scoring = 'neg_mean_squared_error' if problem_type == 'regression' else 'accuracy'

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_scores_mean = (
        -np.mean(train_scores, axis=1)
        if problem_type == 'regression' else np.mean(train_scores, axis=1)
    )
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = (
        -np.mean(test_scores, axis=1)
        if problem_type == 'regression' else np.mean(test_scores, axis=1)
    )
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error" if problem_type == 'regression' else "Accuracy")
    plt.title(f"Learning Curve - {title}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    _show()

"""Streamlit WebUI for PyMLUI."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from core import (
    MODEL_MAP,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    SCALING_METHODS,
    DEFAULT_PARAM_GRIDS,
    train_model_with_config,
    plot_regression_results,
    plot_classification_results,
    plot_feature_importance_comparison,
    plot_model_internal_importance,
    plot_shap_bar,
    SHAP_AVAILABLE,
)


st.set_page_config(page_title="PyMLUI - ML Training", page_icon="🤖", layout="wide")

st.title("PyMLUI - Machine Learning Training")
st.markdown("Upload your CSV file, select features and model to train")

if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.results = None


uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"File loaded: {uploaded_file.name} ({len(data)} rows, {len(data.columns)} columns)")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Feature Selection")
        all_columns = list(data.columns)
        y_col = st.selectbox("Target Column (Y)", all_columns)

        feature_columns = st.multiselect("Feature Columns (X)", [c for c in all_columns if c != y_col])

        use_pca = st.checkbox("Use PCA")
        pca_components = None
        if use_pca:
            pca_components = st.text_input("PCA Components", "all")

    with col2:
        st.subheader("Model Configuration")
        problem_type = st.radio("Problem Type", ["regression", "classification"], horizontal=True)

        if problem_type == "regression":
            model_options = REGRESSION_MODELS
        else:
            model_options = CLASSIFICATION_MODELS

        model_name = st.selectbox("Model", model_options)

        x_scaling = st.selectbox("X Scaling", SCALING_METHODS)

        y_scaling = None
        if problem_type == "regression":
            y_scaling = st.selectbox("Y Scaling", SCALING_METHODS)

        use_param_grid = st.checkbox("Hyperparameter Tuning (GridSearchCV)")

    if st.button("Train Model", type="primary"):
        if not feature_columns:
            st.error("Please select at least one feature column")
        elif not y_col:
            st.error("Please select a target column")
        else:
            with st.spinner("Training model..."):
                feature_config = {
                    "x_cols": feature_columns,
                    "y_col": y_col,
                    "use_pca": use_pca,
                    "pca_components": pca_components,
                    "x_scaling": x_scaling,
                    "y_scaling": y_scaling,
                }

                model_class = MODEL_MAP.get(model_name)
                model_config = {
                    "model_name": model_name,
                    "model": model_class(),
                    "problem_type": problem_type,
                }

                param_grid = DEFAULT_PARAM_GRIDS.get(model_name) if use_param_grid else None

                results = train_model_with_config(data, feature_config, model_config, param_grid)

                if results.get("success"):
                    st.session_state.trained = True
                    st.session_state.results = results
                    st.session_state.problem_type = problem_type
                    st.session_state.model_name = model_name
                    st.success("Model trained successfully!")
                else:
                    st.error(f"Training failed: {results.get('error', 'Unknown error')}")

if st.session_state.trained and st.session_state.results:
    st.divider()
    st.subheader("Results")

    results = st.session_state.results
    problem_type = st.session_state.problem_type
    model_name = st.session_state.model_name

    col1, col2, col3 = st.columns(3)

    if problem_type == "regression":
        col1.metric("MSE", f"{results.get('mse', 0):.4f}")
        col2.metric("MAE", f"{results.get('mae', 0):.4f}")
        col3.metric("R²", f"{results.get('r2', 0):.4f}")
    else:
        col1.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
        col2.metric("F1", f"{results.get('f1', 0):.4f}")
        col3.metric("", "")

    if "best_params" in results:
        st.info(f"**Best Parameters:** {results['best_params']}")
    if "best_score" in results:
        st.info(f"**Best CV Score:** {results['best_score']:.4f}")
    if "pca_explained_variance" in results:
        st.info(f"**PCA Explained Variance:** {results['pca_explained_variance']:.4f}")

    st.divider()
    st.subheader("Plots")

    plot_options = ["Results"]
    if results.get("feature_importance"):
        plot_options.append("Feature Importance Comparison")
    imp = results.get("feature_importance", {}).get("model_internal") or \
          results.get("feature_importance", {}).get("model_coefficients")
    if imp:
        plot_options.append("Model Internal Importance")
    if results.get("shap_importance") and SHAP_AVAILABLE:
        plot_options.append("SHAP Importance")

    selected_plots = st.multiselect("Select plots to display", plot_options, default=["Results"])

    if "Results" in selected_plots:
        st.markdown("### Results Plot")
        if problem_type == "regression":
            fig = plot_regression_results_py(results["y_test"], results["y_pred"], f"{model_name} Results")
        else:
            fig = plot_classification_results_py(results["y_test"], results["y_pred"], f"{model_name} Results")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    if "Feature Importance Comparison" in selected_plots and results.get("feature_importance"):
        st.markdown("### Feature Importance Comparison")
        fig = plot_feature_importance_comparison_py(results["feature_importance"], "Feature Importance Comparison")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    if "Model Internal Importance" in selected_plots and imp:
        st.markdown("### Model Internal Importance")
        fig = plot_model_internal_importance_py(imp, "Model Internal Feature Importance")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    if "SHAP Importance" in selected_plots and results.get("shap_importance") and SHAP_AVAILABLE:
        st.markdown("### SHAP Importance")
        fig = plot_shap_bar_py(results["shap_importance"], "SHAP Feature Importance")
        if fig:
            st.pyplot(fig)
            plt.close(fig)


def plot_regression_results_py(y_true, y_pred, title):
    """Plot regression results using matplotlib."""
    import matplotlib.pyplot as plt
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
    return fig


def plot_classification_results_py(y_true, y_pred, title):
    """Plot classification results using matplotlib."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score
    import numpy as np

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
    return fig


def plot_feature_importance_comparison_py(feature_importance_dicts, title="Feature Importance Comparison"):
    """Plot feature importance comparison using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not feature_importance_dicts:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
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
        return None

    data = np.array(data)
    data_norm = data / (data.sum(axis=1, keepdims=True) + 1e-10)

    x = np.arange(len(features))
    bottom = np.zeros(len(features))
    for i in range(len(method_names)):
        ax.bar(x, data_norm[i], bottom=bottom, label=method_names[i], alpha=0.7)
        bottom += data_norm[i]

    ax.set_xlabel('Features')
    ax.set_ylabel('Normalized Importance')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_model_internal_importance_py(importance_dict, title="Model Internal Feature Importance"):
    """Plot model internal importance using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not importance_dict:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [item[0] for item in sorted_items]
    importance = [item[1] for item in sorted_items]

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, importance, color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_shap_bar_py(shap_importance, title="SHAP Feature Importance"):
    """Plot SHAP bar chart using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    if shap_importance is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_items = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [item[0] for item in sorted_items]
    importance = [item[1] for item in sorted_items]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, importance, color=colors)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.argv = ["streamlit", "run", __file__]
    import os
    os.system(" ".join(sys.argv))

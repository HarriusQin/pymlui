"""Core ML training functions - GUI agnostic."""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def create_scaler(method):
    """Create a scaler based on the specified method."""
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    return scalers.get(method)


def train_model_with_config(data, feature_config, model_config, param_grid=None):
    """Train model with given configuration.

    Args:
        data: pandas.DataFrame with the dataset
        feature_config: dict with keys:
            - x_cols: list of feature column names
            - y_col: target column name
            - use_pca: bool
            - pca_components: int or "all"
            - x_scaling: str ("none", "standard", "minmax", "robust")
            - y_scaling: str for regression
        model_config: dict with keys:
            - model_name: str
            - model: sklearn estimator instance
            - problem_type: "regression" or "classification"
        param_grid: dict or None for GridSearchCV

    Returns:
        dict with training results including:
            - success: bool
            - model: trained model
            - metrics (mse, mae, r2 or accuracy, f1)
            - feature_importance, shap_importance, etc.
    """
    results = {}

    try:
        X = data[feature_config['x_cols']].values
        y = data[feature_config['y_col']].values
        original_feature_names = feature_config['x_cols']

        if model_config['problem_type'] == 'regression' and y.ndim == 1:
            y = y.reshape(-1, 1)

        x_scaler = create_scaler(feature_config['x_scaling'])
        y_scaler = (
            create_scaler(feature_config['y_scaling'])
            if model_config['problem_type'] == 'regression' else None
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test_original = X_test.copy()

        if x_scaler:
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        if y_scaler and model_config['problem_type'] == 'regression':
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test

        pca_transformer = None
        if feature_config['use_pca']:
            try:
                n_components = feature_config['pca_components']
                n_comp_str = str(n_components) if n_components is not None else "0"
                if n_comp_str in ("0", "all", "All"):
                    n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
                else:
                    n_components = min(int(float(n_components)), X_train_scaled.shape[1])

                pca_transformer = PCA(n_components=n_components, random_state=42)
                X_train_scaled = pca_transformer.fit_transform(X_train_scaled)
                X_test_scaled = pca_transformer.transform(X_test_scaled)
                results['pca_explained_variance'] = pca_transformer.explained_variance_ratio_.sum()
                pca_feature_names = [f'PCA_{i+1}' for i in range(n_components)]
            except Exception as e:
                results['pca_error'] = str(e)
                pca_transformer = None
                pca_feature_names = original_feature_names
        else:
            pca_feature_names = original_feature_names

        y_train_flat = (
            y_train_scaled.ravel()
            if len(y_train_scaled.shape) > 1 and y_train_scaled.shape[1] == 1
            else y_train_scaled
        )

        if param_grid:
            scoring = (
                'neg_mean_squared_error'
                if model_config['problem_type'] == 'regression' else 'accuracy'
            )
            grid_search = GridSearchCV(
                model_config['model'], param_grid,
                cv=5, scoring=scoring, n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train_flat)
            best_model = grid_search.best_estimator_
            results['best_params'] = grid_search.best_params_
            results['best_score'] = grid_search.best_score_
        else:
            best_model = model_config['model']
            best_model.fit(X_train_scaled, y_train_flat)

        y_pred_scaled = best_model.predict(X_test_scaled)

        if y_scaler and model_config['problem_type'] == 'regression':
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_test_original = y_scaler.inverse_transform(y_test_scaled).ravel()
        else:
            y_pred = y_pred_scaled
            y_test_original = (
                y_test_scaled.ravel()
                if len(y_test_scaled.shape) > 1 else y_test_scaled
            )

        if model_config['problem_type'] == 'regression':
            results['mse'] = mean_squared_error(y_test_original, y_pred)
            results['mae'] = mean_absolute_error(y_test_original, y_pred)
            results['r2'] = r2_score(y_test_original, y_pred)
        else:
            results['accuracy'] = accuracy_score(y_test_original, y_pred)
            results['f1'] = f1_score(y_test_original, y_pred, average='weighted')

        feature_importance_results = calculate_feature_importance(
            best_model, X_train_scaled, X_test_scaled, y_train_flat,
            pca_feature_names, original_feature_names,
            feature_config, model_config, pca_transformer
        )
        results.update(feature_importance_results)

        results.update({
            'model': best_model,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'pca_transformer': pca_transformer,
            'X_test': X_test_scaled,
            'X_test_original': X_test_original,
            'y_test': y_test_original,
            'y_pred': y_pred,
            'original_feature_names': original_feature_names,
            'pca_feature_names': pca_feature_names,
            'success': True,
        })

    except Exception as e:
        results['success'] = False
        results['error'] = str(e)

    return results


def calculate_feature_importance(model, X_train, X_test, y_train,
                                  pca_feature_names, original_feature_names,
                                  feature_config, model_config, pca_transformer):
    """Calculate feature importance including SHAP values."""
    results = {
        'feature_importance': {},
        'shap_available': SHAP_AVAILABLE,
        'shap_values': None,
        'shap_importance': None,
    }

    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            results['feature_importance']['model_internal'] = dict(zip(pca_feature_names, importance))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            importance = (
                np.mean(np.abs(coef), axis=0)
                if len(coef.shape) > 1 else np.abs(coef)
            )
            if len(importance) == len(pca_feature_names):
                results['feature_importance']['model_coefficients'] = dict(
                    zip(pca_feature_names, importance)
                )

        if SHAP_AVAILABLE and X_test.shape[0] > 0:
            shap_results = calculate_shap_values(
                model, X_train, X_test, pca_feature_names,
                original_feature_names, feature_config, model_config, pca_transformer
            )
            results.update(shap_results)

    except Exception as e:
        results['feature_importance_error'] = str(e)

    return results


def calculate_shap_values(model, X_train, X_test, pca_feature_names,
                          original_feature_names, feature_config, model_config,
                          pca_transformer):
    """Calculate SHAP values for model interpretability."""
    results = {}

    try:
        n_samples = min(100, X_test.shape[0])
        X_test_sample = X_test[:n_samples]
        model_name = model_config['model_name']

        if any(tree_model in model_name for tree_model in
               ['DecisionTree', 'RandomForest', 'GradientBoosting']):
            if hasattr(model, 'predict_proba') and model_config['problem_type'] == 'classification':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                if isinstance(shap_values, list):
                    shap_values = np.mean(np.abs(shap_values), axis=0)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
        elif any(linear_model in model_name for linear_model in
                 ['LinearRegression', 'Ridge', 'Lasso', 'LogisticRegression']):
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test_sample)
        elif 'SVC' in model_name or 'SVR' in model_name:
            n_samples = min(50, X_test.shape[0])
            X_test_sample = X_test[:n_samples]
            explainer = shap.KernelExplainer(model.predict, X_train[:100])
            shap_values = explainer.shap_values(X_test_sample)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train[:50])
            shap_values = explainer.shap_values(X_test_sample)

        if shap_values is not None:
            if isinstance(shap_values, dict):
                shap_abs = np.mean([np.abs(v) for v in shap_values.values()], axis=0)
            elif isinstance(shap_values, list):
                shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_abs = np.abs(shap_values)
            shap_importance = np.mean(shap_abs, axis=0)
            if shap_importance.ndim > 1:
                shap_importance = shap_importance.flatten()

            results['shap_values'] = shap_values
            results['shap_importance'] = dict(zip(pca_feature_names, shap_importance))
            results['shap_explainer'] = explainer

            if feature_config['use_pca'] and pca_transformer is not None:
                try:
                    pca_components = pca_transformer.components_
                    shap_original = np.zeros((shap_abs.shape[0], len(original_feature_names)))
                    for i in range(shap_abs.shape[0]):
                        shap_original[i] = np.dot(shap_abs[i], pca_components)
                    shap_importance_original = np.mean(shap_original, axis=0)
                    results['shap_importance_original'] = dict(
                        zip(original_feature_names, shap_importance_original)
                    )
                except Exception as e:
                    results['shap_pca_mapping_error'] = str(e)

    except Exception as e:
        results['shap_error'] = str(e)

    return results

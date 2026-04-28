"""Flask WebUI for PyMLUI."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
from flask import Flask, render_template, request, jsonify, Response
import matplotlib.pyplot as plt
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html",
                           regression_models=REGRESSION_MODELS,
                           classification_models=CLASSIFICATION_MODELS,
                           scaling_methods=SCALING_METHODS)


@app.route("/train", methods=["POST"])
def train():
    """Train model endpoint."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        df = pd.read_csv(file)
        columns = list(df.columns)

        return jsonify({
            "success": True,
            "columns": columns,
            "row_count": len(df)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/train_model", methods=["POST"])
def train_model():
    """Train model with configuration."""
    try:
        data = pd.read_csv(request.files["file"])
        feature_cols = request.form.getlist("feature_cols")
        y_col = request.form["y_col"]
        model_name = request.form["model_name"]
        problem_type = request.form["problem_type"]
        use_pca = request.form.get("use_pca") == "true"
        pca_components = request.form.get("pca_components", "all")
        x_scaling = request.form.get("x_scaling", "none")
        y_scaling = request.form.get("y_scaling", "none")
        use_param_grid = request.form.get("use_param_grid") == "true"

        feature_config = {
            "x_cols": feature_cols,
            "y_col": y_col,
            "use_pca": use_pca,
            "pca_components": pca_components,
            "x_scaling": x_scaling,
            "y_scaling": y_scaling,
        }

        model_class = MODEL_MAP.get(model_name)
        if model_class is None:
            return jsonify({"success": False, "error": f"Unknown model: {model_name}"})

        model_config = {
            "model_name": model_name,
            "model": model_class(),
            "problem_type": problem_type,
        }

        param_grid = DEFAULT_PARAM_GRIDS.get(model_name) if use_param_grid else None

        results = train_model_with_config(data, feature_config, model_config, param_grid)

        if not results.get("success"):
            return jsonify({"success": False, "error": results.get("error", "Training failed")})

        response_data = {
            "success": True,
            "metrics": {},
        }

        if problem_type == "regression":
            response_data["metrics"] = {
                "MSE": results.get("mse"),
                "MAE": results.get("mae"),
                "R2": results.get("r2"),
            }
            if "pca_explained_variance" in results:
                response_data["pca_explained_variance"] = results["pca_explained_variance"]
        else:
            response_data["metrics"] = {
                "Accuracy": results.get("accuracy"),
                "F1": results.get("f1"),
            }

        if "best_params" in results:
            response_data["best_params"] = results["best_params"]
        if "best_score" in results:
            response_data["best_score"] = results["best_score"]

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/plot/<plot_type>")
def get_plot(plot_type):
    """Generate and return plot as image."""
    try:
        data = pd.read_csv(request.args.get("file"))
        feature_cols = request.args.getlist("feature_cols")
        y_col = request.args["y_col"]
        model_name = request.args["model_name"]
        problem_type = request.args["problem_type"]
        use_pca = request.args.get("use_pca") == "true"
        pca_components = request.args.get("pca_components", "all")
        x_scaling = request.args.get("x_scaling", "none")
        y_scaling = request.args.get("y_scaling", "none")
        use_param_grid = request.args.get("use_param_grid") == "true"

        feature_config = {
            "x_cols": feature_cols,
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

        if not results.get("success"):
            return jsonify({"error": results.get("error", "Training failed")})

        fig = None

        if plot_type == "results":
            if problem_type == "regression":
                plot_regression_results(
                    results["y_test"],
                    results["y_pred"],
                    f"{model_name} Results"
                )
            else:
                plot_classification_results(
                    results["y_test"],
                    results["y_pred"],
                    f"{model_name} Results"
                )

        elif plot_type == "importance":
            if results.get("feature_importance"):
                plot_feature_importance_comparison(
                    results["feature_importance"],
                    "Feature Importance Comparison"
                )

        elif plot_type == "model_importance":
            imp = results.get("feature_importance", {}).get("model_internal") or \
                  results.get("feature_importance", {}).get("model_coefficients")
            if imp:
                plot_model_internal_importance(imp, "Model Internal Feature Importance")

        elif plot_type == "shap" and SHAP_AVAILABLE:
            if results.get("shap_importance"):
                plot_shap_bar(results["shap_importance"], "SHAP Feature Importance")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        plt.close("all")

        return Response(buf.getvalue(), mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


def main():
    """Run Flask app."""
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()

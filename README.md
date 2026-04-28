# PyMLUI - Machine Learning Trainer with SHAP

A GUI-based machine learning tool that provides an intuitive interface for training, evaluating, and interpreting ML models with SHAP feature importance analysis.

## Features

- **Data Loading**: Load CSV files with automatic type detection
- **Feature Selection**: Interactive dialogs for selecting X features and Y target
- **Model Support**:
  - Regression: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting, SVR, KNN
  - Classification: LogisticRegression, DecisionTree, RandomForest, SVC, KNN
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, PCA
- **Hyperparameter Tuning**: GridSearchCV with default parameter grids
- **Visualization**:
  - Regression: True vs Predicted, Residual plots
  - Classification: Confusion matrix, Accuracy pie chart
  - Learning curves
- **SHAP Analysis**:
  - SHAP summary plots
  - Feature importance bar charts
  - Dependence plots
  - Comparison across multiple importance methods

## Installation

```bash
# Create virtual environment
uv venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install pandas numpy matplotlib scikit-learn shap
```

## Usage

```bash
.venv\Scripts\python mlui.py
```

Or with uv:

```bash
uv run --active python mlui.py
```

## Workflow

### 1. Load Data
Click **Load Data** and select a CSV file. The tool displays data shape, types, and missing values.

### 2. Create Configuration
Click **Add Configuration** to define:
- **X Features**: Select multiple features as input variables
- **Y Feature**: Select the target variable
- **Model**: Choose algorithm and problem type (regression/classification)
- **Scaling**: Choose X/Y scaling method
- **PCA**: Optionally apply PCA dimensionality reduction
- **Parameters**: Use default grid or customize

### 3. Train Model
- **Train Selected**: Train the currently selected configuration
- **Train All**: Batch train all configurations

### 4. View Results
Training results display:
- Evaluation metrics (MSE, MAE, R² for regression; Accuracy, F1 for classification)
- Grid search best parameters
- Feature importance from model internal methods and SHAP

### 5. Visualize
- **Plot Results**: Regression/classification visualization
- **Plot Learning Curve**: Model learning performance
- **SHAP Analysis**: Open SHAP visualization window with:
  - Summary plot
  - Feature importance bar plot
  - Dependence plot
  - Comparison of importance methods

### 6. Save Results
Click **Save Results** to export training output to a text file.

## Configuration Management

- **Edit Selected**: Modify an existing configuration
- **Duplicate Selected**: Clone a configuration for variation
- **Delete Selected**: Remove a configuration
- **Clear All**: Remove all configurations

## Project Structure

```
pymlui/
├── mlui.py          # Main application
├── pyproject.toml   # Project configuration
├── .venv/           # Virtual environment
```

## Dependencies

- pandas, numpy - Data manipulation
- matplotlib - Visualization
- scikit-learn - Machine learning
- shap - Model interpretability
- tkinter - GUI (built-in)

## SHAP Integration

SHAP (SHapley Additive exPlanations) provides model-agnostic feature importance:

- **Tree Models**: Uses TreeExplainer for fast exact computation
- **Linear Models**: Uses LinearExplainer
- **Other Models**: Uses KernelExplainer (slower)

When PCA is enabled, SHAP values are mapped back to original feature space for interpretability.

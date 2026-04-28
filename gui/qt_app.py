"""PyQt6 GUI for PyMLUI - Matching Tkinter layout and logic."""

import os
import sys
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QGroupBox, QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QMessageBox, QDialog, QLineEdit, QFrame, QScrollArea,
    QSizePolicy, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon

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
    plot_learning_curve,
    create_scaler,
    set_plot_blocking,
    SHAP_AVAILABLE,
)

set_plot_blocking(False)


class FeatureSelectionDialog(QDialog):
    """Feature selection dialog with two listboxes."""

    def __init__(self, parent, title, features, multiselect=True, initial_selection=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(700, 500)
        self.features = features
        self.multiselect = multiselect
        self.selected_features = initial_selection.copy() if initial_selection else []

        self.init_ui()
        self.center_window()

    def center_window(self):
        self.adjustSize()
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        control_layout = QHBoxLayout()

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.textChanged.connect(self.filter_features)
        search_layout.addWidget(self.search_input)
        control_layout.addLayout(search_layout)

        btn_frame = QFrame()
        btn_frame_layout = QHBoxLayout(btn_frame)
        btn_frame_layout.setContentsMargins(0, 0, 0, 0)
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.deselect_all)
        btn_frame_layout.addWidget(select_all_btn)
        btn_frame_layout.addWidget(clear_btn)
        control_layout.addWidget(btn_frame)

        main_layout.addLayout(control_layout)

        listbox_layout = QHBoxLayout()

        available_frame = QFrame()
        available_layout = QVBoxLayout(available_frame)
        available_layout.setContentsMargins(0, 0, 0, 0)
        available_layout.addWidget(QLabel("Available Features"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection if self.multiselect
            else QListWidget.SelectionMode.SingleSelection
        )
        available_layout.addWidget(self.available_list)

        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(5, 0, 5, 0)
        add_btn = QPushButton(">")
        add_btn.clicked.connect(self.select_feature)
        add_all_btn = QPushButton(">>")
        add_all_btn.clicked.connect(self.select_all_features)
        remove_btn = QPushButton("<")
        remove_btn.clicked.connect(self.remove_feature)
        remove_all_btn = QPushButton("<<")
        remove_all_btn.clicked.connect(self.remove_all_features)
        button_layout.addWidget(add_btn)
        button_layout.addWidget(add_all_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(remove_all_btn)
        button_layout.addStretch()

        selected_frame = QFrame()
        selected_layout = QVBoxLayout(selected_frame)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.addWidget(QLabel("Selected Features"))
        self.selected_list = QListWidget()
        selected_layout.addWidget(self.selected_list)

        listbox_layout.addWidget(available_frame)
        listbox_layout.addWidget(button_frame)
        listbox_layout.addWidget(selected_frame)

        main_layout.addLayout(listbox_layout)

        self.info_label = QLabel(f"Total: {len(self.features)} features")
        main_layout.addWidget(self.info_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel)
        bottom_layout.addWidget(ok_btn)
        bottom_layout.addWidget(cancel_btn)
        main_layout.addLayout(bottom_layout)

        self.update_available_list()
        for feature in self.selected_features:
            self.selected_list.addItem(feature)

    def update_available_list(self):
        self.available_list.clear()
        search_text = self.search_input.text().lower()
        for feature in self.features:
            if feature not in self.selected_features and search_text in feature.lower():
                self.available_list.addItem(feature)
        self.update_info_label()

    def update_info_label(self):
        self.info_label.setText(
            f"Available: {self.available_list.count()}, Selected: {self.selected_list.count()}"
        )

    def filter_features(self):
        self.update_available_list()

    def select_feature(self):
        for item in self.available_list.selectedItems():
            feature = item.text()
            if feature not in self.selected_features:
                self.selected_features.append(feature)
                self.selected_list.addItem(feature)
        self.update_available_list()

    def select_all_features(self):
        for i in range(self.available_list.count()):
            feature = self.available_list.item(i).text()
            if feature not in self.selected_features:
                self.selected_features.append(feature)
                self.selected_list.addItem(feature)
        self.update_available_list()

    def remove_feature(self):
        for item in self.selected_list.selectedItems():
            feature = item.text()
            if feature in self.selected_features:
                self.selected_features.remove(feature)
            self.selected_list.takeItem(self.selected_list.row(item))
        self.update_available_list()

    def remove_all_features(self):
        self.selected_features.clear()
        self.selected_list.clear()
        self.update_available_list()

    def select_all(self):
        self.available_list.selectAll()

    def deselect_all(self):
        self.available_list.clearSelection()

    def accept(self):
        self.selected_features = [self.selected_list.item(i).text()
                                  for i in range(self.selected_list.count())]
        super().accept()

    def cancel(self):
        self.selected_features = []
        super().reject()

    def get_selected_features(self):
        return self.selected_features


class ModelConfigDialog(QDialog):
    """Model configuration dialog."""

    def __init__(self, parent, initial_config=None):
        super().__init__(parent)
        self.setWindowTitle("Model Configuration" if not initial_config else "Edit Configuration")
        self.setMinimumSize(500, 450)
        self.initial_config = initial_config or {}
        self.result = None
        self.custom_param_grid = None

        self.init_ui()
        self.center_window()

        if initial_config:
            self.load_initial_config()

    def center_window(self):
        self.adjustSize()
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        problem_frame = QGroupBox("Problem Type")
        problem_layout = QVBoxLayout(problem_frame)
        self.problem_type = "regression"
        self.regression_rb = QPushButton("Regression")
        self.regression_rb.setCheckable(True)
        self.regression_rb.setChecked(True)
        self.regression_rb.clicked.connect(lambda: self.update_model_options("regression"))
        self.classification_rb = QPushButton("Classification")
        self.classification_rb.setCheckable(True)
        self.classification_rb.clicked.connect(lambda: self.update_model_options("classification"))
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.regression_rb)
        btn_layout.addWidget(self.classification_rb)
        problem_layout.addLayout(btn_layout)
        main_layout.addWidget(problem_frame)

        model_frame = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        model_layout.addWidget(self.model_combo)
        main_layout.addWidget(model_frame)

        scaling_frame = QGroupBox("Data Scaling")
        scaling_layout = QVBoxLayout(scaling_frame)
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Scaling:"))
        self.x_scaling_combo = QComboBox()
        self.x_scaling_combo.addItems(SCALING_METHODS)
        x_layout.addWidget(self.x_scaling_combo)
        x_layout.addStretch()
        scaling_layout.addLayout(x_layout)
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Scaling (regression only):"))
        self.y_scaling_combo = QComboBox()
        self.y_scaling_combo.addItems(SCALING_METHODS)
        y_layout.addWidget(self.y_scaling_combo)
        y_layout.addStretch()
        scaling_layout.addLayout(y_layout)
        main_layout.addWidget(scaling_frame)

        pca_frame = QGroupBox("PCA Options")
        pca_layout = QVBoxLayout(pca_frame)
        self.use_pca_cb = QCheckBox("Use PCA")
        self.use_pca_cb.stateChanged.connect(self.on_pca_changed)
        pca_layout.addWidget(self.use_pca_cb)
        pca_comp_layout = QHBoxLayout()
        pca_comp_layout.addWidget(QLabel("Components:"))
        self.pca_components_input = QLineEdit("2")
        self.pca_components_input.setEnabled(False)
        pca_comp_layout.addWidget(self.pca_components_input)
        pca_comp_layout.addWidget(QLabel("(0 for all)"))
        pca_comp_layout.addStretch()
        pca_layout.addLayout(pca_comp_layout)
        main_layout.addWidget(pca_frame)

        param_frame = QGroupBox("Parameter Grid")
        param_layout = QVBoxLayout(param_frame)
        self.use_default_grid_cb = QCheckBox("Use default parameter grid")
        self.use_default_grid_cb.setChecked(True)
        self.use_default_grid_cb.stateChanged.connect(self.on_param_grid_changed)
        param_layout.addWidget(self.use_default_grid_cb)
        self.custom_param_btn = QPushButton("Edit Custom Parameters")
        self.custom_param_btn.setEnabled(False)
        self.custom_param_btn.clicked.connect(self.edit_custom_parameters)
        param_layout.addWidget(self.custom_param_btn)
        main_layout.addWidget(param_frame)

        main_layout.addStretch()

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel)
        bottom_layout.addWidget(ok_btn)
        bottom_layout.addWidget(cancel_btn)
        main_layout.addLayout(bottom_layout)

        self.update_model_options("regression")

    def on_pca_changed(self, state):
        self.pca_components_input.setEnabled(state == Qt.CheckState.Checked.value)

    def on_param_grid_changed(self, state):
        self.custom_param_btn.setEnabled(state == Qt.CheckState.Unchecked.value)

    def update_model_options(self, problem_type=None):
        if problem_type == "classification":
            models = CLASSIFICATION_MODELS
            self.classification_rb.setChecked(True)
            self.regression_rb.setChecked(False)
        else:
            models = REGRESSION_MODELS
            self.regression_rb.setChecked(True)
            self.classification_rb.setChecked(False)

        self.model_combo.clear()
        self.model_combo.addItems(models)
        if models:
            self.model_combo.setCurrentText(models[0])

    def edit_custom_parameters(self):
        model_name = self.model_combo.currentText()
        default_params = DEFAULT_PARAM_GRIDS.get(model_name, {})

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Custom Parameters for {model_name}")
        dialog.setMinimumSize(450, 350)

        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Enter parameters in JSON format. Example:"))
        layout.addWidget(QLabel("{'n_estimators': [50, 100], 'max_depth': [5, 10]}"))

        default_frame = QGroupBox("Default Parameters (for reference)")
        default_layout = QVBoxLayout(default_frame)
        default_text = QTextEdit()
        default_text.setReadOnly(True)
        text = "Default parameter grid:\n"
        for key, value in default_params.items():
            text += f"  {key}: {value}\n"
        default_text.setPlainText(text)
        default_layout.addWidget(default_text)
        layout.addWidget(default_frame)

        custom_frame = QGroupBox("Custom Parameters")
        custom_layout = QVBoxLayout(custom_frame)
        self.custom_param_text = QTextEdit()
        if self.custom_param_grid:
            import json
            self.custom_param_text.setPlainText(json.dumps(self.custom_param_grid, indent=2))
        custom_layout.addWidget(self.custom_param_text)
        layout.addWidget(custom_frame)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(lambda: self.save_custom_parameters(dialog))
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

    def save_custom_parameters(self, dialog):
        import json
        try:
            param_text = self.custom_param_text.toPlainText().strip()
            if param_text:
                self.custom_param_grid = json.loads(param_text)
            else:
                self.custom_param_grid = {}
            dialog.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON format: {str(e)}")

    def load_initial_config(self):
        if 'problem_type' in self.initial_config:
            self.update_model_options(self.initial_config['problem_type'])
            if self.initial_config['problem_type'] == 'classification':
                self.classification_rb.setChecked(True)
                self.regression_rb.setChecked(False)
        if 'model_name' in self.initial_config:
            self.model_combo.setCurrentText(self.initial_config['model_name'])
        if 'x_scaling' in self.initial_config:
            self.x_scaling_combo.setCurrentText(self.initial_config['x_scaling'])
        if 'y_scaling' in self.initial_config:
            self.y_scaling_combo.setCurrentText(self.initial_config['y_scaling'])
        if 'use_pca' in self.initial_config:
            self.use_pca_cb.setChecked(self.initial_config['use_pca'])
            self.pca_components_input.setEnabled(self.initial_config['use_pca'])
        if 'pca_components' in self.initial_config:
            self.pca_components_input.setText(str(self.initial_config['pca_components']))
        if 'param_grid' in self.initial_config and self.initial_config['param_grid']:
            self.use_default_grid_cb.setChecked(False)
            self.custom_param_grid = self.initial_config['param_grid']

    def ok(self):
        problem_type = "classification" if self.classification_rb.isChecked() else "regression"
        model_name = self.model_combo.currentText()

        if not model_name:
            QMessageBox.warning(self, "Warning", "Please select a model!")
            return

        param_grid = {}
        if not self.use_default_grid_cb.isChecked() and self.custom_param_grid:
            param_grid = self.custom_param_grid
        else:
            param_grid = DEFAULT_PARAM_GRIDS.get(model_name, {})

        self.result = {
            'problem_type': problem_type,
            'model_name': model_name,
            'model': MODEL_MAP[model_name](),
            'x_scaling': self.x_scaling_combo.currentText(),
            'y_scaling': self.y_scaling_combo.currentText(),
            'use_pca': self.use_pca_cb.isChecked(),
            'pca_components': self.pca_components_input.text(),
            'param_grid': param_grid,
        }

        self.close()

    def cancel(self):
        self.result = None
        self.close()

    def exec(self):
        super().exec()
        return self.result


class EditConfigurationDialog(QDialog):
    """Edit configuration dialog with tabs."""

    def __init__(self, parent, data, config, index):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Configuration {index + 1}")
        self.setMinimumSize(650, 550)
        self.data = data
        self.config = config.copy()
        self.index = index
        self.result = None

        self.init_ui()
        self.center_window()

    def center_window(self):
        self.adjustSize()
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)

        feature_tab = QWidget()
        self.create_feature_tab(feature_tab)
        self.notebook.addTab(feature_tab, "Features")

        model_tab = QWidget()
        self.create_model_tab(model_tab)
        self.notebook.addTab(model_tab, "Model")

        preview_tab = QWidget()
        self.create_preview_tab(preview_tab)
        self.notebook.addTab(preview_tab, "Preview")

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel)
        bottom_layout.addWidget(save_btn)
        bottom_layout.addWidget(cancel_btn)
        main_layout.addLayout(bottom_layout)

    def create_feature_tab(self, parent):
        layout = QVBoxLayout(parent)

        x_frame = QGroupBox("X Features")
        x_layout = QVBoxLayout(x_frame)
        self.x_text = QTextEdit()
        self.x_text.setReadOnly(True)
        self.x_text.setMaximumHeight(120)
        x_layout.addWidget(self.x_text)
        x_btn_layout = QHBoxLayout()
        x_btn_layout.addStretch()
        self.edit_x_btn = QPushButton("Edit X Features")
        self.edit_x_btn.clicked.connect(self.edit_x_features)
        x_btn_layout.addWidget(self.edit_x_btn)
        x_layout.addLayout(x_btn_layout)
        layout.addWidget(x_frame)

        y_frame = QGroupBox("Y Feature")
        y_layout = QVBoxLayout(y_frame)
        self.y_text = QTextEdit()
        self.y_text.setReadOnly(True)
        self.y_text.setMaximumHeight(60)
        y_layout.addWidget(self.y_text)
        y_btn_layout = QHBoxLayout()
        y_btn_layout.addStretch()
        self.edit_y_btn = QPushButton("Edit Y Feature")
        self.edit_y_btn.clicked.connect(self.edit_y_feature)
        y_btn_layout.addWidget(self.edit_y_btn)
        y_layout.addLayout(y_btn_layout)
        layout.addWidget(y_frame)

        layout.addStretch()

        self.update_feature_text()

    def create_model_tab(self, parent):
        layout = QVBoxLayout(parent)

        info_frame = QGroupBox("Current Model Configuration")
        info_layout = QVBoxLayout(info_frame)
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        info_layout.addWidget(self.model_info_text)
        layout.addWidget(info_frame)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.edit_model_btn = QPushButton("Edit Model Configuration")
        self.edit_model_btn.clicked.connect(self.edit_model_config)
        btn_layout.addWidget(self.edit_model_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        self.update_model_text()

    def create_preview_tab(self, parent):
        layout = QVBoxLayout(parent)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        layout.addWidget(self.preview_text)
        self.update_preview_text()

    def update_feature_text(self):
        if 'x_cols' in self.config:
            text = f"Selected {len(self.config['x_cols'])} X features:\n"
            for i, feat in enumerate(self.config['x_cols']):
                text += f"{i+1}. {feat}\n"
        else:
            text = "No X features selected"
        self.x_text.setPlainText(text)

        if 'y_col' in self.config:
            self.y_text.setPlainText(f"Selected Y feature: {self.config['y_col']}")
        else:
            self.y_text.setPlainText("No Y feature selected")

    def update_model_text(self):
        if 'model_config' in self.config:
            mc = self.config['model_config']
            text = f"Problem Type: {mc.get('problem_type', 'N/A')}\n"
            text += f"Model: {mc.get('model_name', 'N/A')}\n"
            text += f"X Scaling: {mc.get('x_scaling', 'N/A')}\n"
            text += f"Y Scaling: {mc.get('y_scaling', 'N/A')}\n"
            text += f"PCA: {'Yes' if mc.get('use_pca') else 'No'}\n"
            if mc.get('use_pca'):
                text += f"PCA Components: {mc.get('pca_components', 'N/A')}\n"
            param_grid = mc.get('param_grid', {})
            if param_grid:
                text += f"\nParameter Grid: {len(param_grid)} parameters\n"
                for key, value in param_grid.items():
                    text += f"  {key}: {value}\n"
        else:
            text = "No model configuration found"
        self.model_info_text.setPlainText(text)

    def update_preview_text(self):
        lines = ["=" * 60, "CONFIGURATION PREVIEW", "=" * 60, ""]
        lines.append("FEATURES:")
        lines.append("-" * 30)

        if 'x_cols' in self.config:
            lines.append(f"X Features: {len(self.config['x_cols'])} features")
            if len(self.config['x_cols']) <= 5:
                lines.append(f"  {', '.join(self.config['x_cols'])}")
            else:
                lines.append(f"  {', '.join(self.config['x_cols'][:3])} ... ({len(self.config['x_cols']) - 3} more)")
        else:
            lines.append("X Features: Not configured")

        if 'y_col' in self.config:
            lines.append(f"Y Feature: {self.config['y_col']}")
        else:
            lines.append("Y Feature: Not configured")

        lines.extend(["", "MODEL CONFIGURATION:", "-" * 30])

        if 'model_config' in self.config:
            mc = self.config['model_config']
            lines.append(f"Problem Type: {mc.get('problem_type', 'N/A')}")
            lines.append(f"Model: {mc.get('model_name', 'N/A')}")
            lines.append(f"X Scaling: {mc.get('x_scaling', 'N/A')}")
            lines.append(f"Y Scaling: {mc.get('y_scaling', 'N/A')}")
            lines.append(f"PCA: {'Yes' if mc.get('use_pca') else 'No'}")
            if mc.get('use_pca'):
                lines.append(f"PCA Components: {mc.get('pca_components', 'N/A')}")
            param_grid = mc.get('param_grid', {})
            if param_grid:
                lines.append(f"Parameter Grid: {len(param_grid)} parameters")
        else:
            lines.append("Model Configuration: Not configured")

        lines.extend(["", "=" * 60])

        self.preview_text.setPlainText("\n".join(lines))

    def edit_x_features(self):
        dialog = FeatureSelectionDialog(
            self, "Edit X Features (Multiple)",
            list(self.data.columns),
            multiselect=True,
            initial_selection=self.config.get('x_cols', [])
        )
        if dialog.exec():
            self.config['x_cols'] = dialog.get_selected_features()
            self.update_feature_text()
            self.update_preview_text()

    def edit_y_feature(self):
        dialog = FeatureSelectionDialog(
            self, "Edit Y Feature (Single)",
            list(self.data.columns),
            multiselect=False,
            initial_selection=[self.config['y_col']] if self.config.get('y_col') else []
        )
        if dialog.exec():
            selected = dialog.get_selected_features()
            if selected:
                self.config['y_col'] = selected[0]
                self.update_feature_text()
                self.update_preview_text()

    def edit_model_config(self):
        initial = self.config.get('model_config', {})
        dialog = ModelConfigDialog(self, initial)
        result = dialog.exec()
        if result:
            self.config['model_config'] = result
            self.update_model_text()
            self.update_preview_text()

    def save(self):
        if 'x_cols' not in self.config or not self.config['x_cols']:
            QMessageBox.warning(self, "Warning", "Please select X features!")
            return
        if 'y_col' not in self.config or not self.config['y_col']:
            QMessageBox.warning(self, "Warning", "Please select Y feature!")
            return
        if 'model_config' not in self.config:
            QMessageBox.warning(self, "Warning", "Please configure the model!")
            return
        if self.config['y_col'] in self.config['x_cols']:
            QMessageBox.warning(self, "Warning", "Y feature cannot be the same as X features!")
            return

        x_count = len(self.config['x_cols'])
        y_feature = self.config['y_col']
        model_name = self.config['model_config'].get('model_name', 'Unknown')
        self.config['description'] = f"X[{x_count}] → Y[{y_feature}] | Model: {model_name}"

        self.result = self.config
        self.close()

    def cancel(self):
        self.result = None
        self.close()


class TrainingThread(QThread):
    """Background thread for model training."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, data, feature_config, model_config, param_grid=None):
        super().__init__()
        self.data = data
        self.feature_config = feature_config
        self.model_config = model_config
        self.param_grid = param_grid

    def run(self):
        try:
            self.progress.emit(10, "Preparing data...")
            results = train_model_with_config(
                self.data, self.feature_config, self.model_config, self.param_grid
            )
            self.progress.emit(100, "Complete")
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MLTrainerApp(QMainWindow):
    """Main ML Trainer GUI application - matching Tkinter layout."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Trainer with SHAP")
        self.setMinimumSize(1200, 700)

        self.data = None
        self.features_list = []
        self.current_results = {}

        self.init_ui()
        self.update_status()

        if not SHAP_AVAILABLE:
            QMessageBox.warning(
                self, "SHAP Warning",
                "SHAP library is not installed. Feature importance analysis will be limited.\n"
                "Install it using: pip install shap"
            )

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.create_toolbar(main_layout)

        content_frame = QWidget()
        content_layout = QHBoxLayout(content_frame)

        left_panel = self.create_features_panel()
        right_panel = self.create_operations_panel()

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        content_layout.addWidget(splitter)
        main_layout.addWidget(content_frame)

    def create_toolbar(self, parent):
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 5)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setFixedWidth(100)
        self.load_btn.clicked.connect(self.load_data)
        toolbar_layout.addWidget(self.load_btn)

        self.preview_btn = QPushButton("Preview Data")
        self.preview_btn.setFixedWidth(100)
        self.preview_btn.clicked.connect(self.preview_data)
        toolbar_layout.addWidget(self.preview_btn)

        self.stats_btn = QPushButton("Data Stats")
        self.stats_btn.setFixedWidth(100)
        self.stats_btn.clicked.connect(self.show_data_stats)
        toolbar_layout.addWidget(self.stats_btn)

        toolbar_layout.addStretch()

        shap_status = "Available" if SHAP_AVAILABLE else "Not Available"
        self.shap_label = QLabel(f"SHAP: {shap_status}")
        if SHAP_AVAILABLE:
            self.shap_label.setStyleSheet("color: green;")
        else:
            self.shap_label.setStyleSheet("color: red;")
        toolbar_layout.addWidget(self.shap_label)

        self.status_label = QLabel("No data loaded")
        self.status_label.setStyleSheet("color: gray;")
        toolbar_layout.addWidget(self.status_label)

        parent.addWidget(toolbar)

    def create_features_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        features_frame = QGroupBox("Feature Configurations")
        features_layout = QVBoxLayout(features_frame)

        list_frame = QFrame()
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)

        self.features_listbox = QListWidget()
        self.features_listbox.itemDoubleClicked.connect(self.edit_feature_config)
        list_layout.addWidget(self.features_listbox)

        features_layout.addWidget(list_frame)

        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 5, 0, 0)

        add_btn = QPushButton("Add Configuration")
        add_btn.clicked.connect(self.add_feature_config)
        edit_btn = QPushButton("Edit Selected")
        edit_btn.clicked.connect(self.edit_feature_config)
        dup_btn = QPushButton("Duplicate Selected")
        dup_btn.clicked.connect(self.duplicate_feature_config)
        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self.delete_feature_config)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_feature_configs)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(dup_btn)
        btn_layout.addWidget(del_btn)
        btn_layout.addWidget(clear_btn)

        for btn in [add_btn, edit_btn, dup_btn, del_btn, clear_btn]:
            btn.setFixedWidth(95)

        features_layout.addWidget(btn_frame)
        layout.addWidget(features_frame)

        return widget

    def create_operations_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        train_frame = QGroupBox("Model Training")
        train_layout = QVBoxLayout(train_frame)

        btn_layout = QHBoxLayout()
        self.train_selected_btn = QPushButton("Train Selected Configuration")
        self.train_selected_btn.clicked.connect(self.train_selected)
        self.train_all_btn = QPushButton("Train All Configurations")
        self.train_all_btn.clicked.connect(self.train_all)
        btn_layout.addWidget(self.train_selected_btn)
        btn_layout.addWidget(self.train_all_btn)
        train_layout.addLayout(btn_layout)

        results_frame = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_frame)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        train_layout.addWidget(results_frame)

        viz_layout = QHBoxLayout()

        basic_viz_frame = QFrame()
        basic_viz_layout = QHBoxLayout(basic_viz_frame)
        basic_viz_layout.setContentsMargins(0, 0, 0, 0)
        plot_results_btn = QPushButton("Plot Results")
        plot_results_btn.clicked.connect(self.plot_results)
        plot_learning_btn = QPushButton("Plot Learning Curve")
        plot_learning_btn.clicked.connect(self.plot_learning_curve)
        basic_viz_layout.addWidget(plot_results_btn)
        basic_viz_layout.addWidget(plot_learning_btn)
        viz_layout.addWidget(basic_viz_frame)

        shap_viz_frame = QFrame()
        shap_viz_layout = QHBoxLayout(shap_viz_frame)
        shap_viz_layout.setContentsMargins(0, 0, 0, 0)
        shap_btn = QPushButton("SHAP Analysis")
        shap_btn.clicked.connect(self.show_shap_analysis)
        save_btn = QPushButton("Save Results")
        save_btn.clicked.connect(self.save_results)
        shap_viz_layout.addWidget(shap_btn)
        shap_viz_layout.addWidget(save_btn)
        viz_layout.addWidget(shap_viz_frame)

        train_layout.addLayout(viz_layout)
        layout.addWidget(train_frame)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return widget

    def update_status(self):
        if self.data is None:
            self.status_label.setText("No data loaded")
            self.status_label.setStyleSheet("color: gray;")
        else:
            self.status_label.setText(f"Data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            self.status_label.setStyleSheet("color: black;")

        self.features_listbox.clear()
        for i, feat in enumerate(self.features_list):
            desc = feat.get('description', f'Configuration {i+1}')
            self.features_listbox.addItem(desc)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", "", "CSV Files (*.csv);;All Files (*.*)"
        )
        if path:
            try:
                self.data = pd.read_csv(path)
                self.features_list = []
                self.update_status()
                QMessageBox.information(self, "Success", f"Data loaded successfully!\nShape: {self.data.shape}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def preview_data(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Data Preview")
        dialog.setMinimumSize(800, 500)

        layout = QVBoxLayout(dialog)

        tree = QTreeWidget()
        tree.setColumnCount(len(self.data.columns))
        tree.setHeaderLabels(list(self.data.columns))
        for i, row in self.data.head(100).iterrows():
            item = QTreeWidgetItem([str(v) for v in row])
            tree.addTopLevelItem(item)
        layout.addWidget(tree)

        layout.addWidget(QLabel(f"Showing first 100 rows of {len(self.data)} total"))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_data_stats(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Data Statistics")
        dialog.setMinimumSize(550, 450)

        layout = QVBoxLayout(dialog)

        text = QTextEdit()
        text.setReadOnly(True)

        stats = "Data Information:\n"
        stats += f"Shape: {self.data.shape}\n"
        stats += f"Columns: {len(self.data.columns)}\n"
        stats += f"Rows: {len(self.data)}\n\n"

        stats += "Data Types:\n"
        for col in self.data.columns:
            stats += f"  {col}: {self.data[col].dtype}\n"
        stats += "\n"

        stats += "Missing Values:\n"
        missing = self.data.isnull().sum()
        has_missing = False
        for col in self.data.columns:
            if missing[col] > 0:
                has_missing = True
                stats += f"  {col}: {missing[col]} ({missing[col]/len(self.data):.2%})\n"
        if not has_missing:
            stats += "  No missing values\n"

        stats += "\nNumeric Columns Summary:\n"
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats += self.data[numeric_cols].describe().to_string()
        else:
            stats += "No numeric columns"

        text.setPlainText(stats)
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def add_feature_config(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return

        x_dialog = FeatureSelectionDialog(
            self, "Select X Features (Multiple)",
            list(self.data.columns), multiselect=True
        )
        x_features = x_dialog.exec()
        if not x_features:
            return
        x_features = x_dialog.get_selected_features()

        y_dialog = FeatureSelectionDialog(
            self, "Select Y Feature (Single)",
            list(self.data.columns), multiselect=False
        )
        y_result = y_dialog.exec()
        if not y_result:
            return
        y_features = y_dialog.get_selected_features()
        y_feature = y_features[0] if y_features else None

        if not y_feature:
            return

        if y_feature in x_features:
            QMessageBox.warning(self, "Warning", "Y feature cannot be the same as X features!")
            return

        model_dialog = ModelConfigDialog(self)
        model_config = model_dialog.exec()
        if not model_config:
            return

        feature_config = {
            'x_cols': x_features,
            'y_col': y_feature,
            'use_pca': model_config['use_pca'],
            'pca_components': model_config['pca_components'],
            'x_scaling': model_config['x_scaling'],
            'y_scaling': model_config['y_scaling'],
            'model_config': model_config,
            'description': f"X[{len(x_features)}] → Y[{y_feature}] | Model: {model_config['model_name']}"
        }

        self.features_list.append(feature_config)
        self.update_status()

    def edit_feature_config(self, item=None):
        if item is None:
            current_row = self.features_listbox.currentRow()
            if current_row < 0:
                QMessageBox.warning(self, "Warning", "Please select a configuration to edit!")
                return
        else:
            current_row = self.features_listbox.row(item)

        if current_row >= len(self.features_list):
            return

        current_config = self.features_list[current_row]
        edit_dialog = EditConfigurationDialog(self, self.data, current_config, current_row)
        edited_config = edit_dialog.exec()
        if edited_config:
            self.features_list[current_row] = edited_config
            self.update_status()
            QMessageBox.information(self, "Success", "Configuration updated successfully!")

    def duplicate_feature_config(self):
        current_row = self.features_listbox.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a configuration to duplicate!")
            return

        if current_row >= len(self.features_list):
            return

        current_config = self.features_list[current_row]
        duplicated_config = copy.deepcopy(current_config)
        if 'description' in duplicated_config:
            duplicated_config['description'] = f"Copy of {duplicated_config['description']}"
        self.features_list.append(duplicated_config)
        self.update_status()
        QMessageBox.information(self, "Success", "Configuration duplicated successfully!")

    def delete_feature_config(self):
        current_row = self.features_listbox.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a configuration to delete!")
            return

        if current_row < len(self.features_list):
            self.features_list.pop(current_row)
            self.update_status()

    def clear_feature_configs(self):
        reply = QMessageBox.question(
            self, "Confirm",
            "Are you sure you want to clear all configurations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.features_list = []
            self.update_status()

    def train_selected(self):
        current_row = self.features_listbox.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a configuration to train!")
            return

        if current_row >= len(self.features_list):
            return

        feature_config = self.features_list[current_row]
        model_config = feature_config['model_config']

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.train_selected_btn.setEnabled(False)
        self.train_all_btn.setEnabled(False)

        self.training_thread = TrainingThread(
            self.data, feature_config, model_config, model_config['param_grid']
        )
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.start()

    def train_all(self):
        if not self.features_list:
            QMessageBox.warning(self, "Warning", "No configurations to train!")
            return

        reply = QMessageBox.question(
            self, "Confirm",
            f"Train all {len(self.features_list)} configurations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.results_text.clear()
        self.results_text.append("Batch Training Results:\n")
        self.results_text.append("=" * 50 + "\n\n")

        for i, feature_config in enumerate(self.features_list):
            self.results_text.append(f"Configuration {i+1}: {feature_config['description']}")

            try:
                model_config = feature_config['model_config']
                results = train_model_with_config(
                    self.data, feature_config, model_config, model_config['param_grid']
                )

                if results['success']:
                    if model_config['problem_type'] == 'regression':
                        self.results_text.append(
                            f"  MSE: {results['mse']:.4f}, "
                            f"MAE: {results['mae']:.4f}, "
                            f"R²: {results['r2']:.4f}\n"
                        )
                    else:
                        self.results_text.append(
                            f"  Accuracy: {results['accuracy']:.4f}, "
                            f"F1: {results['f1']:.4f}\n"
                        )
                    if 'best_params' in results:
                        self.results_text.append(f"  Best Params: {results['best_params']}\n")
                else:
                    self.results_text.append(f"  Error: {results['error']}\n")

            except Exception as e:
                self.results_text.append(f"  Error: {str(e)}\n")

            self.results_text.append("\n")

        self.results_text.append("Batch training completed!\n")

    def on_training_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)

    def on_training_finished(self, results):
        self.progress_bar.setVisible(False)
        self.train_selected_btn.setEnabled(True)
        self.train_all_btn.setEnabled(True)
        self.statusBar().showMessage("Training complete")
        self.current_results = results

        current_row = self.features_listbox.currentRow()
        if current_row >= 0 and current_row < len(self.features_list):
            feature_config = self.features_list[current_row]
            model_config = feature_config['model_config']
            self.current_results['feature_config'] = feature_config
            self.current_results['model_config'] = model_config

        if results.get('success'):
            self.display_results(results)
        else:
            QMessageBox.critical(self, "Error", f"Training failed: {results.get('error', 'Unknown error')}")

    def on_training_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.train_selected_btn.setEnabled(True)
        self.train_all_btn.setEnabled(True)
        self.statusBar().showMessage("Training failed")
        QMessageBox.critical(self, "Error", f"Training error: {error_msg}")

    def display_results(self, results):
        self.results_text.clear()

        if not results['success']:
            self.results_text.append(f"Training failed: {results['error']}")
            return

        self.results_text.append("Training Results:\n")
        self.results_text.append("=" * 50 + "\n\n")

        feature_config = self.current_results['feature_config']
        model_config = self.current_results['model_config']

        self.results_text.append(f"Configuration: {feature_config['description']}\n")
        self.results_text.append(f"X Features: {len(feature_config['x_cols'])} features\n")
        self.results_text.append(f"Y Feature: {feature_config['y_col']}\n")
        self.results_text.append(f"Model: {model_config['model_name']}\n")
        self.results_text.append(f"Problem Type: {model_config['problem_type']}\n")

        if feature_config['use_pca']:
            self.results_text.append(f"PCA: Yes ({feature_config['pca_components']} components)\n")
            if 'pca_explained_variance' in results:
                self.results_text.append(f"PCA Explained Variance: {results['pca_explained_variance']:.3f}\n")
        else:
            self.results_text.append("PCA: No\n")

        self.results_text.append(f"X Scaling: {feature_config['x_scaling']}\n")
        if model_config['problem_type'] == 'regression':
            self.results_text.append(f"Y Scaling: {feature_config['y_scaling']}\n")

        self.results_text.append("\n" + "=" * 50 + "\n\n")

        if model_config['problem_type'] == 'regression':
            self.results_text.append("Regression Metrics:\n")
            self.results_text.append(f"  MSE: {results['mse']:.4f}\n")
            self.results_text.append(f"  MAE: {results['mae']:.4f}\n")
            self.results_text.append(f"  R² Score: {results['r2']:.4f}\n")
        else:
            self.results_text.append("Classification Metrics:\n")
            self.results_text.append(f"  Accuracy: {results['accuracy']:.4f}\n")
            self.results_text.append(f"  F1 Score: {results['f1']:.4f}\n")

        if 'best_params' in results:
            self.results_text.append("\nGrid Search Results:\n")
            self.results_text.append(f"  Best Parameters: {results['best_params']}\n")
            self.results_text.append(f"  Best Score: {results['best_score']:.4f}\n")

        self.results_text.append("\n" + "=" * 50 + "\n\n")
        self.results_text.append("FEATURE IMPORTANCE ANALYSIS\n")
        self.results_text.append("-" * 30 + "\n\n")

        self.results_text.append(f"SHAP: {'Available' if results.get('shap_available', False) else 'Not Available'}\n")

        if 'feature_importance' in results and results['feature_importance']:
            for method, importance_dict in results['feature_importance'].items():
                if importance_dict and method != 'shap':
                    self.results_text.append(f"\n{method.replace('_', ' ').title()}:\n")
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    for feature, imp in sorted_importance[:10]:
                        self.results_text.append(f"  {feature}: {imp:.4f}\n")

        if results.get('shap_importance'):
            self.results_text.append("\nSHAP Feature Importance (Top 10):\n")
            sorted_shap = sorted(results['shap_importance'].items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_shap[:10]:
                self.results_text.append(f"  {feature}: {imp:.4f}\n")

            if feature_config['use_pca'] and 'shap_importance_original' in results:
                self.results_text.append("\nSHAP Importance (Original Features, Top 10):\n")
                sorted_original = sorted(results['shap_importance_original'].items(), key=lambda x: x[1], reverse=True)
                for feature, imp in sorted_original[:10]:
                    self.results_text.append(f"  {feature}: {imp:.4f}\n")

        if 'feature_importance_error' in results:
            self.results_text.append(f"\nFeature Importance Error: {results['feature_importance_error']}\n")
        if 'shap_error' in results:
            self.results_text.append(f"\nSHAP Error: {results['shap_error']}\n")

        self.results_text.append("\n" + "=" * 50 + "\n")
        self.results_text.append("\nTraining completed successfully!\n")
        self.results_text.append("\nClick 'SHAP Analysis' for detailed feature importance visualizations.\n")

    def plot_results(self):
        if not self.current_results or not self.current_results.get('success'):
            QMessageBox.warning(self, "Warning", "No training results to plot!")
            return

        model_config = self.current_results['model_config']
        feature_config = self.current_results['feature_config']
        title = f"{model_config['model_name']} - {feature_config['description']}"

        try:
            if model_config['problem_type'] == 'regression':
                plot_regression_results(self.current_results['y_test'], self.current_results['y_pred'], title)
            else:
                plot_classification_results(self.current_results['y_test'], self.current_results['y_pred'], title)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot results: {str(e)}")

    def plot_learning_curve(self):
        if not self.current_results or not self.current_results.get('success'):
            QMessageBox.warning(self, "Warning", "No training results to plot!")
            return

        feature_config = self.current_results['feature_config']
        model_config = self.current_results['model_config']

        X = self.data[feature_config['x_cols']].values
        y = self.data[feature_config['y_col']].values

        x_scaler = create_scaler(feature_config['x_scaling'])
        X_scaled = x_scaler.fit_transform(X) if x_scaler else X

        if model_config['problem_type'] == 'regression' and y.ndim == 1:
            y = y.reshape(-1, 1)

        y_scaler = create_scaler(feature_config['y_scaling']) if model_config['problem_type'] == 'regression' else None
        y_scaled = y_scaler.fit_transform(y) if y_scaler and model_config['problem_type'] == 'regression' else y
        y_flat = y_scaled.ravel() if len(y_scaled.shape) > 1 and y_scaled.shape[1] == 1 else y_scaled

        title = f"{model_config['model_name']} - Learning Curve"
        try:
            plot_learning_curve(model_config['model'], X_scaled, y_flat, title, model_config['problem_type'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot learning curve: {str(e)}")

    def show_shap_analysis(self):
        if not self.current_results or not self.current_results.get('success'):
            QMessageBox.warning(self, "Warning", "No training results available for SHAP analysis!")
            return

        if not SHAP_AVAILABLE:
            QMessageBox.warning(
                self, "SHAP Not Available",
                "SHAP library is not installed. Please install it using: pip install shap"
            )
            return

        if 'shap_values' not in self.current_results:
            QMessageBox.information(
                self, "No SHAP Values",
                "SHAP values were not calculated for this model. Try training with a different model."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("SHAP Feature Importance Analysis")
        dialog.setMinimumSize(500, 450)

        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("<h3>SHAP Analysis Options</h3>"))

        if self.current_results['feature_config']['use_pca']:
            feature_names = self.current_results.get('pca_feature_names', [])
        else:
            feature_names = self.current_results.get('original_feature_names', [])

        buttons_frame = QFrame()
        buttons_layout = QVBoxLayout(buttons_frame)

        summary_btn = QPushButton("SHAP Summary Plot")
        summary_btn.clicked.connect(lambda: self.plot_shap_visualization('summary'))
        buttons_layout.addWidget(summary_btn)

        bar_btn = QPushButton("SHAP Feature Importance Bar Plot")
        bar_btn.clicked.connect(lambda: self.plot_shap_visualization('bar'))
        buttons_layout.addWidget(bar_btn)

        if 'feature_importance' in self.current_results:
            for method in self.current_results['feature_importance']:
                if method != 'shap' and self.current_results['feature_importance'][method]:
                    m = method
                    btn = QPushButton(f"Model Internal Importance ({method})")
                    btn.clicked.connect(lambda _, m=m: self.plot_model_internal_importance_viz(m))
                    buttons_layout.addWidget(btn)

        dependence_frame = QGroupBox("SHAP Dependence Plot")
        dependence_layout = QHBoxLayout(dependence_frame)
        dependence_layout.addWidget(QLabel("Select feature:"))

        self.dependence_combo = QComboBox()
        self.dependence_combo.addItems(feature_names if feature_names else [""])
        dependence_layout.addWidget(self.dependence_combo)

        dep_btn = QPushButton("Plot Dependence")
        dep_btn.clicked.connect(lambda: self.plot_shap_visualization('dependence'))
        dependence_layout.addWidget(dep_btn)
        buttons_layout.addWidget(dependence_frame)

        importance_dicts = {}
        if 'feature_importance' in self.current_results:
            for method, imp_dict in self.current_results['feature_importance'].items():
                if imp_dict and method != 'shap':
                    importance_dicts[method] = imp_dict
        if self.current_results.get('shap_importance'):
            importance_dicts['shap'] = self.current_results['shap_importance']
        if len(importance_dicts) > 1:
            compare_btn = QPushButton("Compare Feature Importance Methods")
            compare_btn.clicked.connect(lambda: self.plot_shap_visualization('comparison'))
            buttons_layout.addWidget(compare_btn)

        layout.addWidget(buttons_frame)

        layout.addWidget(QLabel("Note: Each plot will open in a separate matplotlib window."))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def plot_shap_visualization(self, plot_type):
        if not self.current_results or 'shap_values' not in self.current_results:
            QMessageBox.warning(self, "Warning", "SHAP values not available!")
            return

        from core import plot_shap_summary, plot_shap_bar, plot_shap_dependence, plot_feature_importance_comparison

        if self.current_results['feature_config']['use_pca']:
            feature_names = self.current_results.get('pca_feature_names', [])
        else:
            feature_names = self.current_results.get('original_feature_names', [])

        model_name = self.current_results['model_config']['model_name']

        try:
            success = False
            if plot_type == 'summary':
                title = f"SHAP Summary - {model_name}"
                success = plot_shap_summary(self.current_results['shap_values'], feature_names, title)
            elif plot_type == 'bar':
                if self.current_results.get('shap_importance'):
                    title = f"SHAP Feature Importance - {model_name}"
                    success = plot_shap_bar(self.current_results['shap_importance'], title)
                else:
                    QMessageBox.warning(self, "Warning", "SHAP importance data not available!")
                    return
            elif plot_type == 'dependence':
                feature_name = self.dependence_combo.currentText()
                if feature_name in feature_names:
                    feature_index = feature_names.index(feature_name)
                    title = f"SHAP Dependence Plot - {feature_name}"
                    success = plot_shap_dependence(
                        self.current_results['shap_values'], feature_names, feature_index,
                        self.current_results.get('X_test'), title
                    )
                else:
                    QMessageBox.warning(self, "Warning", f"Feature '{feature_name}' not found!")
                    return
            elif plot_type == 'comparison':
                importance_dicts = {}
                if 'feature_importance' in self.current_results:
                    for method, imp_dict in self.current_results['feature_importance'].items():
                        if imp_dict and method != 'shap':
                            importance_dicts[method] = imp_dict
                if self.current_results.get('shap_importance'):
                    importance_dicts['shap'] = self.current_results['shap_importance']
                if len(importance_dicts) >= 2:
                    title = f"Feature Importance Comparison - {model_name}"
                    success = plot_feature_importance_comparison(importance_dicts, title)
                else:
                    QMessageBox.warning(self, "Warning", "Need at least 2 importance methods to compare!")
                    return

            if not success:
                QMessageBox.critical(self, "Error", f"Failed to generate {plot_type} plot!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting {plot_type}: {str(e)}")

    def plot_model_internal_importance_viz(self, method):
        from core import plot_model_internal_importance

        if (not self.current_results or
            'feature_importance' not in self.current_results or
                method not in self.current_results['feature_importance']):
            QMessageBox.warning(self, "Warning", f"{method} importance data not available!")
            return

        importance_dict = self.current_results['feature_importance'][method]
        if not importance_dict:
            QMessageBox.warning(self, "Warning", f"{method} importance data is empty!")
            return

        model_name = self.current_results['model_config']['model_name']
        title = f"{method.replace('_', ' ').title()} - {model_name}"

        try:
            success = plot_model_internal_importance(importance_dict, title)
            if not success:
                QMessageBox.critical(self, "Error", f"Failed to generate {method} importance plot!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting {method} importance: {str(e)}")

    def save_results(self):
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "Text Files (*.txt);;All Files (*.*)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.toPlainText())
                QMessageBox.information(self, "Success", f"Results saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")


def main():
    """Run the application."""
    app = QApplication(sys.argv)
    window = MLTrainerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

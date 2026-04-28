"""Tkinter GUI for PyMLUI."""

import copy

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import pandas as pd
import numpy as np

from core import (
    train_model_with_config,
    SHAP_AVAILABLE,
    DEFAULT_PARAM_GRIDS,
    MODEL_MAP,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    SCALING_METHODS,
    plot_regression_results,
    plot_classification_results,
    plot_learning_curve,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_dependence,
    plot_feature_importance_comparison,
    plot_model_internal_importance,
)


class FeatureSelectionDialog:
    """Feature selection dialog."""

    def __init__(self, parent, title, features, multiselect=True, initial_selection=None):
        self.parent = parent
        self.features = features
        self.multiselect = multiselect
        self.selected_features = initial_selection.copy() if initial_selection else []

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        search_frame = ttk.Frame(control_frame)
        search_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_features)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        search_entry.focus_set()

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(btn_frame, text="Select All", command=self.select_all, width=10).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(btn_frame, text="Clear All", command=self.deselect_all, width=10).pack(side=tk.LEFT)

        listbox_frame = ttk.Frame(main_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        available_frame = ttk.LabelFrame(listbox_frame, text="Available Features", padding=5)
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.available_listbox = tk.Listbox(
            available_frame, selectmode=tk.EXTENDED if self.multiselect else tk.SINGLE
        )
        available_scrollbar = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, command=self.available_listbox.yview)
        self.available_listbox.configure(yscrollcommand=available_scrollbar.set)
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(listbox_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(button_frame, text=">", command=self.select_feature, width=5).pack(pady=2)
        ttk.Button(button_frame, text=">>", command=self.select_all_features, width=5).pack(pady=2)
        ttk.Button(button_frame, text="<", command=self.remove_feature, width=5).pack(pady=2)
        ttk.Button(button_frame, text="<<", command=self.remove_all_features, width=5).pack(pady=2)

        selected_frame = ttk.LabelFrame(listbox_frame, text="Selected Features", padding=5)
        selected_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.selected_listbox = tk.Listbox(selected_frame, selectmode=tk.EXTENDED)
        selected_scrollbar = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        self.selected_listbox.configure(yscrollcommand=selected_scrollbar.set)
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_label = ttk.Label(main_frame, text=f"Total: {len(self.features)} features", foreground="gray")
        self.info_label.pack(anchor=tk.W, pady=(10, 0))

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(bottom_frame, text="OK", command=self.ok, width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(bottom_frame, text="Cancel", command=self.cancel, width=10).pack(side=tk.RIGHT)

        self.update_available_listbox()
        for feature in self.selected_features:
            self.selected_listbox.insert(tk.END, feature)

    def update_available_listbox(self):
        self.available_listbox.delete(0, tk.END)
        for feature in self.features:
            if feature not in self.selected_features:
                self.available_listbox.insert(tk.END, feature)
        self.update_info_label()

    def update_info_label(self):
        self.info_label.config(text=f"Available: {self.available_listbox.size()}, Selected: {len(self.selected_features)}")

    def filter_features(self, *args):
        search_text = self.search_var.get().lower()
        self.available_listbox.delete(0, tk.END)
        for feature in self.features:
            if feature not in self.selected_features and search_text in feature.lower():
                self.available_listbox.insert(tk.END, feature)
        self.update_info_label()

    def select_feature(self):
        selected = self.available_listbox.curselection()
        for idx in selected:
            feature = self.available_listbox.get(idx)
            if feature not in self.selected_features:
                self.selected_features.append(feature)
                self.selected_listbox.insert(tk.END, feature)
        self.update_available_listbox()

    def select_all_features(self):
        for i in range(self.available_listbox.size()):
            feature = self.available_listbox.get(i)
            if feature not in self.selected_features:
                self.selected_features.append(feature)
                self.selected_listbox.insert(tk.END, feature)
        self.update_available_listbox()

    def remove_feature(self):
        selected = self.selected_listbox.curselection()
        for idx in sorted(selected, reverse=True):
            feature = self.selected_listbox.get(idx)
            self.selected_features.remove(feature)
            self.selected_listbox.delete(idx)
        self.update_available_listbox()

    def remove_all_features(self):
        self.selected_features.clear()
        self.selected_listbox.delete(0, tk.END)
        self.update_available_listbox()

    def select_all(self):
        self.available_listbox.selection_set(0, tk.END)

    def deselect_all(self):
        self.available_listbox.selection_clear(0, tk.END)

    def ok(self):
        self.dialog.destroy()

    def cancel(self):
        self.selected_features = []
        self.dialog.destroy()

    def show(self):
        self.dialog.wait_window()
        return self.selected_features


class ModelConfigDialog:
    """Model configuration dialog."""

    def __init__(self, parent, initial_config=None):
        self.parent = parent
        self.initial_config = initial_config
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Model Configuration" if not initial_config else "Edit Configuration")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()
        self.center_window()

        if initial_config:
            self.load_initial_config()

    def center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        problem_frame = ttk.LabelFrame(main_frame, text="Problem Type", padding=10)
        problem_frame.pack(fill=tk.X, pady=(0, 10))

        self.problem_type = tk.StringVar(value="regression")
        ttk.Radiobutton(problem_frame, text="Regression", value="regression",
                        variable=self.problem_type, command=self.update_model_options).pack(anchor=tk.W)
        ttk.Radiobutton(problem_frame, text="Classification", value="classification",
                        variable=self.problem_type, command=self.update_model_options).pack(anchor=tk.W, pady=(5, 0))

        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Select Model:").pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(fill=tk.X, pady=5)

        scaling_frame = ttk.LabelFrame(main_frame, text="Data Scaling", padding=10)
        scaling_frame.pack(fill=tk.X, pady=(0, 10))

        x_scaling_frame = ttk.Frame(scaling_frame)
        x_scaling_frame.pack(fill=tk.X, pady=5)
        ttk.Label(x_scaling_frame, text="X Scaling:").pack(side=tk.LEFT)
        self.x_scaling_method = tk.StringVar(value="standard")
        x_scaling_combo = ttk.Combobox(x_scaling_frame, textvariable=self.x_scaling_method,
                                       values=SCALING_METHODS, state="readonly", width=15)
        x_scaling_combo.pack(side=tk.LEFT, padx=(5, 0))

        y_scaling_frame = ttk.Frame(scaling_frame)
        y_scaling_frame.pack(fill=tk.X)
        ttk.Label(y_scaling_frame, text="Y Scaling (regression only):").pack(side=tk.LEFT)
        self.y_scaling_method = tk.StringVar(value="none")
        y_scaling_combo = ttk.Combobox(y_scaling_frame, textvariable=self.y_scaling_method,
                                       values=SCALING_METHODS, state="readonly", width=15)
        y_scaling_combo.pack(side=tk.LEFT, padx=(5, 0))

        pca_frame = ttk.LabelFrame(main_frame, text="PCA Options", padding=10)
        pca_frame.pack(fill=tk.X, pady=(0, 10))

        self.use_pca = tk.BooleanVar(value=False)
        ttk.Checkbutton(pca_frame, text="Use PCA", variable=self.use_pca).pack(anchor=tk.W)

        pca_params_frame = ttk.Frame(pca_frame)
        pca_params_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(pca_params_frame, text="Components:").pack(side=tk.LEFT)
        self.pca_components = tk.StringVar(value="2")
        pca_entry = ttk.Entry(pca_params_frame, textvariable=self.pca_components, width=10)
        pca_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(pca_params_frame, text="(0 for all components)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT, padx=(5, 0))

        param_frame = ttk.LabelFrame(main_frame, text="Parameter Grid", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        self.use_default_grid = tk.BooleanVar(value=True)
        ttk.Radiobutton(param_frame, text="Use default parameter grid",
                       variable=self.use_default_grid, value=True).pack(anchor=tk.W)
        ttk.Radiobutton(param_frame, text="Use custom parameter grid",
                       variable=self.use_default_grid, value=False).pack(anchor=tk.W, pady=(5, 0))

        self.custom_param_btn = ttk.Button(param_frame, text="Edit Custom Parameters",
                                          command=self.edit_custom_parameters, width=20)
        self.custom_param_btn.pack(anchor=tk.W, pady=(5, 0))
        self.custom_param_btn.config(state="disabled")

        self.use_default_grid.trace("w", self.on_param_grid_change)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(bottom_frame, text="OK", command=self.ok, width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(bottom_frame, text="Cancel", command=self.cancel, width=10).pack(side=tk.RIGHT)

        self.update_model_options()

    def load_initial_config(self):
        if 'problem_type' in self.initial_config:
            self.problem_type.set(self.initial_config['problem_type'])
        if 'model_name' in self.initial_config:
            self.model_var.set(self.initial_config['model_name'])
        if 'x_scaling' in self.initial_config:
            self.x_scaling_method.set(self.initial_config['x_scaling'])
        if 'y_scaling' in self.initial_config:
            self.y_scaling_method.set(self.initial_config['y_scaling'])
        if 'use_pca' in self.initial_config:
            self.use_pca.set(self.initial_config['use_pca'])
        if 'pca_components' in self.initial_config:
            self.pca_components.set(self.initial_config['pca_components'])

        self.update_model_options()

        if 'param_grid' in self.initial_config and self.initial_config['param_grid']:
            self.use_default_grid.set(False)
            self.custom_param_grid = self.initial_config['param_grid']

    def on_param_grid_change(self, *args):
        self.custom_param_btn.config(state="normal" if not self.use_default_grid.get() else "disabled")

    def edit_custom_parameters(self):
        import json
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model first!")
            return

        default_params = DEFAULT_PARAM_GRIDS.get(model_name, {})

        param_dialog = tk.Toplevel(self.dialog)
        param_dialog.title(f"Custom Parameters for {model_name}")
        param_dialog.geometry("500x400")
        param_dialog.transient(self.dialog)

        main_frame = ttk.Frame(param_dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame,
                 text="Enter parameters in JSON format. Example:\n{'n_estimators': [50, 100], 'max_depth': [5, 10]}",
                 font=("Arial", 9), foreground="gray").pack(anchor=tk.W, pady=(0, 10))

        default_frame = ttk.LabelFrame(main_frame, text="Default Parameters (for reference)", padding=5)
        default_frame.pack(fill=tk.X, pady=(0, 10))

        default_text = scrolledtext.ScrolledText(default_frame, height=5)
        default_text.pack(fill=tk.BOTH, expand=True)
        default_text.insert(tk.END, "Default parameter grid:\n")
        for key, value in default_params.items():
            default_text.insert(tk.END, f"  {key}: {value}\n")
        default_text.configure(state='disabled')

        custom_frame = ttk.LabelFrame(main_frame, text="Custom Parameters", padding=5)
        custom_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.custom_param_text = scrolledtext.ScrolledText(custom_frame, height=8)
        self.custom_param_text.pack(fill=tk.BOTH, expand=True)

        if hasattr(self, 'custom_param_grid') and self.custom_param_grid:
            self.custom_param_text.insert(tk.END, json.dumps(self.custom_param_grid, indent=2))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="OK",
                  command=lambda: self.save_custom_parameters(param_dialog), width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel", command=param_dialog.destroy, width=10).pack(side=tk.RIGHT)

    def save_custom_parameters(self, dialog):
        import json
        try:
            param_text = self.custom_param_text.get(1.0, tk.END).strip()
            if param_text:
                param_grid = json.loads(param_text)
                if not isinstance(param_grid, dict):
                    raise ValueError("Parameter grid must be a dictionary")
                self.custom_param_grid = param_grid
            else:
                self.custom_param_grid = {}
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid JSON format: {str(e)}")

    def update_model_options(self):
        if self.problem_type.get() == "regression":
            models = REGRESSION_MODELS
        else:
            models = CLASSIFICATION_MODELS

        self.model_combo['values'] = models
        if models and not self.model_var.get():
            self.model_combo.set(models[0])

    def ok(self):
        import json
        problem_type = self.problem_type.get()
        model_name = self.model_var.get()

        if not model_name:
            messagebox.showwarning("Warning", "Please select a model!")
            return

        if self.use_default_grid.get():
            param_grid = DEFAULT_PARAM_GRIDS.get(model_name, {})
        else:
            param_grid = getattr(self, 'custom_param_grid', {})

        self.result = {
            'problem_type': problem_type,
            'model_name': model_name,
            'model': MODEL_MAP[model_name](),
            'x_scaling': self.x_scaling_method.get(),
            'y_scaling': self.y_scaling_method.get(),
            'use_pca': self.use_pca.get(),
            'pca_components': self.pca_components.get(),
            'param_grid': param_grid,
        }

        self.dialog.destroy()

    def cancel(self):
        self.dialog.destroy()

    def show(self):
        self.dialog.wait_window()
        return self.result


class EditConfigurationDialog:
    """Edit configuration dialog."""

    def __init__(self, parent, data, config, index):
        self.parent = parent
        self.data = data
        self.config = config.copy()
        self.index = index
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Edit Configuration {index + 1}")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        feature_tab = ttk.Frame(notebook)
        notebook.add(feature_tab, text="Features")
        self.create_feature_tab(feature_tab)

        model_tab = ttk.Frame(notebook)
        notebook.add(model_tab, text="Model")
        self.create_model_tab(model_tab)

        preview_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        self.create_preview_tab(preview_tab)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(bottom_frame, text="Save", command=self.save, width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(bottom_frame, text="Cancel", command=self.cancel, width=10).pack(side=tk.RIGHT)

    def create_feature_tab(self, parent):
        x_frame = ttk.LabelFrame(parent, text="X Features", padding=10)
        x_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        x_text = scrolledtext.ScrolledText(x_frame, height=5)
        x_text.pack(fill=tk.BOTH, expand=True)

        if 'x_cols' in self.config:
            x_text.insert(tk.END, f"Selected {len(self.config['x_cols'])} X features:\n")
            for i, feat in enumerate(self.config['x_cols']):
                x_text.insert(tk.END, f"{i+1}. {feat}\n")
        else:
            x_text.insert(tk.END, "No X features selected")
        x_text.configure(state='disabled')

        x_btn_frame = ttk.Frame(x_frame)
        x_btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(x_btn_frame, text="Edit X Features", command=self.edit_x_features, width=15).pack()

        y_frame = ttk.LabelFrame(parent, text="Y Feature", padding=10)
        y_frame.pack(fill=tk.X, pady=(0, 10))

        y_text = scrolledtext.ScrolledText(y_frame, height=3)
        y_text.pack(fill=tk.BOTH, expand=True)

        if 'y_col' in self.config:
            y_text.insert(tk.END, f"Selected Y feature: {self.config['y_col']}")
        else:
            y_text.insert(tk.END, "No Y feature selected")
        y_text.configure(state='disabled')

        y_btn_frame = ttk.Frame(y_frame)
        y_btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(y_btn_frame, text="Edit Y Feature", command=self.edit_y_feature, width=15).pack()

    def create_model_tab(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Current Model Configuration", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        info_text = scrolledtext.ScrolledText(info_frame, height=10)
        info_text.pack(fill=tk.BOTH, expand=True)

        if 'model_config' in self.config:
            model_config = self.config['model_config']
            info_text.insert(tk.END, f"Problem Type: {model_config.get('problem_type', 'N/A')}\n")
            info_text.insert(tk.END, f"Model: {model_config.get('model_name', 'N/A')}\n")
            info_text.insert(tk.END, f"X Scaling: {model_config.get('x_scaling', 'N/A')}\n")
            info_text.insert(tk.END, f"Y Scaling: {model_config.get('y_scaling', 'N/A')}\n")
            info_text.insert(tk.END, f"PCA: {'Yes' if model_config.get('use_pca', False) else 'No'}\n")
            if model_config.get('use_pca', False):
                info_text.insert(tk.END, f"PCA Components: {model_config.get('pca_components', 'N/A')}\n")

            param_grid = model_config.get('param_grid', {})
            if param_grid:
                info_text.insert(tk.END, f"\nParameter Grid: {len(param_grid)} parameters\n")
                for key, value in param_grid.items():
                    info_text.insert(tk.END, f"  {key}: {value}\n")
        else:
            info_text.insert(tk.END, "No model configuration found")
        info_text.configure(state='disabled')

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Edit Model Configuration",
                  command=self.edit_model_config, width=20).pack()

    def create_preview_tab(self, parent):
        preview_frame = ttk.Frame(parent, padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        preview_text = scrolledtext.ScrolledText(preview_frame, height=15)
        preview_text.pack(fill=tk.BOTH, expand=True)

        preview_lines = ["=" * 60, "CONFIGURATION PREVIEW", "=" * 60, ""]
        preview_lines.append("FEATURES:")
        preview_lines.append("-" * 30)

        if 'x_cols' in self.config:
            preview_lines.append(f"X Features: {len(self.config['x_cols'])} features")
            if len(self.config['x_cols']) <= 5:
                preview_lines.append(f"  {', '.join(self.config['x_cols'])}")
            else:
                preview_lines.append(f"  {', '.join(self.config['x_cols'][:3])} ... ({len(self.config['x_cols']) - 3} more)")
        else:
            preview_lines.append("X Features: Not configured")

        if 'y_col' in self.config:
            preview_lines.append(f"Y Feature: {self.config['y_col']}")
        else:
            preview_lines.append("Y Feature: Not configured")

        preview_lines.extend(["", "MODEL CONFIGURATION:", "-" * 30])

        if 'model_config' in self.config:
            model_config = self.config['model_config']
            preview_lines.append(f"Problem Type: {model_config.get('problem_type', 'N/A')}")
            preview_lines.append(f"Model: {model_config.get('model_name', 'N/A')}")
            preview_lines.append(f"X Scaling: {model_config.get('x_scaling', 'N/A')}")
            preview_lines.append(f"Y Scaling: {model_config.get('y_scaling', 'N/A')}")
            preview_lines.append(f"PCA: {'Yes' if model_config.get('use_pca', False) else 'No'}")
            if model_config.get('use_pca', False):
                preview_lines.append(f"PCA Components: {model_config.get('pca_components', 'N/A')}")
            param_grid = model_config.get('param_grid', {})
            if param_grid:
                preview_lines.append(f"Parameter Grid: {len(param_grid)} parameters")
        else:
            preview_lines.append("Model Configuration: Not configured")

        preview_lines.extend(["", "=" * 60])

        for line in preview_lines:
            preview_text.insert(tk.END, line + "\n")
        preview_text.configure(state='disabled')

    def edit_x_features(self):
        dialog = FeatureSelectionDialog(self.dialog, "Edit X Features (Multiple)",
                                        list(self.data.columns), multiselect=True,
                                        initial_selection=self.config.get('x_cols', []))
        x_features = dialog.show()
        if x_features:
            self.config['x_cols'] = x_features

    def edit_y_feature(self):
        dialog = FeatureSelectionDialog(self.dialog, "Edit Y Feature (Single)",
                                        list(self.data.columns), multiselect=False,
                                        initial_selection=[self.config.get('y_col')] if self.config.get('y_col') else [])
        y_feature = dialog.show()
        if y_feature:
            self.config['y_col'] = y_feature[0] if y_feature else None

    def edit_model_config(self):
        initial_config = self.config.get('model_config', {})
        dialog = ModelConfigDialog(self.dialog, initial_config)
        model_config = dialog.show()
        if model_config:
            self.config['model_config'] = model_config

    def save(self):
        if 'x_cols' not in self.config or not self.config['x_cols']:
            messagebox.showwarning("Warning", "Please select X features!")
            return
        if 'y_col' not in self.config or not self.config['y_col']:
            messagebox.showwarning("Warning", "Please select Y feature!")
            return
        if 'model_config' not in self.config:
            messagebox.showwarning("Warning", "Please configure the model!")
            return
        if self.config['y_col'] in self.config['x_cols']:
            messagebox.showwarning("Warning", "Y feature cannot be the same as X features!")
            return

        x_count = len(self.config['x_cols'])
        y_feature = self.config['y_col']
        model_name = self.config['model_config'].get('model_name', 'Unknown')
        self.config['description'] = f"X[{x_count}] → Y[{y_feature}] | Model: {model_name}"

        self.result = self.config
        self.dialog.destroy()

    def cancel(self):
        self.dialog.destroy()

    def show(self):
        self.dialog.wait_window()
        return self.result


class MLTrainerApp:
    """Main ML Trainer GUI application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Trainer with SHAP")
        self.root.geometry("1200x700")

        self.data = None
        self.features_list = []
        self.current_results = {}

        self.setup_styles()
        self.create_widgets()
        self.update_status()

        if not SHAP_AVAILABLE:
            messagebox.showwarning("SHAP Warning",
                "SHAP library is not installed. Feature importance analysis will be limited.\n"
                "Install it using: pip install shap")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Title.TLabel", font=("Arial", 11, "bold"))
        style.configure("Info.TLabel", font=("Arial", 9))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_toolbar(main_frame)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.create_features_panel(left_panel)

        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.create_operations_panel(right_panel)

    def create_toolbar(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(toolbar, text="Load Data",
                  command=self.load_data, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Preview Data",
                  command=self.preview_data, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Data Stats",
                  command=self.show_data_stats, width=12).pack(side=tk.LEFT, padx=5)

        shap_status = "Available" if SHAP_AVAILABLE else "Not Available"
        shap_color = "green" if SHAP_AVAILABLE else "red"
        self.shap_label = ttk.Label(toolbar, text=f"SHAP: {shap_status}", foreground=shap_color)
        self.shap_label.pack(side=tk.RIGHT, padx=(10, 0))

        self.status_label = ttk.Label(toolbar, text="No data loaded", foreground="gray")
        self.status_label.pack(side=tk.RIGHT)

    def create_features_panel(self, parent):
        features_frame = ttk.LabelFrame(parent, text="Feature Configurations", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(features_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.features_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
        self.features_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.features_listbox.yview)
        self.features_listbox.bind('<Double-Button-1>', self.on_feature_double_click)

        btn_frame = ttk.Frame(features_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="Add Configuration",
                  command=self.add_feature_config, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Edit Selected",
                  command=self.edit_feature_config, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Duplicate Selected",
                  command=self.duplicate_feature_config, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Selected",
                  command=self.delete_feature_config, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All",
                  command=self.clear_feature_configs, width=15).pack(side=tk.LEFT, padx=5)

    def create_operations_panel(self, parent):
        train_frame = ttk.LabelFrame(parent, text="Model Training", padding=10)
        train_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(train_frame, text="Train Selected Configuration",
                  command=self.train_selected, width=25).pack(pady=(0, 10))
        ttk.Button(train_frame, text="Train All Configurations",
                  command=self.train_all, width=25).pack(pady=(0, 20))

        results_frame = ttk.LabelFrame(train_frame, text="Training Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=12)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        viz_frame = ttk.Frame(train_frame)
        viz_frame.pack(fill=tk.X, pady=(10, 0))

        basic_viz_frame = ttk.Frame(viz_frame)
        basic_viz_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(basic_viz_frame, text="Plot Results",
                  command=self.plot_results, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(basic_viz_frame, text="Plot Learning Curve",
                  command=self.plot_learning_curve, width=15).pack(side=tk.LEFT, padx=5)

        shap_viz_frame = ttk.Frame(viz_frame)
        shap_viz_frame.pack(side=tk.RIGHT, fill=tk.X)

        ttk.Button(shap_viz_frame, text="SHAP Analysis",
                  command=self.show_shap_analysis, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(shap_viz_frame, text="Save Results",
                  command=self.save_results, width=12).pack(side=tk.LEFT)

    def update_status(self):
        if self.data is None:
            self.status_label.config(text="No data loaded", foreground="gray")
        else:
            self.status_label.config(text=f"Data: {self.data.shape[0]} rows, {self.data.shape[1]} columns",
                                   foreground="black")

        self.features_listbox.delete(0, tk.END)
        for i, feat in enumerate(self.features_list):
            desc = feat.get('description', f'Configuration {i+1}')
            self.features_listbox.insert(tk.END, desc)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.features_list = []
                self.update_status()
                messagebox.showinfo("Success", f"Data loaded successfully!\nShape: {self.data.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def preview_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        preview_window = tk.Toplevel(self.root)
        preview_window.title("Data Preview")
        preview_window.geometry("800x600")

        tree_frame = ttk.Frame(preview_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        tree = ttk.Treeview(tree_frame,
                           yscrollcommand=tree_scroll_y.set,
                           xscrollcommand=tree_scroll_x.set)
        tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)

        tree["columns"] = list(self.data.columns)
        tree["show"] = "headings"
        for col in self.data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for i, row in self.data.head(100).iterrows():
            tree.insert("", "end", values=list(row))

        ttk.Label(preview_window, text=f"Showing first 100 rows of {len(self.data)} total").pack(pady=5)

    def show_data_stats(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Data Statistics")
        stats_window.geometry("600x500")

        stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        stats_text.insert(tk.END, "Data Information:\n")
        stats_text.insert(tk.END, f"Shape: {self.data.shape}\n")
        stats_text.insert(tk.END, f"Columns: {len(self.data.columns)}\n")
        stats_text.insert(tk.END, f"Rows: {len(self.data)}\n\n")

        stats_text.insert(tk.END, "Data Types:\n")
        for col in self.data.columns:
            dtype = self.data[col].dtype
            stats_text.insert(tk.END, f"  {col}: {dtype}\n")
        stats_text.insert(tk.END, "\n")

        stats_text.insert(tk.END, "Missing Values:\n")
        missing = self.data.isnull().sum()
        for col in self.data.columns:
            if missing[col] > 0:
                stats_text.insert(tk.END, f"  {col}: {missing[col]} ({missing[col]/len(self.data):.2%})\n")
        if missing.sum() == 0:
            stats_text.insert(tk.END, "  No missing values\n")

        stats_text.insert(tk.END, "\nNumeric Columns Summary:\n")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_text.insert(tk.END, self.data[numeric_cols].describe().to_string())
        else:
            stats_text.insert(tk.END, "No numeric columns")

        stats_text.configure(state='disabled')

    def add_feature_config(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        x_dialog = FeatureSelectionDialog(self.root, "Select X Features (Multiple)",
                                          list(self.data.columns), multiselect=True)
        x_features = x_dialog.show()
        if not x_features:
            return

        y_dialog = FeatureSelectionDialog(self.root, "Select Y Feature (Single)",
                                          list(self.data.columns), multiselect=False)
        y_feature = y_dialog.show()
        if not y_feature:
            return
        y_feature = y_feature[0] if y_feature else None

        if y_feature in x_features:
            messagebox.showwarning("Warning", "Y feature cannot be the same as X features!")
            return

        model_dialog = ModelConfigDialog(self.root)
        model_config = model_dialog.show()
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

    def edit_feature_config(self):
        selected = self.features_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a configuration to edit!")
            return

        index = selected[0]
        if index >= len(self.features_list):
            return

        current_config = self.features_list[index]
        edit_dialog = EditConfigurationDialog(self.root, self.data, current_config, index)
        edited_config = edit_dialog.show()
        if edited_config:
            self.features_list[index] = edited_config
            self.update_status()
            messagebox.showinfo("Success", "Configuration updated successfully!")

    def duplicate_feature_config(self):
        selected = self.features_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a configuration to duplicate!")
            return

        index = selected[0]
        if index >= len(self.features_list):
            return

        current_config = self.features_list[index]
        duplicated_config = copy.deepcopy(current_config)
        if 'description' in duplicated_config:
            duplicated_config['description'] = f"Copy of {duplicated_config['description']}"
        self.features_list.append(duplicated_config)
        self.update_status()
        messagebox.showinfo("Success", "Configuration duplicated successfully!")

    def delete_feature_config(self):
        selected = self.features_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a configuration to delete!")
            return

        index = selected[0]
        if index < len(self.features_list):
            self.features_list.pop(index)
            self.update_status()

    def clear_feature_configs(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all configurations?"):
            self.features_list = []
            self.update_status()

    def on_feature_double_click(self, event):
        self.edit_feature_config()

    def train_selected(self):
        selected = self.features_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a configuration to train!")
            return

        index = selected[0]
        if index >= len(self.features_list):
            return

        feature_config = self.features_list[index]
        model_config = feature_config['model_config']

        results = train_model_with_config(
            self.data, feature_config, model_config, model_config['param_grid']
        )

        self.current_results = results
        self.current_results['feature_config'] = feature_config
        self.current_results['model_config'] = model_config
        self.display_results(results)

    def train_all(self):
        if not self.features_list:
            messagebox.showwarning("Warning", "No configurations to train!")
            return

        if not messagebox.askyesno("Confirm", f"Train all {len(self.features_list)} configurations?"):
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Batch Training Results:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        for i, feature_config in enumerate(self.features_list):
            self.results_text.insert(tk.END, f"Configuration {i+1}: {feature_config['description']}\n")

            try:
                model_config = feature_config['model_config']
                results = train_model_with_config(
                    self.data, feature_config, model_config, model_config['param_grid']
                )

                if results['success']:
                    if model_config['problem_type'] == 'regression':
                        self.results_text.insert(tk.END, f"  MSE: {results['mse']:.4f}, "
                                                        f"MAE: {results['mae']:.4f}, "
                                                        f"R²: {results['r2']:.4f}\n")
                    else:
                        self.results_text.insert(tk.END, f"  Accuracy: {results['accuracy']:.4f}, "
                                                        f"F1: {results['f1']:.4f}\n")
                    if 'best_params' in results:
                        self.results_text.insert(tk.END, f"  Best Params: {results['best_params']}\n")
                else:
                    self.results_text.insert(tk.END, f"  Error: {results['error']}\n")

            except Exception as e:
                self.results_text.insert(tk.END, f"  Error: {str(e)}\n")

            self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, "Batch training completed!\n")

    def display_results(self, results):
        self.results_text.delete(1.0, tk.END)

        if not results['success']:
            self.results_text.insert(tk.END, f"Training failed: {results['error']}")
            return

        self.results_text.insert(tk.END, "Training Results:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        feature_config = self.current_results['feature_config']
        model_config = self.current_results['model_config']

        self.results_text.insert(tk.END, f"Configuration: {feature_config['description']}\n")
        self.results_text.insert(tk.END, f"X Features: {len(feature_config['x_cols'])} features\n")
        self.results_text.insert(tk.END, f"Y Feature: {feature_config['y_col']}\n")
        self.results_text.insert(tk.END, f"Model: {model_config['model_name']}\n")
        self.results_text.insert(tk.END, f"Problem Type: {model_config['problem_type']}\n")

        if feature_config['use_pca']:
            self.results_text.insert(tk.END, f"PCA: Yes ({feature_config['pca_components']} components)\n")
            if 'pca_explained_variance' in results:
                self.results_text.insert(tk.END, f"PCA Explained Variance: {results['pca_explained_variance']:.3f}\n")
        else:
            self.results_text.insert(tk.END, "PCA: No\n")

        self.results_text.insert(tk.END, f"X Scaling: {feature_config['x_scaling']}\n")
        if model_config['problem_type'] == 'regression':
            self.results_text.insert(tk.END, f"Y Scaling: {feature_config['y_scaling']}\n")

        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n\n")

        if model_config['problem_type'] == 'regression':
            self.results_text.insert(tk.END, "Regression Metrics:\n")
            self.results_text.insert(tk.END, f"  MSE: {results['mse']:.4f}\n")
            self.results_text.insert(tk.END, f"  MAE: {results['mae']:.4f}\n")
            self.results_text.insert(tk.END, f"  R² Score: {results['r2']:.4f}\n")
        else:
            self.results_text.insert(tk.END, "Classification Metrics:\n")
            self.results_text.insert(tk.END, f"  Accuracy: {results['accuracy']:.4f}\n")
            self.results_text.insert(tk.END, f"  F1 Score: {results['f1']:.4f}\n")

        if 'best_params' in results:
            self.results_text.insert(tk.END, "\nGrid Search Results:\n")
            self.results_text.insert(tk.END, f"  Best Parameters: {results['best_params']}\n")
            self.results_text.insert(tk.END, f"  Best Score: {results['best_score']:.4f}\n")

        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n\n")
        self.results_text.insert(tk.END, "FEATURE IMPORTANCE ANALYSIS\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n\n")

        self.results_text.insert(tk.END, f"SHAP: {'Available ✓' if results.get('shap_available', False) else 'Not Available ✗'}\n")

        if 'feature_importance' in results and results['feature_importance']:
            for method, importance_dict in results['feature_importance'].items():
                if importance_dict and method != 'shap':
                    self.results_text.insert(tk.END, f"\n{method.replace('_', ' ').title()}:\n")
                    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                    for feature, imp in sorted_importance[:10]:
                        self.results_text.insert(tk.END, f"  {feature}: {imp:.4f}\n")

        if results.get('shap_importance'):
            self.results_text.insert(tk.END, "\nSHAP Feature Importance (Top 10):\n")
            sorted_shap = sorted(results['shap_importance'].items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_shap[:10]:
                self.results_text.insert(tk.END, f"  {feature}: {imp:.4f}\n")

            if feature_config['use_pca'] and 'shap_importance_original' in results:
                self.results_text.insert(tk.END, "\nSHAP Importance (Original Features, Top 10):\n")
                sorted_original = sorted(results['shap_importance_original'].items(), key=lambda x: x[1], reverse=True)
                for feature, imp in sorted_original[:10]:
                    self.results_text.insert(tk.END, f"  {feature}: {imp:.4f}\n")

        if 'feature_importance_error' in results:
            self.results_text.insert(tk.END, f"\nFeature Importance Error: {results['feature_importance_error']}\n")
        if 'shap_error' in results:
            self.results_text.insert(tk.END, f"\nSHAP Error: {results['shap_error']}\n")

        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.results_text.insert(tk.END, "\nTraining completed successfully!\n")
        self.results_text.insert(tk.END, "\nClick 'SHAP Analysis' for detailed feature importance visualizations.\n")

    def plot_results(self):
        if not self.current_results or not self.current_results['success']:
            messagebox.showwarning("Warning", "No training results to plot!")
            return

        model_config = self.current_results['model_config']
        feature_config = self.current_results['feature_config']
        title = f"{model_config['model_name']} - {feature_config['description']}"

        if model_config['problem_type'] == 'regression':
            plot_regression_results(self.current_results['y_test'], self.current_results['y_pred'], title)
        else:
            plot_classification_results(self.current_results['y_test'], self.current_results['y_pred'], title)

    def plot_learning_curve(self):
        if not self.current_results or not self.current_results['success']:
            messagebox.showwarning("Warning", "No training results to plot!")
            return

        from core import create_scaler

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
        plot_learning_curve(model_config['model'], X_scaled, y_flat, title, model_config['problem_type'])

    def show_shap_analysis(self):
        if not self.current_results or not self.current_results['success']:
            messagebox.showwarning("Warning", "No training results available for SHAP analysis!")
            return

        if not SHAP_AVAILABLE:
            messagebox.showwarning("SHAP Not Available",
                "SHAP library is not installed. Please install it using: pip install shap")
            return

        if 'shap_values' not in self.current_results:
            messagebox.showinfo("No SHAP Values",
                "SHAP values were not calculated for this model. Try training with a different model.")
            return

        shap_window = tk.Toplevel(self.root)
        shap_window.title("SHAP Feature Importance Analysis")
        shap_window.geometry("600x500")

        main_frame = ttk.Frame(shap_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="SHAP Analysis Options",
                 font=("Arial", 12, "bold")).pack(pady=(0, 20))

        if self.current_results['feature_config']['use_pca']:
            feature_names = self.current_results['pca_feature_names']
        else:
            feature_names = self.current_results['original_feature_names']

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(buttons_frame, text="SHAP Summary Plot",
                  command=lambda: self.plot_shap_visualization('summary'), width=25).pack(pady=10)
        ttk.Button(buttons_frame, text="SHAP Feature Importance Bar Plot",
                  command=lambda: self.plot_shap_visualization('bar'), width=25).pack(pady=10)

        if 'feature_importance' in self.current_results:
            for method in self.current_results['feature_importance']:
                if method != 'shap' and self.current_results['feature_importance'][method]:
                    m = method
                    ttk.Button(buttons_frame,
                              text=f"Model Internal Importance ({method})",
                              command=lambda m=m: self.plot_model_internal_importance_viz(m),
                              width=25).pack(pady=5)

        dependence_frame = ttk.LabelFrame(buttons_frame, text="SHAP Dependence Plot", padding=10)
        dependence_frame.pack(fill=tk.X, pady=20)

        ttk.Label(dependence_frame, text="Select feature:").pack(side=tk.LEFT, padx=(0, 10))

        self.dependence_feature_var = tk.StringVar(value=feature_names[0] if feature_names else "")
        feature_combo = ttk.Combobox(dependence_frame, textvariable=self.dependence_feature_var,
                                      values=feature_names, state="readonly", width=30)
        feature_combo.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(dependence_frame, text="Plot Dependence",
                  command=lambda: self.plot_shap_visualization('dependence'), width=15).pack(side=tk.LEFT)

        importance_dicts = {}
        if 'feature_importance' in self.current_results:
            for method, imp_dict in self.current_results['feature_importance'].items():
                if imp_dict and method != 'shap':
                    importance_dicts[method] = imp_dict
        if self.current_results.get('shap_importance'):
            importance_dicts['shap'] = self.current_results['shap_importance']
        if len(importance_dicts) > 1:
            ttk.Button(buttons_frame, text="Compare Feature Importance Methods",
                      command=lambda: self.plot_shap_visualization('comparison'), width=30).pack(pady=10)

        ttk.Label(main_frame, text="Note: Each plot will open in a separate matplotlib window.",
                 font=("Arial", 9), foreground="gray").pack(pady=(20, 0))
        ttk.Button(main_frame, text="Close", command=shap_window.destroy, width=15).pack(pady=(20, 0))

    def plot_shap_visualization(self, plot_type):
        if not self.current_results or 'shap_values' not in self.current_results:
            messagebox.showwarning("Warning", "SHAP values not available!")
            return

        if self.current_results['feature_config']['use_pca']:
            feature_names = self.current_results['pca_feature_names']
        else:
            feature_names = self.current_results['original_feature_names']

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
                    messagebox.showwarning("Warning", "SHAP importance data not available!")
                    return
            elif plot_type == 'dependence':
                feature_name = self.dependence_feature_var.get() or (feature_names[0] if feature_names else "")
                if feature_name in feature_names:
                    feature_index = feature_names.index(feature_name)
                    title = f"SHAP Dependence Plot - {feature_name}"
                    success = plot_shap_dependence(
                        self.current_results['shap_values'], feature_names, feature_index,
                        self.current_results.get('X_test'), title
                    )
                else:
                    messagebox.showwarning("Warning", f"Feature '{feature_name}' not found!")
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
                    messagebox.showwarning("Warning", "Need at least 2 importance methods to compare!")
                    return

            if not success:
                messagebox.showerror("Error", f"Failed to generate {plot_type} plot!")
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting {plot_type}: {str(e)}")

    def plot_model_internal_importance_viz(self, method):
        if (not self.current_results or
            'feature_importance' not in self.current_results or
            method not in self.current_results['feature_importance']):
            messagebox.showwarning("Warning", f"{method} importance data not available!")
            return

        importance_dict = self.current_results['feature_importance'][method]
        if not importance_dict:
            messagebox.showwarning("Warning", f"{method} importance data is empty!")
            return

        model_name = self.current_results['model_config']['model_name']
        title = f"{method.replace('_', ' ').title()} - {model_name}"

        try:
            success = plot_model_internal_importance(importance_dict, title)
            if not success:
                messagebox.showerror("Error", f"Failed to generate {method} importance plot!")
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting {method} importance: {str(e)}")

    def save_results(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                  filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")


def main():
    root = tk.Tk()
    MLTrainerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

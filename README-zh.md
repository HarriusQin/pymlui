# PyMLUI - 机器学习训练器（含 SHAP）

一款基于 GUI 的机器学习工具，提供直观的界面用于训练、评估和解释 ML 模型，支持 SHAP 特征重要性分析。

## 功能特性

- **数据加载**：加载 CSV 文件，自动检测数据类型
- **特征选择**：交互式对话框选择 X 特征和 Y 目标变量
- **模型支持**：
  - 回归：线性回归、岭回归、Lasso、决策树、随机森林、梯度提升、SVR、KNN
  - 分类：逻辑回归、决策树、随机森林、SVC、KNN
- **数据预处理**：StandardScaler、MinMaxScaler、RobustScaler、PCA
- **超参数调优**：GridSearchCV 配合默认参数网格
- **可视化**：
  - 回归：真实值 vs 预测值、残差图
  - 分类：混淆矩阵、准确率饼图
  - 学习曲线
- **SHAP 分析**：
  - SHAP 摘要图
  - 特征重要性条形图
  - 依赖关系图
  - 多重要性方法对比

## 安装

```bash
# 创建虚拟环境
uv venv .venv

# 激活环境（Windows）
.venv\Scripts\activate

# 安装依赖
uv pip install pandas numpy matplotlib scikit-learn shap
```

## 使用方法

```bash
.venv\Scripts\python mlui.py
```

或使用 uv：

```bash
uv run --active python mlui.py
```

## 工作流程

### 1. 加载数据
点击 **Load Data** 选择 CSV 文件。工具会显示数据形状、数据类型和缺失值统计。

### 2. 创建配置
点击 **Add Configuration** 定义：
- **X Features**：选择多个特征作为输入变量
- **Y Feature**：选择目标变量
- **Model**：选择算法和问题类型（回归/分类）
- **Scaling**：选择 X/Y 标准化方法
- **PCA**：可选启用 PCA 降维
- **Parameters**：使用默认参数网格或自定义

### 3. 训练模型
- **Train Selected**：训练当前选中的配置
- **Train All**：批量训练所有配置

### 4. 查看结果
训练结果显示：
- 评估指标（回归：MSE、MAE、R²；分类：Accuracy、F1）
- 网格搜索最佳参数
- 模型内置特征重要性和 SHAP 重要性

### 5. 可视化
- **Plot Results**：回归/分类结果可视化
- **Plot Learning Curve**：模型学习曲线
- **SHAP Analysis**：打开 SHAP 可视化窗口，包含：
  - 摘要图
  - 特征重要性条形图
  - 依赖关系图
  - 多重要性方法对比

### 6. 保存结果
点击 **Save Results** 将训练输出导出为文本文件。

## 配置管理

- **Edit Selected**：修改选中配置
- **Duplicate Selected**：克隆配置进行对比实验
- **Delete Selected**：删除选中配置
- **Clear All**：清除所有配置

## 项目结构

```
pymlui/
├── mlui.py          # 主程序
├── pyproject.toml   # 项目配置
├── README.md        # 英文文档
├── README-zh.md     # 中文文档
├── .venv/           # 虚拟环境
```

## 依赖

- pandas, numpy - 数据处理
- matplotlib - 可视化
- scikit-learn - 机器学习
- shap - 模型可解释性
- tkinter - GUI（Python 内置）

## SHAP 集成

SHAP（SHapley Additive exPlanations）提供模型无关的特征重要性解释：

- **树模型**：使用 TreeExplainer 进行快速精确计算
- **线性模型**：使用 LinearExplainer
- **其他模型**：使用 KernelExplainer（较慢）

启用 PCA 时，SHAP 值会映射回原始特征空间以便解释。

# ps3r 安装与使用

## 1. 安装与导入

使用 Python 3.8 或更高版本。在 PyPI 正式上传成功后，可以运行：

```bash
python -m pip install ps3r
```

也可以直接安装交付的 wheel 文件，或进入项目目录从源码安装：

```bash
python -m pip install ./dist/ps3r-0.2.0-py3-none-any.whl
# 源码安装：
python -m pip install .
```

```python
import ps3r
from ps3r import PS3RModel, get_config, get_data_path

print(ps3r.__version__)
model = PS3RModel(input_dim=9, solver_kwargs={"config": get_config("simulation")})
data_path = get_data_path("simulation", "simulation_data.csv")
```

安装、导入、创建模型对象、读取配置以及检查数据均不需要 DSO/PyTorch。
**调用默认模型训练时，仍然必须先安装实验使用的 PyTorch DSO 完整源码并应用补丁。**
原目录只有四份补丁，没有完整 DSO 后端；此包没有以其他算法替代 DSO。
`pip install 'ps3r[dso-runtime]'` 只能安装配套依赖，不能补齐后端源码。
具体步骤见 [DSO 安装说明](docs/DSO_SETUP.md)。

原实验记录的环境是 Python 3.8、NumPy 1.19.5、Numba 0.53.1。
包允许较新的 Python 用于安装和数据检查；跨环境重跑能否得到完全相同的训练结果，
需要实际验证。原始安装顺序完整保留在 [原 README](docs/ORIGINAL_README.md)。

## 2. 检查数据和环境

```bash
ps3r validate all
ps3r doctor
ps3r doctor --require-runtime
```

第一条检查三组数据及划分；第二条显示依赖位置；第三条实际检查后端导入和四份补丁，
缺少后端或补丁不符时返回失败。数据检查通过不等于完成模型训练或复现论文结果。

## 3. 运行实验

后端准备完成后，可在任意工作目录运行：

```bash
ps3r-bd --output-dir ./results/bd
ps3r-socd --output-dir ./results/socd --seeds 0 1 2 3 4
ps3r-simulation --output-dir ./results/simulation --seeds 0 1 2 3 4
```

不指定输出目录时，结果保存到当前工作目录下的时间戳目录，避免写入 Python 安装目录。
BD 保留五个预先划定的空间折和模型种子 0；SOCD、模拟实验保留固定训练/测试划分及
种子 0–4。

## 4. 读取和导出数据

```python
import pandas as pd
from ps3r import get_data_path, get_config

frame = pd.read_csv(get_data_path("simulation", "simulation_data.csv"))
config = get_config("simulation")
```

需要编辑数据或配置时，先导出到新目录：

```bash
ps3r export simulation ./simulation-example
python ./simulation-example/run_simulation.py --validate-data-only
ps3r export-patches ./dso-patches
```

导出不会覆盖非空目录。`get_config()` 每次返回独立字典，修改它不会改变包内配置。

## 5. 使用自己的数据

```python
from ps3r import PS3RModel, get_config

model = PS3RModel(
    input_dim=X_train.shape[1],
    solver_kwargs={"config": get_config("simulation")},
    random_state=0,
)
model.fit(X_train, y_train, spatial_block_ids=train_block_ids)
y_pred = model.predict(X_test)
```

`X_train` 是二维特征矩阵，`y_train` 是目标数组，`train_block_ids` 与训练样本逐行对应，
至少包含两个不同的空间块/分组。内部验证按完整组保留；测试集不要传入 `fit`。
请根据实际问题配置哪些特征用于符号表达式和路由，并自行确定合适的超参数。

## 6. 发布文件

- `dist/ps3r-0.2.0-py3-none-any.whl`：可直接安装的包。
- `dist/ps3r-0.2.0.tar.gz`：可构建、可安装的源码分发包。
- `tests/`：模型接口、分组隔离、资源完整性和命令行验证。
- `reference/`：原始模型与实验脚本，供核对改动。
- `docs/VALIDATION.md`：本次实际验证范围。
- `docs/PUBLISHING.md`：GitHub 与 PyPI 发布步骤。

PS3R 采用 [MIT 许可证](LICENSE)。DSO 衍生补丁保留上游 BSD-3-Clause
许可，见 [第三方声明](THIRD_PARTY_NOTICES.md)。

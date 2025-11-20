# 🧪 QSAR 分子活性预测 Web 应用
这是一个基于 Flask 的 QSAR（定量构效关系）建模工具，支持上传含 SMILES 和生物活性值的 CSV 文件，自动训练回归或分类模型，并可视化结果。适用于药物发现、化学信息学等场景。

# ✨ 功能亮点
📤 上传 CSV 文件（需包含 SMILES 列和活性值列）
🔍 自动识别关键列（列名含 smile 和 act/value/target 等即可）
⚙️ 支持两种任务：
回归：预测连续活性值（如 pIC50）
分类：根据中位数阈值分为高/低活性
📊 自动生成结果图表：
回归：真实 vs 预测散点图 + 不确定性误差棒
分类：特征权重条形图 + ROC 曲线
🌐 无需页面刷新，使用 HTMX 实现流畅交互

# 🛠️ 本地部署指南
## 1. 克隆项目

 ```bash
git clone https://github.com/yourname/qsar-web-app.git
cd qsar-web-app
 ```

## 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# 或
venv\Scripts\activate       # Windows
```

## 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 4. 安装 RDKit（关键步骤！）
###  推荐方式：使用 conda（最稳定）
```bash
conda install -c conda-forge rdkit
```
### 不推荐：pip install rdkit
PyPI 上的 rdkit 包通常无法正常工作！
如果你坚持用 pip，请参考 RDKit 官方安装指南。

## 5. 准备目录结构
确保项目根目录包含以下内容：
```text
qsar-web-app/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── file_preview.html
│   ├── regression_result.html
│   └── classification_result.html
├── static/
│   └── plots/          # 自动生成图表存放处
└── uploads/            # 上传文件临时存放处（程序会自动创建）

hint: uploads/ 和 static/plots/ 目录会在首次运行时自动创建。
```

## 6. 启动应用
```bash
python app.py
```

看到如下输出即表示成功：
``` text
Running on http://127.0.0.1:5000
```

打开浏览器访问：http://127.0.0.1:5000

# 📥 使用说明
## 1. 准备 CSV 文件
示例格式（列名不区分大小写）：
```
smiles	pic50
CC(=O)OC	6.2
CN1C=NC2=C1C...	7.8
...	...
```
### 要求：
必须有一列包含 smile（如 smiles, SMILES, Smile）<br>
必须有一列包含活性值，列名需含 act、value、target、y、pic50 等关键词<br>
活性值必须是数字（可含单位，程序会提取数字部分）

## 2. 上传与训练
访问首页 → 选择 CSV 文件<br>
选择任务类型：Regression 或 Classification<br>
点击 “上传并预览”<br>
预览数据后，点击 “开始训练模型”<br>
等待几秒/分钟（显示“训练中，请稍后...”）<br>
查看结果图表与指标

# 📝 注意事项
⚠️ 上传超大文件（>10,000 行）可能导致浏览器卡顿或内存不足<br>
🖼️ 所有生成的图表保存在 static/plots/，可手动清理<br>
🔒 本应用为单机开发版，不可直接用于生产环境（无用户隔离、无并发控制）<br>
🐍 Python 版本建议：3.9-3.12

# 📬 问题反馈
### 如遇问题，请检查：

RDKit 是否正确安装（在 Python 中 from rdkit import Chem 能否导入？）<br>
CSV 列名是否符合要求<br>
活性值是否为有效数字
####  欢迎提交 Issue 或联系开发者！

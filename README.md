# CV-pj1

本项目使用 NumPy 从零实现三层 MLP 分类器，在 EuroSAT RGB 遥感图像数据集上完成 10 类地表覆盖分类。代码手动实现前向传播、Softmax 交叉熵、L2 正则化、反向传播、SGD 更新和学习率衰减，不使用 PyTorch、TensorFlow、JAX 等自动微分框架。

## 环境依赖

```bash
pip install numpy pillow matplotlib tqdm
```

## 数据准备

将 EuroSAT RGB 数据集的 10 个类别文件夹放在 `./data` 下，例如：

```text
data/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

数据加载时会按类别分层划分为训练集、验证集和测试集，比例为 `70% / 15% / 15%`。默认设置下，图像会缩放到 `32x32`，缓存到 `output/cache/`，并使用训练集均值和标准差进行标准化，以加速 MLP 优化。

## 快速训练

```bash
python quick_train.py --data-dir ./data --output-dir ./output/final_model
```

## 超参数搜索

运行顺序超参数搜索，并在搜索后自动使用最优组合进行最终训练：

```bash
python train.py --data-dir ./data --output-dir ./output/final_model --search
```
## 最终训练

本实验报告中使用的最终配置如下：

```bash
python train.py --data-dir ./data --output-dir ./output/final_model \
  --image-size 32 --batch-size 256 --epochs 80 \
  --hidden 512,128 --activation relu --lr 0.08 \
  --lr-decay 0.98 --weight-decay 0.005
```

## 激活函数切换

模型支持在两层隐藏层中切换激活函数，可选值为：

```text
relu
tanh
sigmoid
```

使用 `--activation` 参数切换

## 测试评估

加载保存好的最佳模型，在独立测试集上输出准确率和混淆矩阵：

```bash
python test.py --data-dir ./data \
  --model-path ./output/final_model/model.npy \
  --output-dir ./output/final_model
```

## 主要文件结构

```text
load_data.py        数据加载、划分、预处理和缓存
model.py            三层 MLP、手写反向传播、SGD 训练
train.py            训练入口、学习率衰减、超参数搜索、保存最佳模型
quick_train.py      快速训练入口
test.py             测试集评估、混淆矩阵、错例图
utils.py            准确率、混淆矩阵、画图工具函数
plot/plot.py        生成训练曲线、搜索曲线、第一层权重图
plot/plot_image.py  生成数据集样例图
```
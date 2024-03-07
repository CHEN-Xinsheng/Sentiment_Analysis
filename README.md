# Sentiment Analysis

> 陈鑫圣&emsp;计算机科学与技术系 计13&emsp;2021010760

***

This is a project for the course "Introduction to Artificial Intelligence" in Tsinghua University, 2023. The project is to implement sentiment analysis model using CNN, LSTM, GRU, MLP and word2vec. The goal is to classify the reviews into positive and negative.

|           | Accuracy | F1-score |
| :-------: | :------: | :------: |
|    CNN    |  0.8726  |  0.8712  |
| LSTM(RNN) |  0.8428  |  0.8497  |
| GRU(RNN)  |  0.8347  |  0.8407  |
|    MLP    |  0.8184  |  0.8134  |

Refer to the [report](./report/2021010760_陈鑫圣.pdf) for more details.

## 运行程序

训练并可视化展示结果（默认使用 CNN 模型）

```
python main.py
```

查看帮助

```
python main.py -h
```

更多使用案例

```
python main.py -nw           # 不使用 wandb 进行可视化
python main.py -m MLP -e 25  # 使用 MLP 模型，epoch=25
```

## 文件结构

### `config.py` 

该文件定义了程序的一些常数、默认参数和超参数。

### `data_preprocess.py`

数据预处理。不需要手动运行。

### `model.py`

定义了本次实验使用的神经网络模型。

### `main.py`

主程序。

### `Dataset`

训练集、验证集、测试集语料库，预训练词向量文件。

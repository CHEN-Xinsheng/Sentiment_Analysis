import torch
import torch.nn as nn
import gensim
import numpy as np
import argparse
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR 

from data_preprocess import *
from model import CNN, LSTM, GRU, MLP
from config import *



# def get_dataloader(batch_size: int, num_workers: int):
#     """
#     获取训练集、验证集、测试集的 DataLoader。这些 Dataloader 返回一个二维向量，每个元素是每个词语的词向量。
#     """
#     train_set = My_Dataset(dataset_path=TRAIN_FILE, word2vec_path=WORD2VEC_FILE, embedding_dim=EMBEDDING_DIM, max_sentence_len=MAX_SENTENCE_LEN)
#     valid_set = My_Dataset(dataset_path=VALID_FILE, word2vec_path=WORD2VEC_FILE, embedding_dim=EMBEDDING_DIM, max_sentence_len=MAX_SENTENCE_LEN)
#     test_set  = My_Dataset(dataset_path=TEST_FILE,  word2vec_path=WORD2VEC_FILE, embedding_dim=EMBEDDING_DIM, max_sentence_len=MAX_SENTENCE_LEN)
#     train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_dataloader  = DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     return train_dataloader, valid_dataloader, test_dataloader



def get_dataloader(batch_size: int, num_workers: int):

    train_text_in_id = text2id(TRAIN_FILE, word2id=word2id, max_sentence_len=MAX_SENTENCE_LEN)
    valid_text_in_id = text2id(VALID_FILE, word2id=word2id, max_sentence_len=MAX_SENTENCE_LEN)
    test_text_in_id  = text2id(TEST_FILE , word2id=word2id, max_sentence_len=MAX_SENTENCE_LEN)

    train_labels = get_labels(TRAIN_FILE)
    valid_labels = get_labels(VALID_FILE)
    test_labels  = get_labels(TEST_FILE)

    # train_dataset = TensorDataset(torch.from_numpy(train_text_in_id).type(torch.float),
    train_dataset = TensorDataset(torch.from_numpy(train_text_in_id).type(torch.long),
                                  torch.from_numpy(train_labels)    .type(torch.long))
    # valid_dataset = TensorDataset(torch.from_numpy(valid_text_in_id).type(torch.float), 
    valid_dataset = TensorDataset(torch.from_numpy(valid_text_in_id).type(torch.long), 
                                  torch.from_numpy(valid_labels)    .type(torch.long))
    # test_dataset  = TensorDataset(torch.from_numpy(test_text_in_id) .type(torch.float), 
    test_dataset  = TensorDataset(torch.from_numpy(test_text_in_id) .type(torch.long), 
                                  torch.from_numpy(test_labels)     .type(torch.long))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, valid_dataloader, test_dataloader



def get_embedding():
    """
    获取预训练好的词向量。返回一个 Embedding 层。它将词语的 id 转换为 词向量
    """

    # gensim_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True)
    # weights = torch.FloatTensor(gensim_model.vectors)
    # embedding = nn.Embedding.from_pretrained(weights)
    # embedding.weight.requires_grad = config.update_word2vec

    # return embedding

    # embedding = nn.Embedding(len(word2id) + 1, EMBEDDING_DIM)
    embedding = nn.Embedding(len(id2vec), EMBEDDING_DIM)  # len(id2vec) = len(word2id) + 1
    embedding.weight.data.copy_(torch.from_numpy(id2vec))
    embedding.weight.requires_grad = config.update_word2vec

    return embedding




def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment classification.', allow_abbrev=True)
    
    # 添加命令行参数
    parser.add_argument('-m', '--model', 
                        dest='model', 
                        type=str,
                        default='CNN', choices=['CNN', 'LSTM', 'GRU', 'MLP'], 
                        help="The model used in this program, choose from ['CNN', 'LSTM', 'GRU', 'MLP'].")
    parser.add_argument('-lr', '--learning-rate-init', 
                        dest='learning_rate', 
                        type=float, 
                        default=LEARNING_RATE_INIT,
                        help=f"Initial learing rate, default = {LEARNING_RATE_INIT}.")
    parser.add_argument('-e', '--epoch', 
                        dest='epoch', 
                        type=int, 
                        default=EPOCH, 
                        help='Epoch of training.')
    parser.add_argument('-b', '--batch-size', 
                        dest='batch_size', 
                        type=int, 
                        default=BATCH_SIZE, 
                        help="Batch size.")
    parser.add_argument('-k', '--kernel-num', 
                        dest='kernel_num', 
                        type=int,
                        default=KERNEL_NUM, 
                        help="Number of kernel of each size.")
    parser.add_argument('-d', '--dropout', 
                        dest='dropout', 
                        type=float, 
                        default=DROPOUT, 
                        help="Dropout.")
    parser.add_argument('-nu', '--not-update-word2vec',
                        dest='update_word2vec',
                        action="store_false",
                        default=UPDATE_WORD2VEC,
                        help="No not update word2vec during training.")
    parser.add_argument('-hs', '--hidden-size', 
                        dest='hidden_size',
                        type=int,
                        default=HIDDEN_SIZE,
                        help="Dimension(size) of hidden layer of LSTM, GRU and MLP.")
    parser.add_argument('-l', '--num-layers', 
                        dest='num_layers', 
                        type=int,
                        default=NUM_LAYERS,
                        help="Number of hidden layer of LSTM and GRU.")
    parser.add_argument('-s', '--step-size', 
                        dest='step_size', 
                        type=int,
                        default=SCHEDULER_STEP_SIZE,
                        help="Step size of scheduler.")
    parser.add_argument('-nbd', '--not-bidirectional', 
                        dest='bidirectional', 
                        action="store_false",
                        default=BIDIRECTIONAL,
                        help="Not Bidirectional (LSTM and GRU).")
    parser.add_argument('-nw', '--not-wandb',
                        dest='wandb',
                        action="store_false",
                        default=True,
                        help="Do not use wandb.")
    # 从命令行中解析参数
    config = parser.parse_args()

    return config


def train(dataloader):
    """
    对模型进行一次训练，并返回 acc 和 f1。
    """
    # 将模型置于训练模式
    model.train() 

    # train_loss, train_acc = 0.0, 0.0
    total, correct = 0, 0
    true = []
    pred = []
    for _, (inputs, labels) in enumerate(dataloader):
        # inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # 清空梯度缓存
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward() 
        optimizer.step() # 更新模型参数

        # train_loss += config.batch_size * loss.item()

        correct_tensor = (output.argmax(1) == labels)
        correct += correct_tensor.float().sum().item()
        
        total += len(inputs)
        # full_true.extend(labels.cpu().numpy().tolist())
        true.extend(labels.cpu().numpy())
        pred.extend(output.argmax(1).cpu().numpy())

    # train_loss /= len(dataloader.dataset)
    acc = correct / total
    f1 = f1_score(np.array(true), np.array(pred), average="binary")
    # train_f1 = f1_score(full_true, full_pred, average="binary")

    # 调整学习率
    scheduler.step() 
    return acc, f1


def evaluate(dataloader):
    """
    对模型进行一次测试，并返回 acc 和 f1。
    """
    # 将模型置于测试模式
    model.eval()

    total, correct = 0, 0
    true = []
    pred = []
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            # loss = criterion(output, labels)

            correct_tensor = (output.argmax(1) == labels)
            correct += correct_tensor.float().sum().item()

            total += len(inputs)
            true.extend(labels.cpu().numpy())
            pred.extend(output.argmax(1).cpu().numpy())

    acc = correct / total
    f1 = f1_score(np.array(true), np.array(pred), average="binary")
    return acc, f1


if __name__ == "__main__":

    # 运行设备：CPU/GPU 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[finish] DEVICE =", "cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 处理命令行参数
    config = parse_args()

    # 生成“词汇-id” 表，和 “id-词向量” 表
    word2id = gen_word2id([TRAIN_FILE, VALID_FILE])
    id2vec = gen_id2vec(WORD2VEC_FILE, word2id)

    # 读取数据
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(config.batch_size, NUM_WORKERS)
    print("[finish] get_dataloader")

    # 基于预训练好的词向量,创建 Embedding 层
    embedding = get_embedding()
    
    # 建立模型
    if config.model == 'CNN':
        model = CNN (config=config, embedding=embedding, kernel_sizes=KERNEL_SIZES, num_classes=NUM_CLASSES)
    elif config.model == 'LSTM':
        model = LSTM(config=config, embedding=embedding, num_classes=NUM_CLASSES)
    elif config.model == 'GRU':
        model = GRU (config=config, embedding=embedding, num_classes=NUM_CLASSES)        
    elif config.model == 'MLP':
        model = MLP (config=config, embedding=embedding, num_classes=NUM_CLASSES, max_sentence_len=MAX_SENTENCE_LEN)
    else:
        print(f'Bad param "model", received {config.model}.')
        exit(0)
    model = model.to(device)
    print(f"[finish] model = {config.model}")
    print(f"epoch = {config.epoch}")
    print(f"update_word2vec = {config.update_word2vec}")
    print(f"learning_rate = {config.learning_rate}")
    print(f"step_size = {config.step_size}")
    print(f"dropout = {config.dropout}")
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=config.step_size)  # 在每 step_size 个 epoch 之后将学习率乘以 gamma=0.1。
    print("[finish] optimizer")
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    print("[finish] criterion")

    # 使用 wandb 记录训练过程
    if config.wandb:
        wandb.init(
            project = "Sentiment_Classification", 
            name = f"{config.model}|epoch-{config.epoch}|update-{config.update_word2vec}|lr-{config.learning_rate}|step-{config.step_size}|dropout-{config.dropout}",
            config = {
                "learning_rate": config.learning_rate, 
                "epoch": config.epoch
            }
        )
        print("[finish] wandb.init")

    # 进行训练和测试
    for epoch in tqdm(range(config.epoch)):
        train_acc, train_f1 = train(train_dataloader)
        valid_acc, valid_f1 = evaluate(valid_dataloader)
        test_acc, test_f1 = evaluate(test_dataloader)

        lineared = {
            "train_acc": train_acc,
            "train_f1": train_f1,
            "valid_acc": valid_acc,
            "valid_f1": valid_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
        if config.wandb:
            wandb.log(lineared)
        print(f"Epoch {epoch + 1} of {config.epoch}: ", lineared)


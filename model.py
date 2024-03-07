import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *



class CNN(nn.Module):
    def __init__(self, config, embedding: nn.Embedding, kernel_sizes, num_classes: int):
        super().__init__()
        
        self.embedding = embedding
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels = 1,
                       out_channels = config.kernel_num,
                       kernel_size = (kernel_size, embedding.embedding_dim))
            for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(len(kernel_sizes) * config.kernel_num, num_classes)  # kernel 的总数为 len(kernel_sizes) * config.kernel_num

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [Wrong ?]
        # x = self.embedding(x.to(torch.int64)).unsqueeze(1)
        # unsqueeze(1) 是在第 1 个维度上插入（ a*b*c 变为 a*1*b*c）。因为 nn.Conv2d 接收的输入是一个 4 维张量： nSamples * nChannels * Height * Width 。

        # x = x.unsqueeze(1)
        # unsqueeze(1) 是在第 1 个维度上插入（ a*b*c 变为 a*1*b*c）。因为 nn.Conv2d 接收的输入是一个 4 维张量： nSamples * nChannels * Height * Width 。

        conved = [conv(x) for conv in self.convs]
        # 每个元素是 4 维张量，形状为 (batch_size, kernel_num, out_sequence_length, 1)

        relued = [F.relu(x).squeeze(3) for x in conved]
        # 每个元素是 3 维张量，形状为 (batch_size, kernel_num, out_sequence_length)

        pooled = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in relued]
        # 每个元素是 2 维张量，形状为 (batch_size, kernel_num)

        catted = self.dropout(torch.cat(pooled, dim=1))
        # 2 维张量，形状为 (batch_size, len(kernel_sizes) * kernel_num)

        lineared = self.linear(catted)
        # 2 维张量，形状为 (batch_size, num_classes)

        softmaxed = F.softmax(lineared, dim=1)
        # 2 维张量，形状为 (batch_size, num_classes)

        return softmaxed
    

class LSTM(nn.Module):
    def __init__(self, config, embedding: nn.Embedding, num_classes: int):
        super().__init__()

        self.config = config 
        self.D = (2 if config.bidirectional else 1)
        self.embedding = embedding

        self.lstm = nn.GRU(
            input_size=embedding.embedding_dim, # 本实验中是 50
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first = True   # 输入和输出的张量形状为 (batch, sequence_length, feature)
        )        
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size * self.D, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        # embedded 形状是 (batch, sequence_length, embedding_dim)

        _, (h_n, _) = self.lstm(input=embedded)
        # h_n 形状是 (num_layers * D, batch, hidden_size)
         
        h_n = h_n.view(self.config.num_layers, self.D, -1, self.config.hidden_size)
        # h_n 形状是 (num_layers, D, batch, hidden_size)

        if self.D == 2:
            dropped = self.dropout(torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=-1))
        else:
            dropped = self.dropout(h_n[-1, 0, :, :])
        # dropped 形状是 (batch, hidden_size * D) （self.dropout 的输入形状相同）

        lineared = self.linear(dropped)
        # dropped 形状是 (batch, num_classes) 

        return lineared
        # return lineared[-1].squeeze(1) if lineared.dim() == 3 else lineared.squeeze(1)


class GRU(nn.Module):
    def __init__(self, config, embedding: nn.Embedding, num_classes: int):
        super().__init__()

        self.config = config
        self.D = (2 if config.bidirectional else 1)
        self.embedding = embedding

        self.gru = nn.GRU(
            input_size=embedding.embedding_dim, # 本实验中是 50
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first = True   # 输入和输出的张量形状为 (batch, sequence_length, feature)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size * self.D, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        # embedded 形状是 (batch, sequence_length, embedding_dim)

        _, h_n = self.gru(input=embedded)
        # h_n 形状是 (num_layers * D, batch, hidden_size)
         
        h_n = h_n.view(self.config.num_layers, self.D, -1, self.config.hidden_size)
        # h_n 形状是 (num_layers, D, batch, hidden_size)

        if self.D == 2:
            dropped = self.dropout(torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=-1))
        else:
            dropped = self.dropout(h_n[-1, 0, :, :])
        # dropped 形状是 (batch, hidden_size * D) （self.dropout 的输入形状相同）

        lineared = self.linear(dropped)
        # dropped 形状是 (batch, num_classes) 

        return lineared


class MLP(nn.Module):
    def __init__(self, config, embedding: nn.Embedding, num_classes: int, max_sentence_len: int) -> None:
        super().__init__()

        self.embedding = embedding
        self.fc1 = nn.Linear(embedding.embedding_dim * max_sentence_len, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, num_classes)
        self.droupout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.embedding(x)
        # x 形状是 (batch, sequence_length, embedding_dim)

        x = x.view(x.size(0), -1)
        # x 形状是 (batch, sequence_length * embedding_dim)

        x = F.relu(self.fc1(x))
        # x 形状是 (batch, hidden_size)

        x = self.droupout(x)

        x = F.relu(self.fc2(x))
        # x 形状是 (batch, num_classes)

        return x


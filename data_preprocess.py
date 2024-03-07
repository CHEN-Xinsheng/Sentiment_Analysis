import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from config import *


def gen_word2id(files):
    """
    将（训练集、测试集的）所有词语编码， id 区间是 [1, len(word2id)]
    """
    word2id = dict()
    for file in files:
        # with open(file, encoding='utf-8', errors='ignore') as f:
        with open(file, encoding='utf-8') as f:
            for line in f.readlines():
                sentence = line.strip().split('\t')[1].split()
                for word in sentence:
                    # if word not in word2id.keys():
                    if word not in word2id:
                        word2id[word] = len(word2id) + 1
    return word2id


def gen_id2vec(word2vec_file, word2id):
    """
    生成“词语 id -词向量”的对应表。
    词向量表中不存在的词，词向量默认为 [0, 0, ..., 0]
    id = 0 ，词向量默认为 [0, 0, ..., 0]
    """
    word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    # word2vecs = np.array(np.zeros([len(word2id) + 1, preModel.vector_size]))  # preModel.vector_size == 50
    id2vec = np.zeros([len(word2id) + 1, EMBEDDING_DIM])

    for word in word2id:
        id = word2id[word]
        try:
            id2vec[id] = word2vec[word]
        except:
            # 词向量表中如果不存在这个词，那么它的词向量默认为 [0, 0, ..., 0]
            pass
    return id2vec


def text2id(file, word2id, max_sentence_len):
    """
    将语料库的文本转换成 id，返回一个二维数组，每一个元素是一个一维数组，表示一个句子的所有词的 id
    """
    sentences_in_id = np.array([0] * max_sentence_len)
    with open(file, encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            sentence = line.split('\t')[1].split()
            sentence_in_id = [word2id.get(word, 0) for word in sentence]
            # 将长度置为 max_sentence_len
            if len(sentence_in_id) < max_sentence_len:
                sentence_in_id += [0] * (max_sentence_len - len(sentence_in_id))
            else:
                sentence_in_id = sentence_in_id[:max_sentence_len]

            sentence_in_id = np.array(sentence_in_id)
            sentences_in_id = np.vstack([sentences_in_id, sentence_in_id]) # 按垂直方向（行顺序）堆叠数组
    sentences_in_id = np.delete(sentences_in_id, 0, axis=0) # 删除首行
    return sentences_in_id


def get_labels(file):
    """
    获取语料库的 labels 列表
    """
    labels = np.array([])
    with open(file, encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            label = int(line.strip().split('\t')[0])
            labels = np.append(labels, label)
    return labels



# class My_Dataset(Dataset):

#     def __init__(self, dataset_path, word2vec_path, embedding_dim, max_sentence_len):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.max_sentence_len = max_sentence_len
#         self.data = []
#         self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             for line in f.readlines():
#                 label, senetnce = line.split('\t')
#                 label = int(label)
#                 senetnce = senetnce.split()[:max_sentence_len]  # 每个句子只取前 max_sentence_len 个词
#                 if len(senetnce) < max_sentence_len:
#                     senetnce += [''] * (max_sentence_len - len(senetnce))
#                 self.data.append((senetnce, label))
#         # print(f"len = {len(self.data)}")
    
#     def __getitem__(self, index):
#         # print(f"__getitem__(self, index), index = {index}, len = {len(self.data)}")
#         # try:
#         #     words = self.data[index][0]
#         #     label = torch.tensor(self.data[index][1])
#         # except Exception as e:
#         #     print("****************************************")
#         #     print(e)
#         #     print("****************************************")
    
#         label = torch.tensor(self.data[index][1])

#         sentence = self.data[index][0]
#         # return sentence, label
#         sentence_vec = []
#         for word in sentence:
#             if word in self.word2vec:
#                 sentence_vec.append(self.word2vec[word])
#             else:
#                 sentence_vec.append([0] * self.embedding_dim)
#         sentence_vec = torch.cat([torch.FloatTensor(np.array(sentence_vec)),
#                           torch.zeros(self.max_sentence_len - len(sentence_vec), self.embedding_dim)],
#                           dim=0)
#         # return data, label, len(vecs)
#         return sentence_vec, label

#     def __len__(self):
#         return len(self.data)
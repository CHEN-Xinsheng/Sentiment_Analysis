from pathlib import Path



ROOT_DIR = Path.cwd()
TRAIN_FILE = ROOT_DIR / 'Dataset' / 'train.txt'
VALID_FILE = ROOT_DIR / 'Dataset' / 'validation.txt'
TEST_FILE = ROOT_DIR / 'Dataset' / 'test.txt'
WORD2VEC_FILE = ROOT_DIR / 'Dataset' / 'wiki_word2vec_50.bin'

# 常数
EMBEDDING_DIM = 50          # 每个词向量的维数
NUM_CLASSES = 2             # 划分的种类数（本实验为二分类）

# 默认参数
EPOCH = 10                  # 模型训练的 epoch 数目
LEARNING_RATE_INIT = 0.001  # 初始学习率
BATCH_SIZE = 40             # 模型初始化参数
NUM_WORKERS = 2             # 模型初始化参数
KERNEL_NUM = 20             # 相同大小的卷积核的个数
DROPOUT = 0.2               # dropout 的概率
UPDATE_WORD2VEC = True      # 是否在训练中更新词向量
HIDDEN_SIZE = 256           # LSTM, GRU, MLP 的隐藏层维度
NUM_LAYERS = 2              # LSTM, GRU 的隐藏层数量
BIDIRECTIONAL = True        # LSTM, GRU 是否双向

# 超参数
KERNEL_SIZES = [2, 3, 4]    # 卷积核的大小 
MAX_SENTENCE_LEN = 60       # 每个句子的最长词数
SCHEDULER_STEP_SIZE = 5     # 在每 step_size 个 epoch 之后将学习率乘以 gamma

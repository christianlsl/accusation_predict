import numpy as np
import jieba
import jieba.analyse
import torch
import pickle
import re
import torch.nn as nn
import torch.nn.functional as F
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

class GlobalMaxPool1d(nn.Module):
  # 通过普通的池化来实现全局池化
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 195)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embedding_dim,
                                        out_channels = c,
                                        kernel_size = k))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.long())
        embeds = embeds.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return x


def get_sequence(text, num_words=80000, topK=20, word_len=2, maxlen=400):
    digit_clear = re.compile("[\w.]*\d[\w.]*")
    words_cut = []
    jieba.analyse.set_stop_words(".\chinesestopword.txt")
    text_cut = jieba.analyse.extract_tags(text, topK=topK, allowPOS=('n', 'v'))
    words = []
    for word in text_cut:
        if len(word) >= word_len:
            words.append(word)
        if re.search(digit_clear, word) is not None:
            text_cut.remove(word)
    words_cut.append(text_cut)

    with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
        tokenizer_fact = pickle.load(f)  # 读取Tokenizer
    sequence = tokenizer_fact.texts_to_sequences(texts=words_cut)
    sequence = pad_sequences(sequence, maxlen=maxlen,
                                            padding='post', value=0, dtype='int')
    device = "cuda" if torch.cuda.is_available else "cpu"
    return np.array(sequence)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available else "cpu"
    with open('./data_deal/accusation_type_list.pkl', mode='rb') as f:
        accusation_type_list = pickle.load(f)

    fact = np.load('./data_deal/big_fact_pad_seq_80000_400.npy')
    while True:
        a = input("请输入需要预测罪名的案件详情,退出请输入0\n")
        sequence = get_sequence(a, num_words=80000, topK=30, word_len=2, maxlen=400)
        print(fact)
        print(sequence)
        if sequence in fact:
            print("1")


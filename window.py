from tkinter import *
import jieba
import jieba.analyse
import torch
import pickle
import re
import tkinter as tk
import torch.nn as nn
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences


class GlobalMaxPool1d(nn.Module):
    # 通过普通的池化来实现全局池化
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


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
            self.convs.append(nn.Conv1d(in_channels=embedding_dim,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence.long())
        embeds = embeds.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


class Application(tk.Frame):
    def __init__(self, master=None, model=None, accusation_type_list=None):
        self.master = master
        self.createWidget()
        self.model = model
        self.accusation_type_list = accusation_type_list

    def get_sequence(self, text, num_words=80000, topK=20, word_len=2, maxlen=400):
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
        return torch.Tensor(sequence).to(device)

    def predict(self):
        a = self.w1.get('1.0',END)
        sequence = self.get_sequence(a, num_words=80000, topK=30, word_len=2, maxlen=400)
        out = self.model(sequence)
        out = int(out.argmax(dim=1))
        self.label.config(text=self.accusation_type_list[out])

    def createWidget(self):
        self.w1 = Text(self.master, width=60, height=12)
        # 宽度 20 个字母(10 个汉字)，高度一个行高
        self.w1.pack()
        self.w1.insert(1.0, "在此输入案情陈述")
        self.label = Label(self.master, text='', width=10, height=2,
                           bg='orange', fg='white')
        self.label.pack()
        Button(self.master, text="判断罪名", command=self.predict).pack()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available else "cpu"
    with open('./data_deal/accusation_type_list.pkl', mode='rb') as f:
        accusation_type_list = pickle.load(f)
    state_dict = torch.load('./model/model_dict_26_acc_0.646640826873385.pth',map_location=torch.device(device))
    num_words = 80000
    embedding_dim, kernel_sizes, num_channels = 400, [3, 4, 5], [300, 300, 300]
    model = TextCNN(num_words, embedding_dim, kernel_sizes, num_channels).to(device)
    model.load_state_dict(state_dict)

    root = Tk()
    root.geometry("450x300+200+300")
    app = Application(root, model, accusation_type_list)
    root.mainloop()

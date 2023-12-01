import jieba
import jieba.analyse
import torch
import pickle
import re
import torch.nn as nn
from keras_preprocessing.sequence import pad_sequences


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


def get_sequence(text, num_words=80000, topK=20, word_len=2, maxlen=800):
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


if __name__ == '__main__':

    with open('./data_deal/accusation_type_list.pkl', mode='rb') as f:
        accusation_all_list = pickle.load(f)
    print(accusation_all_list)
    model = torch.load('./model/dnn2.pth')
    while True:
        a = input("请输入需要预测罪名的案件详情,退出请输入0")
        if a == '0':
            break
        sequence = get_sequence(a, num_words=80000, topK=20, word_len=2, maxlen=800)
        out = int(model(sequence).argmax(dim=1))
        print(out)

        print(accusation_all_list[out])

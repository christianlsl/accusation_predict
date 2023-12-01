import numpy as np
import re
import jieba
import jieba.analyse
import json
import pickle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def stopwordslist():  # 读入停用词词表
    stopwords = [line.strip() for line in open(
        'chinesestopword.txt', encoding='UTF-8').readlines()]
    return stopwords


def create_label(label, label_set):
    '''
    构建标签one-hot
    :param label: 原始标签
    :param label_set: 标签集合
    :return: 标签one-hot
    eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
    '''
    label_zero = np.zeros(len(label_set))
    label_zero[np.in1d(label_set, label)] = 1
    return label_zero


def create_labels(path):
    accusation_all = []
    with open(path, 'r', encoding='utf8') as f:
        data_raw = f.readlines()
    for num, data_one in enumerate(data_raw):  # 读取文件
        try:
            case = json.loads(data_one)
            accusation = case['meta']['accusation'][0]
            accusation_all.append(accusation)
        except Exception as e:
            print(num, ": ", e)

    accusation_all_list = list(set(accusation_all))
    label_array = np.array(accusation_all_list)
    with open('./data_deal/accusation_type_list.pkl', mode='wb') as f:
        pickle.dump(label_array, f)
    print(label_array)
    labels_one_hot = list(map(lambda x: create_label(label=x, label_set=label_array), accusation_all))
    np.save('./data_deal/big_labels_accusation.npy', labels_one_hot)


def cut_texts(texts=None, word_len=2, topK=30):
    '''
    文本分词
    texts:文本列表
    word_len:保留词语长度
    topK:每篇文书中选取的特征词个数
    '''
    digit_clear = re.compile("[\w.]*\d[\w.]*")
    texts_cut = []
    jieba.analyse.set_stop_words(".\chinesestopword.txt")
    for one_text in texts:
        text_cut = jieba.analyse.extract_tags(one_text, topK=topK, allowPOS=('n', 'v'))
        words = []
        for word in text_cut:
            if len(word) >= word_len:
                words.append(word)
            if re.search(digit_clear, word) is not None:
                text_cut.remove(word)
        texts_cut.append(text_cut)
    return texts_cut


def fact_cut(path):  # 文字转换
    for i in range(16):  # 分批处理大文件
        fact_extraction = []
        with open(path, 'r', encoding='utf8') as f:
            data_raw = f.readlines()
        for num, data_one in enumerate(data_raw):  # 读取文件
            try:
                case = json.loads(data_one)
                fact = case['fact']
                fact_extraction.append(fact)
            except Exception as e:
                print(num, ": ", e)
        texts = fact_extraction[i * 10000:(i * 10000 + 10000)]
        del fact_extraction  # 删除，减少内存占用
        big_fact_cut = cut_texts(texts=texts, word_len=1, topK=15)
        with open('./data_deal/data_cut/big_fact_cut_%d_%d.pkl' % (i * 10000, i * 10000 + 10000), mode='wb') as f:
            pickle.dump(big_fact_cut, f)
        print('finish big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))


def tokenizer_to_sequence(num_words=80000, maxlen=400):
    # num_words: 保留的单词的最大数量，基于单词频率。只有出现频率最高的num_words-1个单词会被保存
    # train tokenizer
    tokenizer_fact = Tokenizer(num_words=num_words)
    for i in range(16):
        print('start big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))
        with open('./data_deal/data_cut/big_fact_cut_%d_%d.pkl' % (i * 10000, i * 10000 + 10000),
                  mode='rb') as f:
            big_fact_cut = pickle.load(f)
        texts_cut_len = len(big_fact_cut)
        n = 0
        # 分批训练
        while n < texts_cut_len:
            tokenizer_fact.fit_on_texts(texts=big_fact_cut[n:n + 10000])  # 学习出文本的字典
            n += 10000
            if n < texts_cut_len:
                print('tokenizer finish fit %d samples' % n)
            else:
                print('tokenizer finish fit %d samples' % texts_cut_len)
        print('finish big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))

    with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='wb') as f:
        pickle.dump(tokenizer_fact, f)  # 保存Tokenizer模型

    with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
        tokenizer_fact = pickle.load(f)

    # texts_to_sequences
    for i in range(16):
        print('start big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))
        with open('./data_deal/data_cut/big_fact_cut_%d_%d.pkl' % (i * 10000, i * 10000 + 10000),
                  mode='rb') as f:
            big_fact_cut = pickle.load(f)
        # 分批执行 texts_to_sequences
        big_fact_seq = tokenizer_fact.texts_to_sequences(texts=big_fact_cut)
        # 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)]
        with open('./data_deal/fact_seq/fact_seq_%d_%d.pkl' % (i * 10000, i * 10000 + 10000), mode='wb') as f:
            pickle.dump(big_fact_seq, f)
        print('finish big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))

    # pad_sequences
    for i in range(16):
        print('start big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))
        with open('./data_deal/fact_seq/fact_seq_%d_%d.pkl' % (i * 10000, i * 10000 + 10000), mode='rb') as f:
            big_fact_seq = pickle.load(f)
        texts_cut_len = len(big_fact_seq)
        n = 0
        fact_pad_seq = []
        # 分批执行pad_sequences
        while n < texts_cut_len:
            fact_pad_seq += list(pad_sequences(big_fact_seq[n:n + 10000], maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
            # 将文本序列列表 (每个序列为整数列表) 转换成一个 2D Numpy数组，数组形状为 (big_fact_seq, maxlen)
            n += 10000
            if n < texts_cut_len:
                print('finish pad_sequences %d samples' % n)
            else:
                print('finish pad_sequences %d samples' % texts_cut_len)
        with open('./data_deal/fact_pad_seq/fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * 10000, i * 10000 + 10000),
                  mode='wb') as f:
            pickle.dump(fact_pad_seq, f)

    # 汇总pad_sequences
    fact_pad_seq = []
    for i in range(16):
        print('start big_fact_cut_%d_%d' % (i * 10000, i * 10000 + 10000))
        with open('./data_deal/fact_pad_seq/fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * 10000, i * 10000 + 10000),
                  mode='rb') as f:
            fact_pad_seq += pickle.load(f)
    fact_pad_seq = np.array(fact_pad_seq)
    np.save('./data_deal/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen), fact_pad_seq)

if __name__ == '__main__':
    create_labels("./data/data_test.json")
    fact_cut("./data/data_test.json")
    tokenizer_to_sequence(num_words=80000, maxlen=400)
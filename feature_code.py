import numpy as np
import torch
import torch.nn as nn
from keras.layers.convolutional import Conv2D

def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence.append(line.strip('\n'))

    k = 5
    kmer_list = []
    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            ind = index.index(sequence[number][i:i + k])
            seq.append(ind)
        kmer_list.append(seq)

    '''sum_length = 0
    cnt = 0
    for number in range(len(sequence)):
        sum_length += (len(sequence[number]) - k + 1)
        cnt = number
    average_length = round(sum_length / (cnt + 1))'''

    feature_word2vec = []
    for number in range(len(kmer_list)):
        #print(number)
        feature_seq = []
        for i in range(len(kmer_list[number])):
            kmer_index = kmer_list[number][i]
            for j in word2vec[kmer_index].tolist():
                feature_seq.append(j)
        feature_seq_tensor = torch.Tensor(feature_seq)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor = torch.unsqueeze(feature_seq_tensor, 0)
        feature_seq_tensor_avg = nn.AdaptiveAvgPool1d(1000 * 8)(feature_seq_tensor)

        feature_seq_numpy = feature_seq_tensor_avg.numpy()
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_list = feature_seq_numpy.tolist()

        feature_word2vec.append(feature_seq_list)

    return feature_word2vec


'''cell_lines = 'NHEK'
sets = 'train'
filename = 'data/' + cell_lines + '/' +sets + '/data.fasta'
# filename = 'data/' + cell_lines + '/' + element + '/test/test.fasta'
f = open('index_promoters.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')
word2vec = np.loadtxt('word2vec_promoters.txt')
feature_word2vec = word_embedding(filename, index, word2vec)
feature_word2vec = np.array(feature_word2vec)
print(feature_word2vec.shape)
np.savetxt('feature/' + cell_lines + '/' +sets + '/word2vec.txt', feature_word2vec)'''


cell_lines = 'K562'
filename = 'EPdata/' + cell_lines + '/data.fasta'
# filename = 'data/' + cell_lines + '/' + element + '/test/test.fasta'
f = open('index_promoters.txt', 'r')
index = f.read()
f.close()
index = index.strip().split(' ')
word2vec = np.loadtxt('word2vec_promoters.txt')
feature_word2vec = word_embedding(filename, index, word2vec)
feature_word2vec = np.array(feature_word2vec)
print(feature_word2vec.shape)
np.savetxt('EPfeature/' + cell_lines + '/word2vec.txt', feature_word2vec)

import numpy as np
import os, sys, re
import platform
import pickle
import itertools
from collections import Counter
from keras.models import load_model
import joblib
import torch
import torch.nn as nn
import argparse

# --------------------TPCP------------------------
def TPCP(sequence_list):
    file_name = 'tridnaPhyche.data'
    data_file = os.path.split(os.path.realpath(__file__))[
                    0] + r'\data\%s' % file_name if platform.system() == 'Windows' else \
        os.path.split(os.path.realpath(__file__))[0] + r'/data/%s' % file_name
    with open(data_file, 'rb') as handle:
        property_dict = pickle.load(handle)
    property_name = property_dict.keys()
    AA = 'ACGT'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 + '_' + p_name for p_name in property_name for aa1 in AA for aa2 in AA for aa3 in
                   AA]
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in sequence_list:
        normalized_code = []
        sequence = i
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                            AADict[
                                                                                                                sequence[
                                                                                                                    j]] * 16 +
                                                                                                            AADict[
                                                                                                                sequence[
                                                                                                                    j + 1]] * 4 +
                                                                                                            AADict[
                                                                                                                sequence[
                                                                                                                    j + 2]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        for p_name in property_name:
            for j in range(len(tmpCode)):
                normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
        encodings.append(normalized_code)
    return encodings


# --------------------PseTNC------------------------
def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    return CC / len(myPropertyName)


def get_kmer_frequency(sequence, kmer):
    baseSymbol = 'ACGT'
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency


def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def PseTNC(sequence_list):
    baseSymbol = 'ACGT'
    lamadaValue = 2
    weight = 0.1
    kmer = 3
    encodings = []
    myDiIndex = {
        'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
        'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
        'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
        'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
    }
    file_path = 'data/didnaPhyche.data'
    with open(file_path, 'rb') as f:
        myProperty = pickle.load(f)
    myIndex = myDiIndex
    myPropertyName = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
    myPropertyValue = myProperty
    for i in sequence_list:
        code = []
        sequence = i
        kmerFreauency = get_kmer_frequency(sequence, kmer)
        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
        for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
            code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
            code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings


# -----------------------RCKmer------------------------
def RC(kmer):
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    return ''.join([myDict[nc] for nc in kmer[::-1]])


def generateRCKmer(kmerList):
    rckmerList = set()
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    for kmer in kmerList:
        rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
    return sorted(rckmerList)


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def RCKmer(sequence_list):
    k = 5
    normalize = True

    encoding = []

    NA = 'ACGT'
    tmpHeader = []
    for kmer in itertools.product(NA, repeat=k):
        tmpHeader.append(''.join(kmer))
    header = generateRCKmer(tmpHeader)
    myDict = {}
    for kmer in header:
        rckmer = RC(kmer)
        if kmer != rckmer:
            myDict[rckmer] = kmer
    encoding = []
    for i in sequence_list:
        sequence = i
        kmers = kmerArray(sequence, k)
        for j in range(len(kmers)):
            if kmers[j] in myDict:
                kmers[j] = myDict[kmers[j]]
        count = Counter()
        count.update(kmers)
        if normalize == True:
            for key in count:
                count[key] = count[key] / len(kmers)
        code = []
        for j in range(len(header)):
            if header[j] in count:
                code.append(count[header[j]])
            else:
                code.append(0)
        encoding.append(code)
    return encoding


# --------------------Mismatch----------------------
def mismatch_count(seq1, seq2):
    mismatch = 0
    for i in range(min([len(seq1), len(seq2)])):
        if seq1[i] != seq2[i]:
            mismatch += 1
    return mismatch


def Mismatch(sequence_list):
    k = 5
    m = 1
    NN = 'ACGT'

    encoding = []
    template_dict = {}
    for kmer in itertools.product(NN, repeat=k):
        template_dict[''.join(kmer)] = 0

    for elem in sequence_list:
        sequence = elem
        kmers = kmerArray(sequence, k)
        tmp_dict = template_dict.copy()
        for kmer in kmers:
            for key in tmp_dict:
                if mismatch_count(kmer, key) <= m:
                    tmp_dict[key] += 1
        code = [tmp_dict[key] for key in sorted(tmp_dict.keys())]
        encoding.append(code)
    return encoding


def CKSNAP(sequence_list):
    gap = 5
    AA = 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    for i in sequence_list:
        sequence = i
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def nor_train_test(x_train, x_test):
    x = np.concatenate((x_train, x_test), axis=0)
    x = noramlization(x)
    x_train = x[0:len(x_train)]
    x_test = x[len(x_train):]
    return x_train, x_test


def word_embedding(sequence, index, word2vec):
    k = 5
    kmer_list = []
    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            ind = index.index(sequence[number][i:i + k])
            seq.append(ind)
        kmer_list.append(seq)
    feature_word2vec = []
    for number in range(len(kmer_list)):
        # print(number)
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


def word2vec(sequence_list):
    f = open('feature/' + species + '/index_promoters.txt', 'r')
    index = f.read()
    f.close()
    index = index.strip().split(' ')
    word2vec = np.loadtxt('feature/' + species + '/word2vec_promoters.txt')
    feature_word2vec = word_embedding(sequence_list, index, word2vec)
    feature_word2vec = np.array(feature_word2vec)
    return feature_word2vec


def calculate_label(final_prediction, test_num):
    final_prediction_label = np.zeros(test_num)
    for i in range(len(final_prediction)):
        if final_prediction[i] >= 0.5:
            final_prediction_label[i] = 1
        else:
            final_prediction_label[i] = 0
    return final_prediction_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='iPro-WAEL: a predictor to identify promoters in multiple species')
    parser.add_argument('-input', dest='inputfile', type=str, required=True,
                        help='Query DNA sequences in fasta format')
    parser.add_argument('-species', dest='species', type=str, required=True,
                        help='Species to be predicted')
    parser.add_argument('-output', dest='outputfile', type=str, required=True,
                        help='The path where you want to save the prediction results')
    args = parser.parse_args()

    filename = args.inputfile
    species = args.species
    outputfile = args.outputfile

    sequence_list = []
    name_list = []
    if not os.path.exists(filename):
        print('Error: file " %s " does not exist.' % filename)
        sys.exit(1)
    with open(filename) as f:
        record = f.readlines()
    if re.search('>', record[0]) is None:
        print('Error: the input file " %s " must be fasta format!' % filename)
        sys.exit(1)
    f = open(filename, 'r')
    for i in f.readlines():
        if i.strip() != '':
            if i.strip()[0] == '>':
                name_list.append(i.strip()[1:])
            else:
                sequence_list.append(i.strip().upper())
    f.close()

    feature_cksnap = CKSNAP(sequence_list)
    feature_mismatch = Mismatch(sequence_list)
    feature_rckmer = RCKmer(sequence_list)
    feature_psetnc = PseTNC(sequence_list)
    feature_tpcp = TPCP(sequence_list)
    x_test_rf = np.concatenate((feature_cksnap, feature_mismatch, feature_rckmer, feature_psetnc, feature_tpcp), axis=1)
    x_test_cnn = word2vec(sequence_list)
    x_test_cnn = np.expand_dims(x_test_cnn, axis=2)
    global x_train_rf
    # load features of training sets

    features = ['cksnap', 'mismatch', 'rckmer', 'psetnc', 'tpcp']
    for feature in features:
        feature_path = 'feature/' + species + '/' + feature + '.csv'
        fea = np.loadtxt(feature_path, delimiter=',')[:, 1:]
        if feature == 'cksnap':
            x_train_rf = fea
        else:
            x_train_rf = np.concatenate((x_train_rf, fea), axis=1)
    x_train_rf, x_test_rf = nor_train_test(x_train_rf, x_test_rf)

    RF_path = 'model/' + species + '/RF.model'
    CNN_path = 'model/' + species + '/model_cnn.hdf5'
    model_rf = joblib.load(RF_path)
    model_cnn = load_model(CNN_path)
    weight_list = {'human': [0.99027316, 0.00972684],
                   'arabidopsis': [0.03449942, 0.96550058],
                   'bacillus': [0, 1],
                   'ecoli K-12': [0.26110104, 0.73889896],
                   'ecoli s70': [0.54012514, 0.45987486],
                   'mouse': [0, 1],
                   'rhodobacter': [0.88174169, 0.11825831]
                   }

    rf_test_proba = model_rf.predict_proba(x_test_rf)[:, 1]  # probability
    rf_test_class = model_rf.predict(x_test_rf)  # class

    cnn_test_proba = model_cnn.predict(x_test_cnn)
    cnn_test_class = model_cnn.predict_classes(x_test_cnn)

    cnn_test_proba = np.squeeze(cnn_test_proba.reshape(1, -1))
    proba = weight_list[species][0] * rf_test_proba + weight_list[species][1] * cnn_test_proba  # proba 是预测概率
    label = calculate_label(proba, len(sequence_list))  # label 是预测的标签

    out = open(outputfile, 'w')
    for i in range(len(sequence_list)):
        out.writelines(name_list[i] + '\t' + sequence_list[i] + '\t' + str(proba[i]) + '\t')
        if label[i] == 1:
            out.writelines('promoter')
        else:
            out.writelines('non-promoter')
        out.writelines('\n')
    out.close()

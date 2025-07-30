import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5EncoderModel
import re
import torch
import os
from collections import Counter
import argparse

def file2str(filename):
    fr = open(filename)
    numline = fr.readlines()
    m = len(numline)
    index = -1
    A = []
    F = []
    for eachline in numline:
        index += 1
        if '>' in eachline:
            A.append(index)
    B = []
    for eachline in numline:
        line = eachline.strip()
        listfoemline = line.split()
        B.append(listfoemline)

    for i in range(len(A) - 1):
        K = A[i]
        input_sequence = B[K + 1]
        input_sequence = str(input_sequence)
        input_sequence = input_sequence[1:-1]
        for j in range(A[i + 1] - A[i]):
            if K < A[i + 1] - 2:
                C = str(B[K + 2])
                input_sequence = input_sequence + C[1:-1]
                K += 1
        input_sequence = input_sequence.replace('\'', '')
        F.append(input_sequence)
    return F

def fasta2num(filename):
    input_sequence = file2str(filename)
    X = []
    for i in range(len(input_sequence)):
        X.append(input_sequence[i])
    return X

def seq2vec(sequence_examples,model):

    sequence_examples = [sequence_examples]

    tokenizer = T5Tokenizer.from_pretrained('/home/gl/Desktop/ProstT5', do_lower_case=False)

    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    sequence_examples = ["<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
                         for s in sequence_examples
                         ]

    ids = tokenizer.batch_encode_plus(sequence_examples,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='pt').to(device)
    with torch.no_grad():
        embedding_repr = model(
            ids.input_ids,
            attention_mask=ids.attention_mask
        )

    emb_0 = embedding_repr.last_hidden_state  # shape (7 x 1024)
    emb_0_per_protein = emb_0[0].mean(dim=0)  # shape (1024)

    return emb_0_per_protein

if __name__ == '__main__':

    maxsequence = 5000

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_train = fasta2num('train.fasta')
    X_test = fasta2num('test.fasta')

    y_train = np.loadtxt('trainLabel.txt') - 1
    y_test = np.loadtxt('testLabel.txt') - 1

    X, X_val, Y, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=20, stratify=y_train)

    torch.save(Y, 'trainLabel.pt')
    torch.save(Y_val, 'valLabel.pt')
    torch.save(y_test, 'testLabel.pt')

    trainMat = torch.zeros((len(X),1024))
    valMat = torch.zeros((len(X_val),1024))
    testMat = torch.zeros((len(X_test),1024))

    model = T5EncoderModel.from_pretrained("/home/gl/Desktop/ProstT5").to(device)

    model.float() if device.type == 'cpu' else model.half()

    for i in range(len(X)):
        if len(X[i]) > maxsequence:
            u = X[i][:maxsequence]
            trainMat[i] = seq2vec(u, model)
        else:
            trainMat[i] = seq2vec(X[i],model)
    torch.save(trainMat, 'trainMat1024.pt')

    for i in range(len(X_val)):
        if len(X_val[i]) > maxsequence:
            u = X_val[i][:maxsequence]
            valMat[i] = seq2vec(u, model)
        else:
            valMat[i] = seq2vec(X_val[i], model)
    torch.save(valMat, 'valMat1024.pt')

    for i in range(len(X_test)):
        if len(X_test[i]) > maxsequence:
            u = X_test[i][:maxsequence]
            testMat[i] = seq2vec(u, model)
        else:
            testMat[i] = seq2vec(X_test[i], model)
    torch.save(testMat, 'testMat1024.pt')
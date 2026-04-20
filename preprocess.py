import numpy as np
import torch
import re
def vec_to_onehot(mat,m,n,pc,k):
    return_mat = np.zeros((m, k, 5000))
    for i in range(len(mat)):
        metrix = np.zeros((5000, k))
        for j in range(len(mat[i])):
            if j < 5000:
                metrix[j] = pc[mat[i][j]]
        return_mat[i,:,:] = np.transpose(metrix)
    return return_mat[:,:,:n]

def adjust_tensor(tensor, maxsequence):
    m = tensor.shape[0]
    if m < maxsequence:
        zeros = np.zeros((maxsequence - m, tensor.shape[1]), dtype=tensor.dtype)
        return np.concatenate([tensor, zeros], axis=0)
    elif m > maxsequence:
        return tensor[:maxsequence]
    else:
        return tensor

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

def str2dic(input_sequence):
    char = sorted(['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H'])
    char_to_index = {}
    index = 0
    result_index = []
    for c in char:
        char_to_index[c] = index
        index = index + 1
    for word in input_sequence:
        if word in char:
            result_index.append(char_to_index[word])
        else:
            result_index.append(0)
    return result_index

def fasta2num(filename):
    input_sequence = file2str(filename)
    X = []
    XX = []
    for i in range(len(input_sequence)):
        X.append(str2dic(input_sequence[i]))
        XX.append(input_sequence[i])
    return X,XX


def seq2vec(sequence_examples, model, tokenizer):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequence_examples = [sequence_examples]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    sequence_examples = ["<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
                         for s in sequence_examples]

    ids = tokenizer.batch_encode_plus(sequence_examples,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        embedding_repr = model(
            ids.input_ids,
            attention_mask=ids.attention_mask
        )

    emb_0 = embedding_repr.last_hidden_state
    emb_0_per_protein = emb_0[0].mean(dim=0)

    return emb_0_per_protein


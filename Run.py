import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from collections import Counter
from preprocess import fasta2num, vec_to_onehot
from model import *

MAX_SEQUENCE_LEN = 1000
INPUT_SIZE = 1024
HIDDEN_SIZES = [512, 256, 128]
NUM_CLASSES = 8
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
EMBED_DIM = 20

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess_data():
    X_train = fasta2num('Dataset/train.fasta')
    X_test = fasta2num('Dataset/test.fasta')
    y_train = np.loadtxt('Dataset/trainLabel.txt') - 1
    y_test = np.loadtxt('Dataset/testLabel.txt') - 1

    embed = np.loadtxt('autoEnconder.txt')
    embed = np.hstack((embed, np.zeros((EMBED_DIM, 1))))

    train_embed = vec_to_onehot(X_train, len(y_train), MAX_SEQUENCE_LEN, embed, EMBED_DIM)
    test_embed = vec_to_onehot(X_test, len(y_test), MAX_SEQUENCE_LEN, embed, EMBED_DIM)
    one_hot = np.eye(EMBED_DIM, EMBED_DIM)
    train_onehot = vec_to_onehot(X_train, len(y_train), MAX_SEQUENCE_LEN, one_hot, EMBED_DIM)
    test_onehot = vec_to_onehot(X_test, len(y_test), MAX_SEQUENCE_LEN, one_hot, EMBED_DIM)

    X_train_all = np.concatenate((train_embed, train_onehot), axis=1)
    X_test_all = np.concatenate((test_embed, test_onehot), axis=1)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_all, y_train, test_size=0.2, random_state=20, stratify=y_train
    )

    X_train_tensor = torch.from_numpy(X_train_split.astype(np.float32))
    X_val_tensor   = torch.from_numpy(X_val_split.astype(np.float32))
    X_test_tensor  = torch.from_numpy(X_test_all.astype(np.float32))

    y_train_tensor = torch.from_numpy(y_train_split).long()
    y_val_tensor   = torch.from_numpy(y_val_split).long()
    y_test_tensor  = torch.from_numpy(y_test).long()

    train_mat = torch.load('Dataset/trainMat1024.pt')
    val_mat   = torch.load('Dataset/valMat1024.pt')
    test_mat  = torch.load('Dataset/testMat1024.pt')

    return (
        X_train_tensor, y_train_tensor, train_mat,
        X_val_tensor, y_val_tensor, val_mat,
        X_test_tensor, y_test_tensor, test_mat
    )


def get_loader(X_tensor, y_tensor, batch_size, shuffle=True):

    dataset = torch.utils.data.TensorDataset(torch.arange(len(X_tensor)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_single_epoch(model, loader, X, y, labels, optimizer, criterion, mode='transformer'):
    model.train()
    total_loss = 0.0
    for batch_idx, (indices,) in enumerate(loader):
        batch_X = X[indices].to(DEVICE)
        batch_labels = labels[indices].to(DEVICE)
        optimizer.zero_grad()
        if mode == 'transformer':
            outputs = model.forward_transformer(batch_X)
        if mode == 'mlp':
            outputs = model.forward_mlp(y[indices].to(DEVICE))
        if mode == 'fusion':
            outputs = model(batch_X,y[indices].to(DEVICE))

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, X, y, labels):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for indices, in loader:
            batch_X = X[indices].to(DEVICE)
            batch_y = y[indices].to(DEVICE)
            batch_labels = labels[indices].to(DEVICE)
            outputs = model(batch_X, batch_y)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = np.mean(all_preds == all_targets)
    return acc, all_preds, all_targets

def Runtest(model, loader, X, y, labels):
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for indices, in loader:
            batch_X = X[indices].to(DEVICE)
            batch_y = y[indices].to(DEVICE)
            batch_labels = labels[indices].to(DEVICE)
            outputs = model(batch_X, batch_y)
            print(outputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(batch_labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = correct / total * 100
    recall = recall_score(all_labels, all_preds, average='macro')
    recall_each = recall_score(all_labels, all_preds, average=None)
    return accuracy, recall, recall_each

def main():
    (
        X_train, y_train, train_mat,
        X_val, y_val, val_mat,
        X_test, y_test, test_mat
    ) = preprocess_data()

    train_loader = get_loader(X_train, train_mat, BATCH_SIZE)
    val_loader   = get_loader(X_val, val_mat, BATCH_SIZE)
    test_loader  = get_loader(X_test, test_mat, BATCH_SIZE, shuffle=False)

    model = TransformerModel(40, 4, 20, 2, MAX_SEQUENCE_LEN,
                             INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    transformer_params = (
        list(model.pos_encoder.parameters()) +
        list(model.transformer.parameters()) +
        list(model.transformer2.parameters()) +
        list(model.fc.parameters())
    )
    transformer_optimizer = optim.Adam(transformer_params, lr=LEARNING_RATE)
    print('==> Pretraining transformer branch...')
    for epoch in range(NUM_EPOCHS):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, transformer_optimizer, criterion, mode='transformer'
        )
        if (epoch + 1) % 20 == 0:
            print(f'[Transformer Pretrain] Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}')

    mlp_optimizer = optim.Adam(model.network.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, mlp_optimizer, criterion, mode='mlp'
        )

    for param in transformer_params:
        param.requires_grad = False

    fusion_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    print("==> Start full model training ...")
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, fusion_optimizer, criterion, mode='fusion'
        )
        val_acc, _, _ = validate(model, val_loader, X_val, val_mat, y_val)
        print(f'[Fusion] Epoch {epoch+1}/{NUM_EPOCHS}, TrainLoss: {loss:.4f}, ValAcc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), "bestParameter.pt")
            test_acc, _, _ = Runtest(model, test_loader, X_test, test_mat, y_test)
            print(f'==> [Best Val] Test Accuracy: {test_acc:.2f}')

    model.load_state_dict(torch.load('bestParameter.pt'))
    test_acc, _, _ = Runtest(model, test_loader, X_test, test_mat, y_test)
    print(len(X_test))
    print(f'==> [Best Val] Test Accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    main()

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from preprocess import fasta2num, vec_to_onehot, seq2vec
from model import *
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5EncoderModel
import re
import torch
import os
import argparse

# Set global DEVICE, will be re-initialized in main() based on args
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Train Protein Sequence Classification Model")

    # ------------------- File Paths -------------------
    parser.add_argument('--train_fasta', type=str, default='Dataset/train.fasta',
                        help='Path to training set FASTA file')
    parser.add_argument('--train_label', type=str, default='Dataset/trainLabel.txt',
                        help='Path to training set label file')
    parser.add_argument('--data_dir', type=str, default='Dataset',
                        help='Dataset directory for saving feature cache files')
    parser.add_argument('--t5_model_path', type=str, 
                        help='Local path to ProstT5 pretrained model')
    parser.add_argument('--autoencoder_path', type=str, default='autoEnconder.txt',
                        help='Path to autoencoder feature file')
    parser.add_argument('--model_save_path', type=str, default='bestParameter.pt',
                        help='Path to save the best model weights')

    # ------------------- Model Hyperparameters -------------------
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--input_size', type=int, default=1024, help='Input feature dimension (T5 feature dimension)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                        help='List of hidden layer dimensions')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classification classes')
    parser.add_argument('--embed_dim', type=int, default=20, help='Embedding dimension')

    # ------------------- Training Hyperparameters -------------------
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use')

    return parser.parse_args()


def preprocess_data(args):
    X_train, X_train_seq = fasta2num(args.train_fasta)
    y_train = np.loadtxt(args.train_label) - 1

    train_mat_path = os.path.join(args.data_dir, 'trainProstMatrix.pt')
    val_mat_path = os.path.join(args.data_dir, 'valProstMatrix.pt')

    if not (os.path.exists(train_mat_path) and os.path.exists(val_mat_path)):

        print("==> Extracting features using ProstT5, please wait...")
        maxsequence = 5000
        # Split train and validation (80/20)
        X, X_val, Y, Y_val = train_test_split(X_train_seq, y_train, test_size=0.2, random_state=20, stratify=y_train)

        trainMat = torch.zeros((len(X), 1024))
        valMat = torch.zeros((len(X_val), 1024))

        tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path, do_lower_case=False)
        t5_model = T5EncoderModel.from_pretrained(args.t5_model_path).to(DEVICE)
        t5_model.float() if DEVICE.type == 'cpu' else t5_model.half()

        for i in range(len(X)):
            if len(X[i]) > maxsequence:
                u = X[i][:maxsequence]
                trainMat[i] = seq2vec(u, t5_model, tokenizer).cpu()
            else:
                trainMat[i] = seq2vec(X[i], t5_model, tokenizer).cpu()
        torch.save(trainMat, train_mat_path)

        for i in range(len(X_val)):
            if len(X_val[i]) > maxsequence:
                u = X_val[i][:maxsequence]
                valMat[i] = seq2vec(u, t5_model, tokenizer).cpu()
            else:
                valMat[i] = seq2vec(X_val[i], t5_model, tokenizer).cpu()
        torch.save(valMat, val_mat_path)

        del t5_model
        torch.cuda.empty_cache()
        print("==> ProstT5 feature extraction completed!")

    embed = np.loadtxt(args.autoencoder_path)
    embed = np.hstack((embed, np.zeros((args.embed_dim, 1))))

    train_embed = vec_to_onehot(X_train, len(y_train), args.max_seq_len, embed, args.embed_dim)
    one_hot = np.eye(args.embed_dim, args.embed_dim)
    train_onehot = vec_to_onehot(X_train, len(y_train), args.max_seq_len, one_hot, args.embed_dim)

    X_train_all = np.concatenate((train_embed, train_onehot), axis=1)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_all, y_train, test_size=0.2, random_state=20, stratify=y_train
    )

    X_train_tensor = torch.from_numpy(X_train_split.astype(np.float32))
    X_val_tensor = torch.from_numpy(X_val_split.astype(np.float32))

    y_train_tensor = torch.from_numpy(y_train_split).long()
    y_val_tensor = torch.from_numpy(y_val_split).long()

    train_mat = torch.load(train_mat_path)
    val_mat = torch.load(val_mat_path)

    return (
        X_train_tensor, y_train_tensor, train_mat,
        X_val_tensor, y_val_tensor, val_mat
    )


def get_loader(X_tensor, batch_size, shuffle=True):
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
            outputs = model(batch_X, y[indices].to(DEVICE))

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, X, y, labels):
    model.eval()
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


def evaluate(model, loader, X, y, labels):
    # 保留evaluate函数以备后续需要计算recall等指标
    model.eval()
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


def main(args):
    (
        X_train, y_train, train_mat,
        X_val, y_val, val_mat
    ) = preprocess_data(args)

    train_loader = get_loader(X_train, args.batch_size)
    val_loader = get_loader(X_val, args.batch_size, shuffle=False)

    model = TransformerModel(40, 4, args.embed_dim, 2, args.max_seq_len,
                             args.input_size, args.hidden_sizes, args.num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    transformer_params = (
            list(model.pos_encoder.parameters()) +
            list(model.transformer.parameters()) +
            list(model.transformer2.parameters()) +
            list(model.fc.parameters())
    )
    transformer_optimizer = optim.Adam(transformer_params, lr=args.lr)

    print('==> Pretraining transformer branch...')
    for epoch in range(args.epochs):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, transformer_optimizer, criterion, mode='transformer'
        )
        if (epoch + 1) % 20 == 0:
            print(f'[Transformer Pretrain] Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}')

    mlp_optimizer = optim.Adam(model.network.parameters(), lr=args.lr)
    print('==> Pretraining MLP branch...')
    for epoch in range(args.epochs):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, mlp_optimizer, criterion, mode='mlp'
        )

    for param in transformer_params:
        param.requires_grad = False

    fusion_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print("==> Start full model training ...")

    best_val_acc = 0
    for epoch in range(args.epochs):
        loss = train_single_epoch(
            model, train_loader, X_train, train_mat, y_train, fusion_optimizer, criterion, mode='fusion'
        )
        val_acc, _, _ = validate(model, val_loader, X_val, val_mat, y_val)
        print(f'[Fusion] Epoch {epoch + 1}/{args.epochs}, TrainLoss: {loss:.4f}, ValAcc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_save_path)
            print(f'==> [Best Val Saved] New best Validation Accuracy: {best_val_acc:.4f}')

    print(f"==> Training finished. Best model saved to {args.model_save_path} with ValAcc {best_val_acc:.4f}")


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)

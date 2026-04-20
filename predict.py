from torch.utils.data import DataLoader
from preprocess import fasta2num, vec_to_onehot, seq2vec
from model import *
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import re
import torch
import os
import argparse

# Set global DEVICE, will be re-initialized in main() based on args
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Protein Sequence Classification")

    # ------------------- File Paths -------------------
    parser.add_argument('--input_fasta', type=str, required=True, help='Path to input FASTA file for prediction')
    parser.add_argument('--output_path', type=str, default='output.txt', help='Path to save predicted labels')
    parser.add_argument('--data_dir', type=str, default='Dataset',
                        help='Directory to save/load temporary feature cache')
    parser.add_argument('--t5_model_path', type=str, 
                        help='Local path to ProstT5 pretrained model')
    parser.add_argument('--autoencoder_path', type=str, default='autoEnconder.txt',
                        help='Path to autoencoder feature file')
    parser.add_argument('--model_load_path', type=str, default='bestParameter.pt',
                        help='Path to the pre-trained model weights')

    # ------------------- Model Hyperparameters -------------------
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--input_size', type=int, default=1024, help='Input feature dimension (T5 feature dimension)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                        help='List of hidden layer dimensions')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classification classes')
    parser.add_argument('--embed_dim', type=int, default=20, help='Embedding dimension')

    # ------------------- Execution Options -------------------
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')

    return parser.parse_args()


def preprocess_inference_data(args):
    # Only process the single input file, no train/test splits needed
    X_input, X_input_seq = fasta2num(args.input_fasta)
    num_samples = len(X_input)

    # Define cache path based on the input filename to prevent overwriting
    base_name = os.path.basename(args.input_fasta).split('.')[0]
    input_mat_path = os.path.join(args.data_dir, f'{base_name}ProstMatrix.pt')

    # Ensure dataset dir exists
    os.makedirs(args.data_dir, exist_ok=True)

    if not os.path.exists(input_mat_path):
        print("==> Extracting features using ProstT5 for input sequences...")
        maxsequence = 5000
        inputMat = torch.zeros((num_samples, 1024))

        tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path, do_lower_case=False)
        t5_model = T5EncoderModel.from_pretrained(args.t5_model_path).to(DEVICE)
        t5_model.float() if DEVICE.type == 'cpu' else t5_model.half()

        for i in range(num_samples):
            if len(X_input_seq[i]) > maxsequence:
                u = X_input_seq[i][:maxsequence]
                inputMat[i] = seq2vec(u, t5_model, tokenizer).cpu()
            else:
                inputMat[i] = seq2vec(X_input_seq[i], t5_model, tokenizer).cpu()

        torch.save(inputMat, input_mat_path)

        del t5_model
        torch.cuda.empty_cache()
        print("==> ProstT5 feature extraction completed!")

    embed = np.loadtxt(args.autoencoder_path)
    embed = np.hstack((embed, np.zeros((args.embed_dim, 1))))

    input_embed = vec_to_onehot(X_input, num_samples, args.max_seq_len, embed, args.embed_dim)
    one_hot = np.eye(args.embed_dim, args.embed_dim)
    input_onehot = vec_to_onehot(X_input, num_samples, args.max_seq_len, one_hot, args.embed_dim)

    X_input_all = np.concatenate((input_embed, input_onehot), axis=1)
    X_input_tensor = torch.from_numpy(X_input_all.astype(np.float32))
    input_mat = torch.load(input_mat_path)

    return X_input_tensor, input_mat, num_samples


def get_loader(X_tensor, batch_size):
    dataset = torch.utils.data.TensorDataset(torch.arange(len(X_tensor)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_inference(model, loader, X, y):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for indices, in loader:
            batch_X = X[indices].to(DEVICE)
            batch_y = y[indices].to(DEVICE)
            outputs = model(batch_X, batch_y)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    return all_preds


def main(args):
    if not os.path.exists(args.model_load_path):
        raise FileNotFoundError(f"Cannot find model weights at {args.model_load_path}. Please train the model first.")

    X_input, input_mat, num_samples = preprocess_inference_data(args)
    inference_loader = get_loader(X_input, args.batch_size)

    # Initialize model
    model = TransformerModel(40, 4, args.embed_dim, 2, args.max_seq_len,
                             args.input_size, args.hidden_sizes, args.num_classes).to(DEVICE)

    # Load weights
    print(f"==> Loading model weights from {args.model_load_path}...")
    model.load_state_dict(torch.load(args.model_load_path, map_location=DEVICE))

    print("==> Running inference...")
    preds = run_inference(model, inference_loader, X_input, input_mat)

    # Map predictions back to 1-indexed format (assuming 1-based indexing for labels)
    formatted_preds = preds + 1
    np.savetxt(args.output_path, formatted_preds, fmt='%d')

    print(f"==> Processed {num_samples} sequences.")
    print(f"==> Predictions successfully saved to {args.output_path}")


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)

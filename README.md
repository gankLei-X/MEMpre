# MEMpre
MEMpre is a novel Transformer-based framework that leverages the AlphaFold-derived pLLM for accurately predicting membrane protein type. It comprises three primary stages:  extraction of 3D structural features; modeling of long-range sequence dependencies; membrane protein type prediction through the integration of structural and sequential features. Developer is Lei Guo from Fuzhou University of China.

# Overview of MEMpre
<div align=center>
<img src="https://github.com/user-attachments/assets/cd3765db-f39d-4e35-b2a8-cfb10026c203" width="600" height="500" /><br/>
</div>

__Overflow of the proposed MEMpre for membrane protein type prediction__. The primary protein sequence is ﬁrst processed by a pre‐trained ProtT5 followed by a fully connected block to generate the 3D structural feature, where the parameters of prostT5 model keep frozen during subsequent training. In parallel, the sequence is also encoded into a numerical representation using both one‐hot encoding and physicochemical property‐based encoding. This representation is then processed by a residual transformer block to capture long‐range sequence dependencies, generating the sequence feature. Finally, a linear layer integrates 3D structural feature and sequence feature to produce the ﬁnal prediction. 

# Requirement

    python >= 3.8

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia sentencepiece

    conda install numpy=1.26.4 matplotlib scipy=1.12.0 transformers=4.37.2 

    conda install scikit-learn=1.4.1

# Quickly start

## Predict For Your Test Data

cd to the MEMpre fold

If you want to predict membrane protein type, taking the test.fasta as an example, run:

    python predict.py --input_fasta Dataset/test.fasta --output_path output.txt --t5_model_path .../ProstT5 --model_load_path bestParameter.pt 

The output is the predicted membrane type with the shape of [N, 1], where the N is the number of inputting sequence. 

## Train For Your own Data

cd to the MEMpre fold

If you want to train for membrane protein type prediction, taking the train.fasta as an example, run:

    python train.py --train_fasta Dataset/train.fasta --train_label Dataset/trainLable.txt --t5_model_path .../ProstT5 --model_save_path bestParameter.pt

After training, the parameter of model with the best validation accuracy will be saved to the specified bestParameter.pt file.

# Acknowledge

Special thanks to the developers of ProstT5 for their groundbreaking work in protein language modeling.

# Contact

Please contact me if you have any help: gl5121405@gmail.com

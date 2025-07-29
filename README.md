# MEMpre
MEMpre is a novel Transformer-based framework that leverages the AlphaFold-derived pLLM for accurately predicting membrane protein type. It comprises three primary stages:  extraction of 3D structural features; modeling of long-range sequence dependencies; membrane protein type prediction through the integration of structural and sequential features. Developer is Lei Guo from Fuzhou University of China.

# Overview of MEMpre
<div align=center>
<img src="https://github.com/user-attachments/assets/cd3765db-f39d-4e35-b2a8-cfb10026c203" width="600" height="700" /><br/>
</div>

__Overflow of the proposed MEMpre for membrane protein type prediction__. The primary protein sequence is ﬁrst processed by a pre‐trained ProtT5 followed by a fully connected block to generate the 3D structural feature, where the parameters of prostT5 model keep frozen during subsequent training. In parallel, the sequence is also encoded into a numerical representation using both one‐hot encoding and physicochemical property‐based encoding. This representation is then processed by a residual transformer block to capture long‐range sequence dependencies, generating the sequence feature. Finally, a linear layer integrates 3D structural feature and sequence feature to produce the ﬁnal prediction. 

# Requirement

    python == 3.5, 3.6 or 3.7

    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    
    conda install opencv=4.5.3 numpy=1.8.0 matplotlib=2.2.2 scipy=1.12.0

    conda install scikit-learn=1.4.1

    
# Quickly start

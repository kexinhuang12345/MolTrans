# MolTrans: Molecular Interaction Transformer for Drug Target Interaction Prediction

Drug target interaction (DTI) prediction is a foundational task for in-silico drug discovery, which is costly and time-consuming due to the need of experimental search over large drug compound space. Recent years have witnessed promising progress for deep learning in DTI predictions. However, the following challenges are still open: (1) existing molecular representation learning approaches ignore the sub-structural nature of DTI, thus produce results that are less accurate and difficult to explain; (2) existing methods focus on limited labeled data while ignoring the value of massive unlabelled molecular data. We propose a Molecular Interaction Transformer (MolTrans) to address these limitations via: (1) knowledge inspired sub-structural pattern mining algorithm and interaction modeling module for more accurate and interpretable DTI prediction; (2) an augmented transformer encoder to better extract and capture the semantic relations among substructures extracted from massive unlabeled biomedical data. We evaluate MolTrans on real world data and show it improved DTI prediction performance compared to state-of-the-art baselines.

## Run
```
python train.py
```

More details will be released in a week or so.

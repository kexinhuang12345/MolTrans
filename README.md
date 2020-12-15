# MolTrans: Molecular Interaction Transformer for Drug Target Interaction Prediction

Drug target interaction (DTI) prediction is a foundational task for in-silico drug discovery, which is costly and time-consuming due to the need of experimental search over large drug compound space. Recent years have witnessed promising progress for deep learning in DTI predictions. However, the following challenges are still open: (1) existing molecular representation learning approaches ignore the sub-structural nature of DTI, thus produce results that are less accurate and difficult to explain; (2) existing methods focus on limited labeled data while ignoring the value of massive unlabelled molecular data. We propose a Molecular Interaction Transformer (MolTrans) to address these limitations via: (1) knowledge inspired sub-structural pattern mining algorithm and interaction modeling module for more accurate and interpretable DTI prediction; (2) an augmented transformer encoder to better extract and capture the semantic relations among substructures extracted from massive unlabeled biomedical data. We evaluate MolTrans on real world data and show it improved DTI prediction performance compared to state-of-the-art baselines.


## Datasets

In the dataset folder, we provide all three processed datasets used in MolTrans: BindingDB, DAVIS, and BIOSNAP. In BIOSNAP folder, there is full dataset for the main experiment, and also missing data experiment (70%, 80%, 90%, 95%) and unseen drug and unseen protein datasets.

## Run

We provide an example jupyter notebook in the repository. Although it runs for 100 epochs, we find 50 epochs is way enough and all the results in paper are run by 50 epochs. 

You can also directly run `python train.py --task ${task_name}` to run the experiments. `${task_name}` could either be `biosnap`,`bindingdb` , and `davis`. For the BindingDB and DAVIS, please refer this [Page](https://zitniklab.hms.harvard.edu/TDC/multi_pred_tasks/dti/) for more details.

Will add more codes and tests in the next couple of weeks. But this should be enough to try on MolTrans.
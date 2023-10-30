# IG-GCN
 A preliminary implementation of "Interpretable Graph Convolutional Network for Alzheimer's Disease Diagnosis using Multi-Modal Imaging Genetics". In our experimentation, SGCN and GO-based hierarchical graphs learned the sparse regional importance probability to find signature regions of interest (ROIs), and the connective importance probability to reveal the biomarkers.

## Usage
### Setup
The whole implementation is built upon [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)

**conda**

See the `environment.yml` for environment configuration. 
```bash
conda env create -f environment.yml
```
**PYG**

To install pyg library, [please refer to the document](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Dataset 
**ADNI**

We download this dataset from [here](https://adni.loni.usc.edu/data-samples/access-data/).
We treat multi-modal imaging scans as a brain graph, and SNPs info as a GO-based hierarchical graph.

### How to run classification?
The whole training framework is integrated in file `main.py`. To run
```
python main.py 
```
You can also specify the learning hyperparameters to run
```
python main.py --epochs 200 --lr 0.001 --search --cuda 0
```
`main.py`: tunning hyperparameters

`kernel/train_eval_sgcn_img_snps.py`: training framework

`kernel/sgcn_img_snp.py`: training model

`snps_graph.py`: build the GO graph

`kernel/go_model.py`: build GO hierarchical GAT model

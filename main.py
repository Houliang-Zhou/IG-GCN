import os.path as osp
import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.train_eval_sgcn_img_snps import cross_validation_with_val_set
from kernel.gcn import *
from kernel.graph_sage import *
from kernel.gin import *
from kernel.gat import *
from kernel.sgcn_img_snp import *
from sgcn_data import ADNIDataset, loadBrainImg_Snps_ADNI874
from util_gdc import preprocess_diffusion_imgs_snps


# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.filterwarnings("ignore")

# setting we need to change
parser = argparse.ArgumentParser(description='GNN for ADNI graphs')
parser.add_argument('--model', type=str, default='SGCN_GCN_IMGSNP')
parser.add_argument('--knn', type=int, default=5,
                    help='k for knn graph')
parser.add_argument('--isPPr', action='store_true', default=True,
                    help='is PPr in building DGC')
parser.add_argument('--isTopK', action='store_true', default=True,
                    help='is TopK in building DGC')
parser.add_argument('--top_k', type=int, default=3)

parser.add_argument('--disease_id', type=int, default=3,
                    help='disease_id for classification: 0, 1, 2')
parser.add_argument('--isCrossAtten', action='store_true', default=True,
                    help='isCrossAtten')
parser.add_argument('--isSoftSimilarity', action='store_true', default=True,
                    help='isSoftSimilarity in consistency constraint')
parser.add_argument('--isMultilModal4Similarity', action='store_true', default=False,
                    help='isMultilModal4Similarity in consistency constraint')

parser.add_argument('--rbf_gamma', type=float, default=0.01)
parser.add_argument('--clinical_score_index', type=int, default=-1)
parser.add_argument('--num_regr', type=int, default=3)
parser.add_argument('--model4eachregr', action='store_true', default=False,
                    help='FC layer for each clinical scores')
parser.add_argument('--isPermutTest', action='store_true', default=False,
                    help='is Permutation Test')
parser.add_argument('--isMultiFusion', action='store_true', default=False,
                    help='isMultiFusion')

parser.add_argument('--isuseFeat4Regr', action='store_true', default=True,
                    help='isuseFeat4Regr')
parser.add_argument('--isImageOnly', action='store_true', default=False,
                    help='isImageOnly')
parser.add_argument('--isSNPsOnly', action='store_true', default=False,
                    help='isSNPsOnly')
parser.add_argument('--Seed4PermutTest', type=int, default=1)
parser.add_argument('--lambda_disease', type=float, default=0.0)
parser.add_argument('--lambda_regr', type=float, default=1.0)
parser.add_argument('--lambda_prob', type=float, default=0.5)
parser.add_argument('--lambda_reco', type=float, default=0.0000015)
parser.add_argument('--lambda_simi', type=float, default=0.1)
parser.add_argument('--lambda_orth', type=float, default=0.01)

# General settings.
parser.add_argument('--data', type=str, default='ADNI')
parser.add_argument('--clean', action='store_true', default=False)

# GNN settings.
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hiddens', type=int, default=5)
parser.add_argument('--h', type=int, default=2)
parser.add_argument('--node_label', type=str, default='hop')
parser.add_argument('--use_rd', action='store_true', default=False)
parser.add_argument('--use_rp', type=int, default=None)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)

# Training settings.
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--fold', type=int, default=5)

# Other settings.
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--search', action='store_true', default=True,
                    help='search hyperparameters (layers, hiddens)')
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--cuda', type=int, default=0, help='which cuda to use')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.isPermutTest:
    args.res_dir = os.path.join(file_dir, 'results_permuttest/ADNI{}'.format(args.save_appendix))
else:
    args.res_dir = os.path.join(file_dir, 'results/ADNI{}'.format(args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data == 'all':
    datasets = [ 'DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
else:
    datasets = [args.data]

if args.search:
    if args.h is None:
        layers = [2, 3, 4, 5]
        hiddens = [32]
        hs = [None]
    else:
        # layers = [2]
        # hiddens = [16]
        # hs = [2]

        # current use
        if args.isMultiFusion:
            layers = [3, 2, 4, 2, 3]
            hiddens = [2, 3, 3, 5, 10]
            hs = [3, 2, 4, 2, 4]
        else:
            layers = [2, 3, 2, 3, 4]
            hiddens = [16, 16, 10, 10, 5]
            hs = [2, 3, 4, 4, 2]

        # layers = [2, 4, 2, 3, 2, 3, 4]
        # hiddens = [5, 5, 16, 16, 10, 10, 16]
        # hs = [2, 3, 2, 3, 4, 4, 2]
else:
    layers = [args.layers]
    hiddens = [args.hiddens]
    hs = [args.h]

if args.model == 'all':
    #nets = [GCN, GraphSAGE, GIN, GAT]
    nets = [NestedGCN, NestedGraphSAGE, NestedGIN, NestedGAT]
else:
    nets = [eval(args.model)]

def logger(info):
    f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
    print(info, file=f)

device = torch.device(
    'cuda:%d'%(args.cuda)  if torch.cuda.is_available() and not args.cpu else 'cpu'
)
print(device)

cross_val_method = cross_validation_with_val_set

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)
    log = '-----\n{} - {}'.format(dataset_name, Net.__name__)
    print(log)
    logger(log)
    if args.h is not None:
        combinations = zip(layers, hiddens, hs)
    else:
        combinations = product(layers, hiddens, hs)
    for num_layers, hidden, h in combinations:
        if dataset_name == 'DD' and Net.__name__ == 'NestedGAT' and h >= 5:
            print('NestedGAT on DD will OOM for h >= 5. Skipped.')
            continue
        log = "Using {} layers, {} hidden units, h = {}".format(num_layers, hidden, h)
        print(log)
        logger(log)
        result_file_name = "result_sgcn_img_snp_layers{}_hidden{}_h{}".format(num_layers, hidden, h)
        result_path = os.path.join(args.res_dir, '%s.npy'%(result_file_name))
        max_nodes_per_hop = None
        pre_transform = lambda x: preprocess_diffusion_imgs_snps(x, isPPr=args.isPPr, isTopK=args.isTopK, top_k=args.top_k)
        data_path = './data/brain_image/knn/%d/'%(args.knn)
        json_path = './data/snps/analysis.json'
        if args.disease_id <3:
            num_classes=2
        else:
            num_classes=3
        adni_dataset, scaler4score = loadBrainImg_Snps_ADNI874(disease_id=args.disease_id, path = './data/snps/data/preprocessing/', k_inknn = args.knn, num_cluster=2,
                                                               clinical_scores=args.clinical_score_index, isPermutTest=args.isPermutTest, Seed4PermutTest=args.Seed4PermutTest,
                                                               isMultiFusion=args.isMultiFusion, isMultilModal4Similarity=args.isMultilModal4Similarity)
        dataset = ADNIDataset('./', 'SGCN', adni_dataset, pre_transform=pre_transform)
        lambda_loss = [args.lambda_disease, args.lambda_regr, args.lambda_prob, args.lambda_reco, args.lambda_simi, args.lambda_orth]
        loss, acc, std = cross_val_method(
            dataset,
            args=args,
            num_layers=num_layers,
            hidden=hidden,
            folds=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            device=device,
            logger=logger,
            result_path=result_path,
            result_file_name=result_file_name,
            json_path=json_path,
            disease_id=args.disease_id,
            isCrossAtten=args.isCrossAtten,
            isSoftSimilarity=args.isSoftSimilarity,
            rbf_gamma=args.rbf_gamma,
            num_classes=num_classes,
            lambda_loss=lambda_loss,
            num_regr=args.num_regr,
            model4eachregr = args.model4eachregr,
            scaler4score=scaler4score)
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden, h)

    desc = '{:.3f} ± {:.3f}'.format(
        best_result[1], best_result[2]
    )
    log = 'Best result - {}, with {} layers and {} hidden units and h = {}'.format(
        desc, best_hyper[0], best_hyper[1], best_hyper[2]
    )
    print(log)
    logger(log)
    results += ['{} - {}: {}'.format(dataset_name, Net.__name__, desc)]

log = '-----\n{}'.format('\n'.join(results))
print(cmd_input[:-1])
print(log)
logger(log)

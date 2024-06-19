import argparse
import math
import time
import dgl
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import numpy as np
import numpy_indexed as npi
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
import dgl.data
from torch.utils.data import DataLoader
from dgl.sampling import random_walk
# from torch_cluster import random_walk
import os.path as osp
from gen_model import gen_model
from logger import Logger
from loss import calculate_loss
from utils import *
import sys
import os
from data import *
from train_test import *
import pickle


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'agdn', 'memagdn'])
    parser.add_argument('--clip-grad-norm', type=float, default=1)
    parser.add_argument('--use-valedges-as-input', action='store_true',
                        help='This option can only be used for ogbl-collab')
    parser.add_argument('--no-node-feat', action='store_true')
    parser.add_argument('--use-emb', action='store_true')
    parser.add_argument('--use-edge-feat', action='store_true')
    parser.add_argument('--train-on-subgraph', action='store_true')
    parser.add_argument('--year', type=int, default=0)

    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--transition-matrix', type=str, default='gat')
    parser.add_argument('--hop-norm', action='store_true')
    parser.add_argument('--weight-style', type=str, default='HA',
                        choices=['HC', 'HA', 'HA+HC', 'HA1', 'sum', 'max_pool', 'mean_pool', 'lstm'])
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--no-share-weights', action='store_true')
    parser.add_argument('--pre-act', action='store_true')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-hidden', type=int, default=512)
    parser.add_argument('--out-hidden', type=int, default=512)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input-drop', type=float, default=0.)
    parser.add_argument('--edge-drop', type=float, default=0.)
    parser.add_argument('--attn-drop', type=float, default=0.)
    parser.add_argument('--diffusion-drop', type=float, default=0.)
    parser.add_argument('--bn', action='store_true')
    # parser.add_argument('--bn', default=True)
    parser.add_argument('--output-bn', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--no-dst-attn', action='store_true')

    parser.add_argument('--advanced-optimizer', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--negative-sampler', type=str, default='strict_global',
                        choices=['global', 'strict_global', 'persource'])
    parser.add_argument('--n-neg', type=int, default=1)
    parser.add_argument('--eval-metric', type=str, default='hits')
    parser.add_argument('--loss-func', type=str, default='CE')
    parser.add_argument('--predictor', type=str, default='MLP')

    parser.add_argument('--random_walk_augment', action='store_true')
    parser.add_argument('--walk_start_type', type=str, default='edge')
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--adjust-lr', action='store_true')
    # parser.add_argument('--adjust-lr', default=True)

    parser.add_argument('--use-heuristic', action='store_true')
    parser.add_argument('--extra-training-edges', action='store_true')

    args = parser.parse_args()

    save_path = f'./result/log/{args.dataset}/standard/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sys.stdout = Log(save_path, filename=f'p-{args.model}_lo-{args.loss_func}|{time.time()}.log', stream=sys.stdout)
    print(args)

    graph, split_edge = get_data(args.dataset)

    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    # device = 'cpu'
    device = torch.device(device)

    edge_weight = graph.edata['weight'].view(-1).cpu().numpy() \
        if 'weight' in graph.edata.keys() else torch.ones(graph.number_of_edges())
    A = ssp.csr_matrix((edge_weight, (graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy())),
                       shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    # multiplier = 1 / np.log(A.sum(axis=0))
    # multiplier[np.isinf(multiplier)] = 0
    # A_ = A.multiply(multiplier).tocsr()
    adjs = precompute_adjs(A)

    # evaluator = Evaluator(name=args.dataset)
    evaluators = {
        "ogbl-collab": Evaluator(name='ogbl-collab'),
        "reddit": Evaluator(name='ogbl-collab'),
        "ogbl-ddi": Evaluator(name='ogbl-ddi'),
        "email": Evaluator(name='ogbl-ddi'),
        "twitch": Evaluator(name='ogbl-ddi'),
        "fb": Evaluator(name='ogbl-collab'),
        "cora": Evaluator(name='ogbl-ddi'),
        "citeseer": Evaluator(name='ogbl-ddi'),
        "pubmed": Evaluator(name='ogbl-ddi')
    }
    evaluator = evaluators[args.dataset]
    hits = {
        "ogbl-collab": [10, 50, 100],
        "reddit": [10, 50, 100],
        "ogbl-ddi": [10, 20, 30],
        "email": [10, 20, 30],
        "twitch": [10, 50, 100],
        "fb": [10, 20, 30],
        'citeseer': [10, 20, 30],
        'cora': [10, 20, 30],
        'pubmed': [10, 20, 30]
    }

    hitslist = hits[args.dataset]

    loggers_CN = {
        f'Hits@{hitslist[0]}': Logger(args.runs, args),
        f'Hits@{hitslist[1]}': Logger(args.runs, args),
        f'Hits@{hitslist[2]}': Logger(args.runs, args),
        'accuracy': Logger(args.runs, args),
        'recall': Logger(args.runs, args),
        'precision': Logger(args.runs, args),
        'F1-score': Logger(args.runs, args),
        'AUC': Logger(args.runs, args)
    }
    loggers_AA = {
        f'Hits@{hitslist[0]}': Logger(args.runs, args),
        f'Hits@{hitslist[1]}': Logger(args.runs, args),
        f'Hits@{hitslist[2]}': Logger(args.runs, args),
        'accuracy': Logger(args.runs, args),
        'recall': Logger(args.runs, args),
        'precision': Logger(args.runs, args),
        'F1-score': Logger(args.runs, args),
        'AUC': Logger(args.runs, args)
    }
    target_metrics = {'ogbl-collab': 'Hits@50',
                      'ogbl-ddi': 'Hits@20',
                      'cora': 'Hits@20',
                      'citeseer': 'Hits@20',
                      'pubmed': 'Hits@20',
                      'email': 'Hits@20',
                      'fb': 'Hits@20',
                      'reddit': 'Hits@50',
                      'twitch': 'Hits@50'}



    pos_train_pred = RA_AA_CN(adjs, split_edge['train']['edge'].t())[:,1]
    pos_valid_pred = RA_AA_CN(adjs, split_edge['valid']['edge'].t())[:,1]
    neg_valid_pred = RA_AA_CN(adjs, split_edge['valid']['edge_neg'].t())[:,1]
    pos_test_pred = RA_AA_CN(adjs, split_edge['test']['edge'].t())[:,1]
    neg_test_pred = RA_AA_CN(adjs, split_edge['test']['edge_neg'].t())[:,1]

    results_AA = evaluate_hits(evaluator, hitslist,pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    train_pred = (pos_train_pred > 0).int().cpu().numpy()
    train_label = torch.ones(pos_train_pred.size(0)).int().cpu().numpy()
    valid_pred = (torch.cat([pos_valid_pred, neg_valid_pred]) > 0).int().cpu().numpy()
    valid_label = torch.cat(
        [torch.ones(pos_valid_pred.size(0)), torch.zeros(neg_valid_pred.size(0))]).int().cpu().numpy()
    test_pred = (torch.cat([pos_test_pred, neg_test_pred]) > 0).int().cpu().numpy()
    test_label = torch.cat([torch.ones(pos_test_pred.size(0)), torch.zeros(neg_test_pred.size(0))]).int().cpu().numpy()

    train_acc = accuracy_score(train_label, train_pred)
    valid_acc = accuracy_score(valid_label, valid_pred)
    test_acc = accuracy_score(test_label, test_pred)

    train_recall = recall_score(train_label, train_pred)
    valid_recall = recall_score(valid_label, valid_pred)
    test_recall = recall_score(test_label, test_pred)

    train_precision = precision_score(train_label, train_pred)
    valid_precision = precision_score(valid_label, valid_pred)
    test_precision = precision_score(test_label, test_pred)

    train_f1 = f1_score(train_label, train_pred)
    valid_f1 = f1_score(valid_label, valid_pred)
    test_f1 = f1_score(test_label, test_pred)

    valid_auc = roc_auc_score(valid_label, valid_pred)
    test_auc = roc_auc_score(test_label, test_pred)

    results_AA['accuracy'] = (train_acc, valid_acc, test_acc)
    results_AA['recall'] = (train_recall, valid_recall, test_recall)
    results_AA['precision'] = (train_precision, valid_precision, test_precision)
    results_AA['F1-score'] = (train_f1, valid_f1, test_f1)
    results_AA['AUC'] = (0, valid_auc, test_auc)


    for key, result in results_AA.items():
        loggers_CN[key].add_result(0, result)

    for key in loggers_CN.keys():
        print(key)
        loggers_CN[key].print_statistics(0)
    print('-------------------------------------------------------------------------------')

    pos_train_pred = RA_AA_CN(adjs, split_edge['train']['edge'].t())[:, 2]
    pos_valid_pred = RA_AA_CN(adjs, split_edge['valid']['edge'].t())[:, 2]
    neg_valid_pred = RA_AA_CN(adjs, split_edge['valid']['edge_neg'].t())[:, 2]
    pos_test_pred = RA_AA_CN(adjs, split_edge['test']['edge'].t())[:, 2]
    neg_test_pred = RA_AA_CN(adjs, split_edge['test']['edge_neg'].t())[:, 2]

    results_CN = evaluate_hits(evaluator, hitslist, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred,
                               neg_test_pred)

    train_pred = (pos_train_pred > 0).int().cpu().numpy()
    train_label = torch.ones(pos_train_pred.size(0)).int().cpu().numpy()
    valid_pred = (torch.cat([pos_valid_pred, neg_valid_pred]) > 0).int().cpu().numpy()
    valid_label = torch.cat(
        [torch.ones(pos_valid_pred.size(0)), torch.zeros(neg_valid_pred.size(0))]).int().cpu().numpy()
    test_pred = (torch.cat([pos_test_pred, neg_test_pred]) > 0).int().cpu().numpy()
    test_label = torch.cat([torch.ones(pos_test_pred.size(0)), torch.zeros(neg_test_pred.size(0))]).int().cpu().numpy()

    train_acc = accuracy_score(train_label, train_pred)
    valid_acc = accuracy_score(valid_label, valid_pred)
    test_acc = accuracy_score(test_label, test_pred)

    train_recall = recall_score(train_label, train_pred)
    valid_recall = recall_score(valid_label, valid_pred)
    test_recall = recall_score(test_label, test_pred)

    train_precision = precision_score(train_label, train_pred)
    valid_precision = precision_score(valid_label, valid_pred)
    test_precision = precision_score(test_label, test_pred)

    train_f1 = f1_score(train_label, train_pred)
    valid_f1 = f1_score(valid_label, valid_pred)
    test_f1 = f1_score(test_label, test_pred)

    valid_auc = roc_auc_score(valid_label, valid_pred)
    test_auc = roc_auc_score(test_label, test_pred)

    results_CN['accuracy'] = (train_acc, valid_acc, test_acc)
    results_CN['recall'] = (train_recall, valid_recall, test_recall)
    results_CN['precision'] = (train_precision, valid_precision, test_precision)
    results_CN['F1-score'] = (train_f1, valid_f1, test_f1)
    results_CN['AUC'] = (0, valid_auc, test_auc)

    for key, result in results_CN.items():
        loggers_AA[key].add_result(0, result)

    for key in loggers_CN.keys():
        print(key)
        loggers_AA[key].print_statistics(0)






if __name__ == "__main__":
    main()

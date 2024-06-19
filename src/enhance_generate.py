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
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'gat', 'sage', 'agdn', 'memagdn'])
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
    parser.add_argument('--weight-style', type=str, default='HA', choices=['HC', 'HA', 'HA+HC', 'HA1', 'sum', 'max_pool', 'mean_pool', 'lstm'])
    parser.add_argument('--no-pos-emb', action='store_true')
    parser.add_argument('--no-share-weights', action='store_true')
    parser.add_argument('--pre-act', action='store_true')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-hidden', type=int, default=256)
    parser.add_argument('--out-hidden', type=int, default=256)
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
    parser.add_argument('--batch-size', type=int, default=16*1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--negative-sampler', type=str, default='strict_global', choices=['global', 'strict_global', 'persource'])
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
    sys.stdout = Log(save_path,filename=f'p-{args.model}_lo-{args.loss_func}|{time.time()}.log', stream=sys.stdout)
    print(args)

    graph, split_edge = get_data(args.dataset)

    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    # device = 'cpu'
    device = torch.device(device)

    if args.model in ['gcn']:
        graph = graph.remove_self_loop().add_self_loop()
    print(graph)


    has_edge_attr = len(graph.edata.keys()) > 0



    if 'weight' in graph.edata:
        graph.edata['weight'] = graph.edata['weight'].float()

    if 'year' in split_edge['train'].keys() and args.year > 0:
        mask = split_edge['train']['year'] >= args.year
        split_edge['train']['edge'] = split_edge['train']['edge'][mask]
        split_edge['train']['year'] = split_edge['train']['year'][mask]
        split_edge['train']['weight'] = split_edge['train']['weight'][mask]
        graph.remove_edges((graph.edata['year']<args.year).nonzero(as_tuple=False).contiguous().view(-1))
        graph = to_undirected(graph)

    torch.manual_seed(12345)
    if args.dataset == 'ogbl-citation2':
        idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
        split_edge['eval_train'] = {
            'source_node': split_edge['train']['source_node'][idx],
            'target_node': split_edge['train']['target_node'][idx],
            'target_node_neg': split_edge['valid']['target_node_neg'],
        }
    else:
        idx = torch.randperm(split_edge['train']['edge'].size(0))
        idx = idx[:split_edge['valid']['edge'].size(0)]
        split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        # val_edge_index = split_edge['valid']['edge'].t()
        # full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        # data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        # data.full_adj_t = data.full_adj_t.to_symmetric()

        full_graph = graph.clone()
        # split_edge['valid']['year'] = split_edge['valid']['year'] - 1900

        full_graph.remove_edges(torch.arange(full_graph.number_of_edges()))
        full_graph.add_edges(split_edge['train']['edge'][:, 0], split_edge['train']['edge'][:, 1], 
                            {'weight': split_edge['train']['weight'].unsqueeze(1).float()})
        full_graph.add_edges(split_edge['valid']['edge'][:, 0], split_edge['valid']['edge'][:, 1],
                            {'weight': split_edge['valid']['weight'].unsqueeze(1).float()})
        full_graph = to_undirected(full_graph)

        # In official OGB example, use_valedges_as_input options only utilizes validation edges in inference.
        # However, as described in OGB rules, validation edges can also participate training after all hyper-parameters
        # are fixed. The suitable pipeline is: 1. Tune hyperparameters using validation set without touching it during training
        # and inference (except as targets). 2. Re-train final model using tuned hyperparemters using validation edges as input.
        split_edge['train']['edge'] = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0)
        split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=0)
        # mask = full_graph.edges()[0] < full_graph.edges()[1]
        # split_edge['train']['edge'] = torch.stack([full_graph.edges()[0][mask], full_graph.edges()[1][mask]], dim=1)
        # split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=0)
        # split_edge['train']['year'] = torch.cat([split_edge['train']['year'], split_edge['valid']['year']], dim=0)
    else:
        full_graph = graph
    
    if args.train_on_subgraph and 'year' in split_edge['train'].keys():
        mask = (graph.edata['year'] >= 2010).view(-1)
        
        filtered_nodes = torch.cat([graph.edges()[0][mask], graph.edges()[1][mask]], dim=0).unique()
        graph.remove_edges((~mask).nonzero(as_tuple=False).view(-1))
        
        split_edge['train'] = filter_edge(split_edge['train'], filtered_nodes)
        split_edge['valid'] = filter_edge(split_edge['valid'], filtered_nodes)
        # split_edge['test'] = filter_edge(split_edge['test'], filtered_nodes)
  

    edge_weight = graph.edata['weight'].view(-1).cpu().numpy() \
        if 'weight' in graph.edata.keys() else torch.ones(graph.number_of_edges())
    A = ssp.csr_matrix((edge_weight, (graph.edges()[0].cpu().numpy(), graph.edges()[1].cpu().numpy())), 
                       shape=(graph.number_of_nodes(), graph.number_of_nodes()))
    # multiplier = 1 / np.log(A.sum(axis=0))
    # multiplier[np.isinf(multiplier)] = 0
    # A_ = A.multiply(multiplier).tocsr()
    adjs = precompute_adjs(A)
    if args.use_heuristic:
        # We implement preliminary version of Edge Proposal Set: https://arxiv.org/abs/2106.15810
        method_dict = {'RA':0, 'AA':1, 'CN':2}
        target_idx = method_dict[args.heuristic_method]
        target_size = args.n_extra_edges
        if not osp.exists(f'../extra_edges/{args.dataset}.pt'):
            A2 = A @ A
            A2[A > 0] = 0
            row, col = A2.nonzero()
            row, col = row[~(row==col)], col[~(row==col)]
            extra_edges = torch.from_numpy(np.stack([row, col], axis=1)).long()
            print(f'Initial extra edge number: {len(extra_edges)}')
            # extra_edges = extra_edges[~(npi.in_(extra_edges, split_edge['train']['edge']) \
            #                                     | npi.in_(extra_edges[:, [1,0]], split_edge['train']['edge']))]
            if args.use_valedges_as_input:
                extra_edges = extra_edges[~(npi.in_(extra_edges, split_edge['valid']['edge']) \
                                                    | npi.in_(extra_edges[:, [1,0]], split_edge['valid']['edge']))]
            print(f'Additional edge number after filtering existing edges: {len(extra_edges)}')

            extra_scores = RA_AA_CN(adjs, extra_edges.t())
            
            torch.save([extra_edges, extra_scores], f'../extra_edges/{args.dataset}.pt')
        else:
            extra_edges, extra_scores = torch.load(f'../extra_edges/{args.dataset}.pt')
        _, idx = torch.sort(extra_scores[:, target_idx], descending=True)
        extra_edges = extra_edges[idx]
        extra_edges = extra_edges[:target_size]
        extra_scores = extra_scores[idx]
        extra_scores = extra_scores[:target_size, [target_idx]]
        print(extra_scores.max(), extra_scores.min())
        extra_scores = extra_scores.clamp(max=100)
        extra_scores = extra_scores / extra_scores.max()
        print(extra_scores.max(), extra_scores.min())
        # extra_scores = extra_scores / extra_scores.max()
        # extra_scores.clamp_(min=0.01)
        # graph.add_edges(extra_edges[:,0], extra_edges[:,1], {'weight': extra_scores})
        # full_graph.add_edges(extra_edges[:,0], extra_edges[:,1], {'weight': extra_scores})
        # graph = to_undirected(graph)
        # full_graph = to_undirected(full_graph)
        if args.extra_training_edges:
            split_edge['train']['edge'] = torch.cat([split_edge['train']['edge'], extra_edges], dim=0)
            if 'weight' in split_edge['train'].keys():
                split_edge['train']['weight'] = torch.ones_like(split_edge['train']['weight'])
                split_edge['train']['weight'] = torch.cat([split_edge['train']['weight'], extra_scores.view(-1,)], dim=0)


    full_edge_weight = full_graph.edata['weight'].view(-1).cpu().numpy() \
        if 'weight' in full_graph.edata.keys() else torch.ones(full_graph.number_of_edges())
    full_A = ssp.csr_matrix((full_edge_weight, (full_graph.edges()[0].cpu().numpy(), full_graph.edges()[1].cpu().numpy())), 
                       shape=(full_graph.number_of_nodes(), full_graph.number_of_nodes()))
    full_adjs = precompute_adjs(full_A)

    graph = graph.to(device)
    full_graph = full_graph.to(device)

    has_node_attr = 'feat' in graph.ndata
    if has_node_attr and (not args.use_emb) and (not args.no_node_feat):
        emb = None
        feat = graph.ndata['feat'].float()
    else:
        # Use learnable embedding if node attributes are not available
        n_heads = args.n_heads if args.model in ['gat', 'agdn'] else 1
        emb = torch.nn.Embedding(graph.number_of_nodes(), args.n_hidden).to(device)
        if not has_node_attr or args.no_node_feat:
            feat = emb.weight
        else:
            feat = torch.cat([graph.ndata['feat'].float(), emb.weight], dim=-1)

    # degs = graph.in_degrees()
    # inv_degs = 1. / degs
    # inv_degs[torch.isinf(inv_degs)] = 0
    # inv_log_degs = 1. / torch.log(degs)
    # inv_log_degs[torch.isinf(inv_log_degs)] = 0
    # deg_feat = torch.cat([degs.unsqueeze(-1), inv_degs.unsqueeze(-1), inv_log_degs.unsqueeze(-1)], dim=-1)
    # # deg_feat = (deg_feat - deg_feat.min(0)[0]) / (deg_feat.max(0)[0] - deg_feat.min(0)[0])
    # feat = feat * inv_log_degs.unsqueeze(-1)

    in_feats = feat.shape[1]

    if has_edge_attr and args.use_edge_feat:
        edge_feat = graph.edata['weight'].float()
        full_edge_feat = full_graph.edata['weight'].float()
        in_edge_feats = graph.edata['weight'].shape[1]
    else:
        edge_feat = None
        full_edge_feat = None
        in_edge_feats = 0

    # evaluator = Evaluator(name=args.dataset)
    evaluators = {
        "ogbl-collab": Evaluator(name='ogbl-collab'),
        "reddit": Evaluator(name='ogbl-collab'),
        "ogbl-ddi": Evaluator(name='ogbl-ddi'),
        "email": Evaluator(name='ogbl-ddi'),
        "twitch": Evaluator(name='ogbl-ddi'),
        "fb": Evaluator(name='ogbl-collab'),
        "cora":Evaluator(name='ogbl-ddi'),
        "citeseer":Evaluator(name='ogbl-ddi'),
        "pubmed":Evaluator(name='ogbl-ddi')
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
    if args.eval_metric == 'hits':
        loggers = {
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

    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

        target_metrics = {'ogbl-citation2': 'MRR'}

    if args.random_walk_augment:
        if args.walk_start_type == 'edge':
            rw_start = torch.reshape(split_edge['train']['edge'], (-1,)).to(device)
        else:
            rw_start = torch.arange(0, graph.number_of_nodes(), dtype=torch.long).to(device)

    loss_dict = {}
    for run in range(args.runs):
        seed(args.seed + run)
        
        model, predictor = gen_model(args, in_feats, in_edge_feats, device)
        print(model)
        parameters = list(model.parameters()) + list(predictor.parameters())
        if emb is not None:
            parameters = parameters + list(emb.parameters())
            torch.nn.init.xavier_uniform_(emb.weight)
            num_param = count_parameters(model) + count_parameters(predictor) + count_parameters(emb)
        else:
            num_param = count_parameters(model) + count_parameters(predictor)
        print(f'Number of parameters: {num_param}')
        
        if args.advanced_optimizer:
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50 // args.eval_steps, verbose=True)
        else:
            optimizer = torch.optim.Adam(
                parameters,
                lr=args.lr)

        best_val = {}
        best_test = {}
        save_loss = []
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            if args.random_walk_augment:
                # Random walk augmentation from PLNLP repository
                # We add a restart prob to reduce sampled pairs
                # walk = random_walk(full_graph.edges()[0], full_graph.edges()[1], rw_start, walk_length=args.walk_length)
                walk, _ = random_walk(full_graph, rw_start, length=args.walk_length)
                pairs = []
                weights = []
                for j in range(args.walk_length):
                    pairs.append(walk[:, [0, j + 1]])
                    weights.append(torch.ones((walk.size(0),), dtype=torch.float) / (j + 1))
                    # weights.append(torch.ones((walk.size(0),), dtype=torch.float) *  math.exp(-(j + 1.) / 2))
                pairs = torch.cat(pairs, dim=0)
                weights = torch.cat(weights, dim=0)
                # remove self-loop edges
                mask = ((pairs[:, 0] - pairs[:, 1]) != 0) * (pairs[:, 1] != -1)
                
                split_edge['train']['edge'] = torch.masked_select(pairs, mask.view(-1, 1)).view(-1, 2)
                split_edge['train']['weight'] = torch.masked_select(weights, mask)
                # edges_and_weights = torch.cat([split_edge['train']['edge'], split_edge['train']['weight'].view(-1,1).to(device)], dim=1)
                # edges_and_weights = torch.unique(edges_and_weights, dim=0)
                # split_edge['train']['edge'] = edges_and_weights[:, :2].long().to(device)
                # split_edge['train']['weight'] = edges_and_weights[:, 2].cpu()
            
            loss = train(model, predictor, feat, full_edge_feat, full_graph, split_edge, optimizer,
                         args.batch_size, args)
            save_loss.append(loss)
            t2 = time.time()
            if epoch % args.eval_steps == 0:
                
                results = test(model, predictor, feat, edge_feat, graph, full_edge_feat, full_graph, split_edge, evaluator,
                               args.batch_size, args,hitslist,epoch)
                t3 = time.time()
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    if key not in best_val:
                        best_val[key] = result[1]
                        best_test[key] = result[2]
                    elif result[1] > best_val[key]:
                        best_val[key] = result[1]
                        best_test[key] = result[2]
                        if key == target_metrics[args.dataset]:
                            save_path = f'./result/new_model/standard/{args.dataset}/'
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            save_model(os.path.join(save_path,
                                                    f'p-{args.model}_lo-{args.loss_func}_r-{run + 1}.pt'),
                                       epoch, optimizer, model,
                                       predictor,
                                       emb)

                if args.advanced_optimizer:
                    lr_scheduler.step(results[target_metrics[args.dataset]][1])

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Train/Val/Test: {100 * train_hits:.2f}/{100 * valid_hits:.2f}/{100 * test_hits:.2f}%, '
                              f'Best Val/Test: {100 * best_val[key]:.2f}/{100 * best_test[key]:.2f}%')
                    print(f'---Loss: {loss:.4f}---Train time: {(t2-t1):.4f}---Test time: {(t3-t2):.4f}---')
            if args.adjust_lr:
                adjust_lr(optimizer, epoch / args.epochs, args.lr)
        loss_dict[f'run{run + 1}'] = {'loss': save_loss}
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    # print(loss_dict)
    save_path = f'./result/loss/standard/{args.dataset}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(loss_dict, file=open(os.path.join(save_path,
                           f'p-{args.model}_lo-{args.loss_func}_r-{run + 1}.pkl'), 'wb+'))
    # with open(os.path.join(save_path,
    #                        f'p-{args.model}_lo-{args.loss_func}_r-{run + 1}.pkl')) as tf:
    #     pickle.dump(loss_dict, tf)
    best_run = []
    for key in loggers.keys():
        print(key)
        best_run.append(loggers[key].print_statistics())

    epoch_save = load_model(os.path.join(f'./result/new_model/standard/{args.dataset}/', f'p-{args.model}_lo-{args.loss_func}_r-{best_run[0]}.pt'), optimizer, model, predictor, emb)

    results = test(model, predictor, feat, edge_feat, graph, full_edge_feat, full_graph, split_edge,
                   evaluator,args.batch_size, args,hitslist,epoch)
    for key, result in results.items():
        train_hits, valid_hits, test_hits = result
        print(key)
        print(
              f'Epoch: {epoch_save:02d}, '
              f'Train/Val/Test: {100 * train_hits:.2f}/{100 * valid_hits:.2f}/{100 * test_hits:.2f}%, ')
              # f'Best Val/Test: {100 * best_val[key]:.2f}/{100 * best_test[key]:.2f}%')
    # print(f'---Loss: {loss:.4f}---Train time: {(t2 - t1):.4f}---Test time: {(t3 - t2):.4f}---')

    with torch.no_grad():
        logits_preds = []
        candidate_index = CN(graph)
        # candidate_index = torch.load('linkwaldo_cnadidate_1000000_uv.pt')
        for perm in DataLoader(range(candidate_index.size(0)), batch_size=args.batch_size,
                               shuffle=False):
            h = model(graph, feat, edge_feat)

            edge = candidate_index[perm]

            logits = predictor(h[edge[:, 0]], h[edge[:, 1]])
            logits_preds.append(logits.cpu().squeeze())
        logits_pred = torch.cat(logits_preds, dim=0)
        score_index = logits_pred.sort(0, True)[1]
        candidate_score = logits_pred.sort(0, True)[0].unsqueeze(1)
        candidate_index = candidate_index[score_index]
        save_path = f'./result/candidate/{args.dataset}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({'candidate_index': candidate_index,
                    'candidate_score': candidate_score}, os.path.join(save_path,f'p-{args.model}_lo-{args.loss_func}_cl.pt'),_use_new_zipfile_serialization=False)
        print('candidate link save successfully')


if __name__ == "__main__":
    main()

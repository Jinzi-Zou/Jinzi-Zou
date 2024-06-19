import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy_indexed as npi
import dgl
from tqdm import tqdm
import torch_sparse
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import math
import os
import sys
import scipy.sparse as sp

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def adjust_lr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

def to_undirected(graph):
    print(f'Previous edge number: {graph.number_of_edges()}')
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
    keys = list(graph.edata.keys())
    for k in keys:
        if k != 'weight':
            graph.edata.pop(k)
        else:
            graph.edata[k] = graph.edata[k].float()
    graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True, aggregator='sum')
    print(f'After adding reversed edges: {graph.number_of_edges()}')
    return graph

def filter_edge(split, nodes):
    mask = npi.in_(split['edge'][:,0], nodes) & npi.in_(split['edge'][:,1], nodes)
    print(len(mask), mask.sum())
    split['edge'] = split['edge'][mask]
    split['year'] = split['year'][mask]
    split['weight'] = split['weight'][mask]
    if 'edge_neg' in split.keys():
        mask = npi.in_(split['edge_neg'][:,0], nodes) & npi.in_(split['edge_neg'][:,1], nodes)
        split['edge_neg'] = split['edge_neg'][mask]
    return split

def evaluate_hits(evaluator, hitslist,pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in hitslist:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results
        

def evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}

    train_mrr = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (train_mrr, valid_mrr, test_mrr)
    
    return results

def evaluate_rocauc(evaluator, pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred):
    results = {}
    train_rocauc = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_train_pred,
    })[f'rocauc']
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'rocauc']
    test_rocauc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'rocauc']
    results['ROC-AUC'] = (train_rocauc, valid_rocauc, test_rocauc)
    return results
    
# def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
#     train_auc = roc_auc_score(train_true, train_pred)
#     valid_auc = roc_auc_score(val_true, val_pred)
#     test_auc = roc_auc_score(test_true, test_pred)
#     results = {}
#     results['AUC'] = (train_auc, valid_auc, test_auc)

#     return results

def precompute_adjs(A):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 0
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()
    return (A, D, D_log, D_common)


def RA_AA_CN(adjs, edge):
    A, D, D_log, D_common = adjs
    ra = []
    cn = []
    aa = []
    jc = []

    src, dst = edge
    # if len(src) < 200000:
    #     ra = np.array(np.sum(A[src].multiply(D[dst]), 1))
    #     aa = np.array(np.sum(A[src].multiply(D_log[dst]), 1))
    #     cn = np.array(np.sum(A[src].multiply(D_common[dst]), 1))
    # else:
    batch_size = 1000000
    for idx in tqdm(DataLoader(np.arange(src.size(0)), batch_size=batch_size, shuffle=False, drop_last=False)):
        ra.append(np.array(np.sum(A[src[idx]].multiply(D[dst[idx]]), 1)))
        aa.append(np.array(np.sum(A[src[idx]].multiply(D_log[dst[idx]]), 1)))
        cn.append(np.array(np.sum(A[src[idx]].multiply(D_common[dst[idx]]), 1)))
        jc.append(np.array(np.sum(A[src[idx]].multiply(D_common[dst[idx]]), 1))/(np.sum(A[src[idx]])+np.sum(A[dst[idx]])-np.sum(A[src[idx]].multiply(D_common[dst[idx]]), 1)))
    ra = np.concatenate(ra, axis=0)
    aa = np.concatenate(aa, axis=0)
    cn = np.concatenate(cn, axis=0)
    jc= np.concatenate(jc, axis=0)
    # count= np.sum(cn==0)
    # a = 1-count/cn.shape[0]
    # print(a)
        # break
    scores = np.concatenate([ra, aa, cn,jc], axis=1)
    return torch.FloatTensor(scores)

def CN(graph):
    edge_weight = torch.ones(graph.number_of_edges())
    adj = torch_sparse.SparseTensor(row=graph.edges()[0].cpu(),
                                    col=graph.edges()[1].cpu(),
                                    value=edge_weight,
                                    sparse_sizes=[graph.number_of_nodes(),graph.number_of_nodes()])
    #求邻接矩阵的平方，A2
    A2 = adj @ adj
    A2 = torch_sparse.remove_diag(A2)
    A2 = A2.to_scipy("csc")
    A2[adj.to_scipy("csc") > 0] = 0
    #至此，A2的值大于0的，两个节点间有共同邻居且无链路

    #A2的行列索引和元素值
    indices, values = torch_sparse.from_scipy(A2)
    #selected：A2非0元素值的索引
    selected = values.nonzero().squeeze(1)
    m = torch.cat([indices[:, selected].t(), values[selected].unsqueeze(1)], 1).long()
    # start_sets:all_edge
    no_neighbor = m[:, :2]
    #对A2的非0元素值和索引升序F,降序T排序，A2的值表示两个节点的共同邻居数
    sort_value,index = torch.sort(m[:, -1], descending=True)
    #neg_indices：对no_neighbor索引值按元素值升序重排
    neg_indices = no_neighbor[index]

    # if args.undirected==True:
    #     neg_indices = neg_indices[:candidate_num*2]
    # else:
    #     neg_indices = neg_indices[:candidate_num]

    # candidate_label = torch.ones(candidate_edge.size(0),dtype=int)
    # candidate_edge = torch.cat([candidate_edge,candidate_label.view(-1,1)],1)
    return neg_indices

def save_model(save_path, epoch, optimizer,model,predictor,emb):
    if emb is not None:
        torch.save({'epoch': epoch,
                    'optimizer_dict': optimizer.state_dict(),
                    # 'lr_scheduler_dict':lr_scheduler.state_dict(),
                    'model_dict': model.state_dict(),
                   'predictor_dict':predictor.state_dict(),
                   'emb_dict':emb.state_dict()},
                    save_path,_use_new_zipfile_serialization='False')
    else:
        torch.save({'epoch': epoch,
                    'optimizer_dict': optimizer.state_dict(),
                    # 'lr_scheduler_dict':lr_scheduler.state_dict(),
                    'model_dict': model.state_dict(),
                    'predictor_dict': predictor.state_dict()},
                   save_path, _use_new_zipfile_serialization='False')
    # print(f"\033[1;36m 第{epoch}次训练结果保存成功 \033[0m")

def load_model(save_name, optimizer, model,predictor,emb):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'],strict=False)
    optimizer.load_state_dict(model_data['optimizer_dict'])
    # lr_scheduler.load_state_dict(model_data['lr_scheduler_dict'])
    predictor.load_state_dict(model_data['predictor_dict'])
    if emb is not None:
        emb.load_state_dict(model_data['emb_dict'])
    print("model load success")
    start_epoch = model_data['epoch']
    return start_epoch

def do_edge_split(data,save_path, val_ratio=0.05, test_ratio=0.1):
    os.makedirs(save_path)
    random.seed(1)
    torch.manual_seed(1)
    # edge_index = torch.stack([data.edges()[0], data.edges()[1]]).t()
    u_ori, v_ori = data.edges()
    eids = np.arange(data.number_of_edges())
    delet = []
    for i in eids:
        if u_ori[i]<v_ori[i]:
            continue
        else:
            delet.append(i)
    data = dgl.remove_edges(data, delet)
    u, v = data.edges()
    eids = np.arange(data.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * test_ratio)
    valid_size = int(len(eids) * val_ratio)
    # train_size = data.number_of_edges() - test_size - valid_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    a = data.has_edges_between(71,1986)
    valid_pos_u, valid_pos_v = u[eids[test_size:test_size + valid_size]], v[eids[test_size:test_size + valid_size]]
    train_pos_u, train_pos_v = u[eids[test_size + valid_size:]], v[eids[test_size + valid_size:]]


    # Find all negative edges and split them for training and testing
    # adj = sp.coo_matrix((np.ones(len(u_ori)), (u_ori.numpy(), v_ori.numpy())),shape=(data.number_of_nodes(),data.number_of_nodes()))
    # adj_neg = 1 - adj.todense() - np.eye(data.number_of_nodes())
    # neg_u, neg_v = np.where(adj_neg != 0)
    # neg_u = torch.tensor(neg_u)
    # neg_v = torch.tensor(neg_v)
    #
    # neg_eids = np.random.choice(len(neg_u), data.number_of_edges())
    # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # valid_neg_u, valid_neg_v = neg_u[neg_eids[test_size:test_size+valid_size]], neg_v[neg_eids[test_size:test_size+valid_size]]

    train_g = dgl.remove_edges(data, eids[:test_size+valid_size])
    u, v = train_g.edges()
    train_g.add_edges(v,u)
    # train_g = dgl.to_bidirected(train_g,copy_ndata=True)
    # train_g2 = dgl.graph((train_pos_u, train_pos_v), num_nodes=data.number_of_nodes())

    neg_valid_edge = torch.randint(0, train_g.number_of_nodes(),
                                   (int(1.1 * valid_pos_u.size(0)),) + (2,), dtype=torch.long,)
    neg_valid_edge = neg_valid_edge[~train_g.has_edges_between(neg_valid_edge[:, 0], neg_valid_edge[:, 1])]

    neg_test_edge = torch.randint(0, train_g.number_of_nodes(),
                                   (int(1.1 * test_pos_u.size(0)),) + (2,), dtype=torch.long, )
    neg_test_edge = neg_test_edge[~train_g.has_edges_between(neg_test_edge[:, 0], neg_test_edge[:, 1])]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = torch.stack([train_pos_u,train_pos_v]).t()
    # split_edge['train']['edge_neg'] = torch.stack([train_pos_u,train_pos_v]).t()
    split_edge['valid']['edge'] = torch.stack([valid_pos_u,valid_pos_v]).t()
    # split_edge['valid']['edge_neg'] = torch.stack([valid_neg_u,valid_neg_v]).t()
    split_edge['valid']['edge_neg'] = neg_valid_edge
    split_edge['test']['edge'] = torch.stack([test_pos_u,test_pos_v]).t()
    # split_edge['test']['edge_neg'] = torch.stack([test_neg_u,test_neg_v]).t()
    split_edge['test']['edge_neg'] = neg_test_edge
    torch.save(split_edge,os.path.join(save_path,'split_edge.pt'),_use_new_zipfile_serialization=False)
    torch.save(train_g,os.path.join(save_path,'graph.pt'),_use_new_zipfile_serialization=False)
    return train_g,split_edge


def mask_test_edges_dgl(graph, adj,save_path):
    os.makedirs(save_path)
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    num_test = int(np.floor(edges_all.shape[0] / 10.0))
    num_val = int(np.floor(edges_all.shape[0] / 20.0))

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val : (num_val + num_test)]
    train_edge_idx = all_edge_idx[(num_val + num_test) :]
    test_edges = edges_all[test_edge_idx]
    val_edges = edges_all[val_edge_idx]
    train_edges = np.delete(
        edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0
    )

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = torch.tensor(train_edges)
    # split_edge['train']['edge_neg'] = torch.stack([train_pos_u,train_pos_v]).t()
    split_edge['valid']['edge'] = torch.tensor(val_edges)
    split_edge['valid']['edge_neg'] = torch.tensor(val_edges_false)
    split_edge['test']['edge'] = torch.tensor(test_edges)
    split_edge['test']['edge_neg'] = torch.tensor(test_edges_false)
    torch.save(split_edge, os.path.join(save_path, 'split_edge.pt'), _use_new_zipfile_serialization=False)
    train_g = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    torch.save(train_g, os.path.join(save_path, 'graph.pt'), _use_new_zipfile_serialization=False)
    return train_g, split_edge
    # NOTE: these edge lists only contain single direction of edge!
    # return (
    #     train_edge_idx,
    #     val_edges,
    #     val_edges_false,
    #     test_edges,
    #     test_edges_false,
    # )


class Log(object):
    def __init__(self, path, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(os.path.join(path,filename), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
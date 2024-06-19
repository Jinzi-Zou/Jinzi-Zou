import torch

from loss import *
from data import *
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,f1_score

def compute_pred(h, predictor, edges, batch_size):
    preds = []
    for perm in DataLoader(range(edges.size(0)), batch_size):
        print(perm)
        edge = edges[perm].t()

        preds += [predictor(h[edge[0]], h[edge[1]]).sigmoid().squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    return pred

def train_split(split_edge, device):
    source = split_edge['train']['source_node'].to(device)
    target = split_edge['train']['target_node'].to(device)
    pos_edge = torch.stack([source, target], dim=1)
    return pos_edge


def train(model, predictor, feat, edge_feat, graph, split_edge, optimizer, batch_size, args):
    model.train()
    predictor.train()

    if args.dataset == 'ogbl-citation2':
        pos_train_edge = train_split(split_edge, feat.device)
    else:
        pos_train_edge = split_edge['train']['edge'].to(feat.device)

    if 'weight' in split_edge['train']:
        edge_weight_margin = split_edge['train']['weight']
    else:
        edge_weight_margin = None
    if args.negative_sampler == 'strict_global':
        neg_train_edge = torch.randint(0, graph.number_of_nodes(),
                                       (int(1.1 * args.n_neg * pos_train_edge.size(0)),) + (2,), dtype=torch.long,
                                       device=feat.device)
        neg_train_edge = neg_train_edge[~graph.has_edges_between(neg_train_edge[:, 0], neg_train_edge[:, 1])]

        # neg_train_edge = negative_sampling(pos_train_edge.t(), graph.number_of_nodes(),
        #                                     args.n_neg * len(pos_train_edge))
        neg_src = neg_train_edge[:, 0]
        neg_dst = neg_train_edge[:, 1]
        if neg_train_edge.size(0) < pos_train_edge.size(0) * args.n_neg:
            k = pos_train_edge.size(0) * args.n_neg - neg_train_edge.size(0)
            rand_index = torch.randperm(neg_train_edge.size(0))[:k]
            neg_src = torch.cat((neg_src, neg_src[rand_index]))
            neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
        else:
            neg_src = neg_src[:pos_train_edge.size(0) * args.n_neg]
            neg_dst = neg_dst[:pos_train_edge.size(0) * args.n_neg]
        neg_train_edge = torch.reshape(
            torch.stack([neg_src, neg_dst], dim=1),
            (-1, args.n_neg, 2))

    # idx = torch.rand(pos_train_edge.shape[0]) < 0.15
    # pos_train_edge = pos_train_edge[idx, :]
    # neg_train_edge = negative_sampling(torch.stack(list(graph.edges()), dim=0), num_nodes=graph.number_of_nodes(), num_neg_samples=pos_train_edge.shape[0])

    total_loss = total_examples = ro_total_loss = auc_total_loss = 0
    # torch.manual_seed(epoch)
    # torch.cuda.manual_seed(0)
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(graph, feat, edge_feat)

        edge = pos_train_edge[perm]

        pos_out = predictor(h[edge[:, 0]], h[edge[:, 1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        if args.negative_sampler == 'global':
            # Just do some trivial random sampling.
            neg_edge = torch.randint(0, graph.number_of_nodes(), (args.n_neg * edge.size(0),) + (2,), dtype=torch.long,
                                     device=h.device)
        elif args.negative_sampler == 'strict_global':
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2))
        else:
            dst_neg = torch.randint(0, graph.number_of_nodes(), (args.n_neg * edge.size(0),) + (1,), dtype=torch.long,
                                    device=h.device)
            neg_edge = torch.cat([edge[:, 0].repeat(args.n_neg).unsqueeze(-1), dst_neg], dim=1)
        # edge = neg_train_edge[:, perm]

        neg_out = predictor(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
        if args.loss_func == 'RoL':
            loss_fct = GCELoss(q=args.q)
            roloss, aucloss = loss_fct(pos_out, neg_out, args.n_neg)
            loss = roloss + aucloss
        else:
            weight_margin = edge_weight_margin[perm].to(feat.device) if edge_weight_margin is not None else None
            loss = calculate_loss(pos_out, neg_out, args.n_neg, margin=weight_margin, loss_func_name=args.loss_func)
        # cross_out = predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)])
        # cross_loss = -torch.log(1 - cross_out.sigmoid() + 1e-15).sum()
        # loss = loss + 0.1 * cross_loss

        loss.backward()

        if args.clip_grad_norm > -1:
            if 'feat' not in graph.ndata:
                torch.nn.utils.clip_grad_norm_(feat, args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad_norm)

        optimizer.step()
        if args.loss_func == 'RoL':
            num_examples = pos_out.size(0)
            ro_total_loss += roloss.item() * num_examples
            auc_total_loss += aucloss.item() * num_examples
            total_examples += num_examples
        else:
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
    if args.loss_func == 'RoL':
        ro_total_loss = ro_total_loss / total_examples
        auc_total_loss = auc_total_loss / total_examples

        return ro_total_loss, auc_total_loss
    else:
        return total_loss / total_examples


def test_split(split, split_edge, device):
    source = split_edge[split]['source_node'].to(device)
    target = split_edge[split]['target_node'].to(device)
    target_neg = split_edge[split]['target_node_neg'].to(device)
    pos_edge = torch.stack([source, target], dim=1)
    neg_edge = torch.stack([source.view(-1, 1).repeat(1, 1000).view(-1), target_neg.view(-1)], dim=1)
    return pos_edge, neg_edge

@torch.no_grad()
def test(model, predictor, feat, edge_feat, graph, full_edge_feat, full_graph, split_edge, evaluator, batch_size, args,
         hitslist,epoch):
    model.eval()
    predictor.eval()
    if args.dataset == 'ogbl-citation2':
        pos_train_edge, neg_train_edge = test_split('eval_train', split_edge, feat.device)
        pos_valid_edge, neg_valid_edge = test_split('valid', split_edge, feat.device)
        pos_test_edge, neg_test_edge = test_split('test', split_edge, feat.device)
    else:
        pos_train_edge = split_edge['eval_train']['edge'].to(feat.device)
        pos_valid_edge = split_edge['valid']['edge'].to(feat.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(feat.device)
        pos_test_edge = split_edge['test']['edge'].to(feat.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(feat.device)

    h = model(graph, feat, edge_feat)
    # h = model(full_graph, feat, full_edge_feat)
    pos_train_pred = compute_pred(h, predictor, pos_train_edge, batch_size)
    pos_valid_pred = compute_pred(h, predictor, pos_valid_edge, batch_size)
    neg_valid_pred = compute_pred(h, predictor, neg_valid_edge, batch_size)

    h = model(full_graph, feat, full_edge_feat)
    pos_test_pred = compute_pred(h, predictor, pos_test_edge, batch_size)
    neg_test_pred = compute_pred(h, predictor, neg_test_edge, batch_size)
    # te_so = torch.sort(pos_test_pred,dim=0,descending=True)
    # ne_so = torch.sort(neg_test_pred,dim=0,descending=True)
    if args.eval_metric == 'hits':
        results = evaluate_hits(evaluator, hitslist, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(evaluator, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)


    train_pred = (pos_train_pred > 0.5).int().cpu().numpy()
    train_label = torch.ones(pos_train_pred.size(0)).int().cpu().numpy()
    valid_pred = (torch.cat([pos_valid_pred, neg_valid_pred]) > 0.5).int().cpu().numpy()
    valid_label = torch.cat(
        [torch.ones(pos_valid_pred.size(0)), torch.zeros(neg_valid_pred.size(0))]).int().cpu().numpy()
    test_pred = (torch.cat([pos_test_pred, neg_test_pred]) > 0.5).int().cpu().numpy()

    # accu = (sum((pos_test_pred>0.5))+sum((neg_test_pred<=0.5)))/(neg_test_pred.size(0)+pos_test_pred.size(0))
    # accu = sum((neg_test_pred<=0.5))/neg_test_pred.size(0)
    # accu1=sum((pos_test_pred>0.5))/pos_test_pred.size(0)
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

    valid_auc = roc_auc_score(valid_label, torch.cat([pos_valid_pred, neg_valid_pred]).cpu().numpy())
    test_auc = roc_auc_score(test_label, torch.cat([pos_test_pred, neg_test_pred]).cpu().numpy())
    # valid_auc = roc_auc_score(valid_label, valid_pred)
    # test_auc = roc_auc_score(test_label, test_pred)

    results['accuracy'] = (train_acc, valid_acc, test_acc)
    if epoch>5:
        results['recall'] = (train_recall, valid_recall, test_recall)
        results['precision'] = (train_precision, valid_precision, test_precision)
        results['F1-score'] = (train_f1, valid_f1, test_f1)
        results['AUC'] = (0, valid_auc, test_auc)
    if epoch==5:
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fpr, tpr, threshold = roc_curve(test_label, torch.cat([pos_test_pred, neg_test_pred]).cpu().numpy(), pos_label=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('Ture Positive Rate')
        plt.title('roc curve')
        plt.plot(fpr, tpr, color='b', linewidth=0.8)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.savefig('./roc_curve/roc_curve.png')

    return results
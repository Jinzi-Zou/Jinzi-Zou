import pandas as pd
from dgl.data import DGLDataset
import json
import networkx as nx
import requests
from collections import defaultdict
from ogb.linkproppred import DglLinkPropPredDataset
from utils import *

def get_dataset(dataset):

    if dataset == "ogbl-ddi":
        dataset = DglLinkPropPredDataset(name='ogbl-ddi')
    elif dataset == "ogbl-collab":
        dataset = DglLinkPropPredDataset(name='ogbl-collab')
    elif dataset == "cora":
        dataset = dgl.data.CoraGraphDataset()
    elif dataset == "citeseer":
        dataset = dgl.data.CiteseerGraphDataset()
    elif dataset == "pubmed":
        dataset = dgl.data.PubmedGraphDataset()
    elif dataset == "email":
        dataset = EmailDataset()
    elif dataset == "reddit":
        dataset = RedditDataset()
    elif dataset == "twitch":
        dataset = TwitchDataset()
    elif dataset == "fb":
        dataset = FBDataset()
    else:
        raise NotImplemented
    return dataset


def get_data(data_name):
    dataset = get_dataset(data_name)
    graph = dataset[0]
    if data_name in ['ogbl-ddi','ogbl-collab','email', 'fb', 'twitch','reddit']:
        split_edge = dataset.get_edge_split()
    if data_name in ['cora', 'citeseer', 'pubmed']:
        save_path = f'./dataset/{data_name}/processed'
        if not os.path.isdir(save_path):
            adj_ori = graph.adjacency_matrix().to_dense()
            graph, split_edge = mask_test_edges_dgl(graph, adj_ori,save_path)
            # graph, split_edge = do_edge_split(graph, save_path)

        else:
            split_edge = torch.load(os.path.join(save_path, 'split_edge.pt'))
            graph = torch.load(os.path.join(save_path, 'graph.pt'))

    return graph,split_edge


class FBDataset(DGLDataset):
    def __init__(self):
        self.split_edges = {'train': {}, 'valid': {}, 'test': {}}
        super().__init__(name='fb',
                         raw_dir='./dataset/fb/',
                         save_dir='./dataset/fb/processed/')

    def process(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        data = pd.read_csv('./dataset/fb/musae_facebook_edges.csv')
        edges = data.values.tolist()
        print(len(edges))
        edges = [list(sorted([int(edge[0]), int(edge[1])])) for edge in edges]
        print(len(edges))
        edges = [edge for edge in edges if edge[0] < edge[1]]  # remove self loops
        print(len(edges))

        assert np.all([ab[0] < ab[1] for ab in edges])

        num_edges = len(edges)
        print('total amount of edges:', num_edges)

        # random.shuffle(edges)
        g = dgl.graph((torch.tensor(edges)[:, 0], torch.tensor(edges)[:, 1]))
        u,v = g.edges()
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.number_of_nodes(),g.number_of_nodes()))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_u = torch.tensor(neg_u)
        neg_v = torch.tensor(neg_v)
        neg_eids = np.random.choice(len(neg_u), int(0.2*num_edges))
        test_neg_u, test_neg_v = neg_u[neg_eids[:int(0.1*num_edges)]], neg_v[neg_eids[:int(0.1*num_edges)]]
        valid_neg_u, valid_neg_v = neg_u[neg_eids[int(0.1*num_edges):]], neg_v[neg_eids[int(0.1*num_edges):]]

        random.shuffle(edges)
        train_edges = edges[:int(0.8 * num_edges)]
        print('train amount:', len(train_edges))
        val_edges = edges[int(0.8 * num_edges):int(0.9 * num_edges)]
        print('val amount:', len(val_edges))
        test_edges = edges[int(0.9 * num_edges):]
        print('test amount:', len(test_edges))
        g = dgl.graph((torch.tensor(train_edges)[:, 0], torch.tensor(train_edges)[:, 1]),num_nodes=g.number_of_nodes())
        self.graph = dgl.to_bidirected(g)

        with open(f"./dataset/fb/musae_facebook_features.json", 'r') as f:
            j = json.load(f)
        n = self.graph.num_nodes()
        features = np.zeros((n, 4714))
        for node, feats in j.items():
            if int(node) >= n:
                continue
                print('continued')
            features[int(node), np.array(feats, dtype=int)] = 1
        features = features[:, np.sum(features, axis=0) != 0]
        self.graph.ndata['feat'] = torch.tensor(features).float()

        self.split_edges['train']['edge']= torch.tensor(train_edges)
        self.split_edges['valid']['edge'] = torch.tensor(val_edges)
        self.split_edges['valid']['edge_neg'] = torch.stack([valid_neg_u,valid_neg_v]).t()
        self.split_edges['test']['edge'] = torch.tensor(test_edges)
        self.split_edges['test']['edge_neg'] = torch.stack([test_neg_u,test_neg_v]).t()

    def save(self):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        torch.save(self.graph,os.path.join(self._save_dir,'graph.pt'),_use_new_zipfile_serialization=False)
        torch.save(self.split_edges, os.path.join(self._save_dir,'edge_split.pt'), _use_new_zipfile_serialization=False)
        # save_graphs(os.path.join(self._save_dir,'/graph.pt'),self.graph)


    def has_cache(self):
        graph_path = os.path.join(self._save_dir, 'graph.pt')
        split_path = os.path.join(self._save_dir, 'edge_split.pt')
        if os.path.exists(graph_path) and os.path.exists(split_path):
            return True
        else:
            return False

    def load(self):
        self.graph = torch.load(os.path.join(self._save_dir,'graph.pt'))
        self.split_edges = torch.load(os.path.join(self._save_dir,'edge_split.pt'))
        # self.graphs = load_graphs(os.path.join(self._save_dir,'/graph.pt'))

    def get_edge_split(self):
        return self.split_edges


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class EmailDataset(DGLDataset):
    def __init__(self):
        self.split_edges = {'train': {}, 'valid': {}, 'test': {}}
        super().__init__(name='email',
                         raw_dir='./dataset/email/',
                         save_dir='./dataset/email/processed/')

    def process(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        edge_df = pd.read_csv('./dataset/email/email-Eu-core-temporal.txt',
                           sep=' ', header=None, names=['src', 'dst', 'unixts']).sort_values('unixts')
        uniq_edge_df = edge_df[~edge_df.duplicated(subset=['src', 'dst'], keep='last')]

        # G = nx.OrderedDiGraph()
        G = nx.DiGraph()
        for idx, row in uniq_edge_df.iterrows():
            src, dst, timestamp = row.loc['src'], row.loc['dst'], row['unixts']
            G.add_edge(src, dst, time=timestamp)

        nodes = np.unique(np.concatenate([edge_df['src'], edge_df['dst']]))
        for node in nodes:
            src_times = np.array(edge_df[edge_df['src'] == node]['unixts'])
            dst_times = np.array(edge_df[edge_df['dst'] == node]['unixts'])
            all_times = np.concatenate([src_times, dst_times])
            # randomly select 256 or length of total times
            num_times = min(len(all_times), 256)
            feature = np.concatenate([np.random.permutation(all_times)[:num_times], np.zeros(256 - num_times)])
            mask = np.array([True] * num_times + [False] * (256 - num_times))

            G.nodes[node]["feature"] = feature
            G.nodes[node]["mask"] = mask

        G = nx.convert_node_labels_to_integers(G, label_attribute='original')

        data_edges = list(G.edges(data=True))
        data_edges = [(u, v, t['time']) for u, v, t in data_edges]
        data_edges = sorted(data_edges, key=lambda x: x[2])
        edges = np.array([(u, v) for u, v, t in data_edges])

        n = edges.shape[0]
        edges = torch.tensor(edges)

        u = edges[:,0]
        v = edges[:,1]
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(len(G), len(G)))
        adj_neg = 1 - adj.todense() - np.eye(len(G))
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_u = torch.tensor(neg_u)
        neg_v = torch.tensor(neg_v)
        neg_eids = np.random.choice(len(neg_u), int(0.2 * n))
        test_neg_u, test_neg_v = neg_u[neg_eids[:int(0.1 * n)]], neg_v[neg_eids[:int(0.1 * n)]]
        valid_neg_u, valid_neg_v = neg_u[neg_eids[int(0.1 * n):]], neg_v[neg_eids[int(0.1 * n):]]
        train_edge = edges[:int(0.8 * n)]
        val_edge = edges[int(0.8 * n):int(0.9 * n)]
        test_edge = edges[int(0.9 * n):]


        g = dgl.graph((train_edge[:,0],train_edge[:,1]),num_nodes=len(G))
        self.graph = dgl.to_bidirected(g)
        self.split_edges['train']['edge'] = train_edge
        self.split_edges['valid']['edge'] = val_edge
        self.split_edges['valid']['edge_neg'] = torch.stack([valid_neg_u, valid_neg_v]).t()
        self.split_edges['test']['edge'] = test_edge
        self.split_edges['test']['edge_neg'] = torch.stack([test_neg_u, test_neg_v]).t()

        data = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                data[key] = [value] if i == 0 else data[key] + [value]

        for key, item in data.items():
            item = np.array(item).astype(np.float32)
            data[key] = torch.tensor(item)

        self.graph.ndata['feat'] = data['feature']


    def save(self):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        torch.save(self.graph,os.path.join(self._save_dir,'graph.pt'),_use_new_zipfile_serialization=False)
        torch.save(self.split_edges, os.path.join(self._save_dir,'edge_split.pt'), _use_new_zipfile_serialization=False)
        # save_graphs(os.path.join(self._save_dir,'/graph.pt'),self.graph)


    def has_cache(self):
        graph_path = os.path.join(self._save_dir, 'graph.pt')
        split_path = os.path.join(self._save_dir, 'edge_split.pt')
        if os.path.exists(graph_path) and os.path.exists(split_path):
            return True
        else:
            return False

    def load(self):
        self.graph = torch.load(os.path.join(self._save_dir,'graph.pt'))
        self.split_edges = torch.load(os.path.join(self._save_dir,'edge_split.pt'))
        # self.graphs = load_graphs(os.path.join(self._save_dir,'/graph.pt'))

    def get_edge_split(self):
        return self.split_edges


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class RedditDataset(DGLDataset):
    def __init__(self):
        self.edge_body_file = "reddit-body.tsv"
        self.edge_title_file = "reddit-title.tsv"
        self.embeddings_file = "reddit-embeddings.csv"
        self._num_nodes = 30744
        self._num_edges = 277041
        self.split_edges = {'train': {}, 'valid': {}, 'test': {}}
        super().__init__(name='reddit',
                         raw_dir='./dataset/reddit/',
                         save_dir='./dataset/reddit/processed/'
                         )

    def process(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        body_df = pd.read_csv(os.path.join(self.raw_dir, self.edge_body_file), sep='\t')
        body_df['TIMESTAMP'] = pd.to_datetime(body_df['TIMESTAMP'])
        title_df = pd.read_csv(os.path.join(self.raw_dir, self.edge_title_file), sep='\t')
        title_df['TIMESTAMP'] = pd.to_datetime(title_df['TIMESTAMP'])
        master_df = pd.concat([body_df, title_df], ignore_index=True)

        emb_df = pd.read_csv(os.path.join(self.raw_dir, self.embeddings_file), header=None)
        emb_subs = set(np.array(emb_df.loc[:, 0]))

        to_remove = []
        for idx, row in master_df.iterrows():
            src = row.loc['SOURCE_SUBREDDIT']
            dst = row.loc['TARGET_SUBREDDIT']
            if src not in emb_subs or dst not in emb_subs:
                to_remove.append(idx)

        clean_df = master_df.drop(to_remove)
        clean_df = clean_df.sort_values("TIMESTAMP")

        G = nx.DiGraph()
        for idx, row in clean_df.iterrows():
            src, dst = row.loc['SOURCE_SUBREDDIT'], row.loc['TARGET_SUBREDDIT']
            timestamp = row['TIMESTAMP']
            G.add_edge(src, dst, time=timestamp)

        for idx, row in emb_df.iterrows():
            subreddit = row.iloc[0]
            feature = row.iloc[1:]
            if subreddit in G.nodes:
                G.nodes[subreddit]['feature'] = feature

        G = nx.convert_node_labels_to_integers(G)

        data_edges = list(G.edges(data=True))
        data_edges = [(u, v, t['time']) for u, v, t in data_edges]
        data_edges = sorted(data_edges, key=lambda x: x[2])

        node_times = defaultdict(list)
        for u, v, t in data_edges:
            node_times[u].append(t)
            node_times[v].append(t)

        node_split = dict()
        for node, times in node_times.items():
            assert node not in node_split
            node_split[node] = max(times)

        node_time_list = [(n, t) for n, t in node_split.items()]
        node_time_list = sorted(node_time_list, key=lambda x: x[1])
        node_time_list = [n[0] for n in node_time_list]
        assert len(node_time_list) == self._num_nodes


        edges = np.array([(u, v) for u, v, t in data_edges])

        n = edges.shape[0]
        edges = torch.tensor(edges)
        u = edges[:, 0]
        v = edges[:, 1]
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(len(G), len(G)))
        adj_neg = 1 - adj.todense() - np.eye(len(G))
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_u = torch.tensor(neg_u)
        neg_v = torch.tensor(neg_v)
        neg_eids = np.random.choice(len(neg_u), int(0.2 * n))
        test_neg_u, test_neg_v = neg_u[neg_eids[:int(0.1 * n)]], neg_v[neg_eids[:int(0.1 * n)]]
        valid_neg_u, valid_neg_v = neg_u[neg_eids[int(0.1 * n):]], neg_v[neg_eids[int(0.1 * n):]]
        train_edge = edges[:int(0.8 * n)]
        val_edge = edges[int(0.8 * n):int(0.9 * n)]
        test_edge = edges[int(0.9 * n):]

        g = dgl.graph((train_edge[:, 0], train_edge[:, 1]), num_nodes=len(G))
        self.graph = dgl.to_bidirected(g)
        self.split_edges['train']['edge'] = train_edge
        self.split_edges['valid']['edge'] = val_edge
        self.split_edges['valid']['edge_neg'] = torch.stack([valid_neg_u, valid_neg_v]).t()
        self.split_edges['test']['edge'] = test_edge
        self.split_edges['test']['edge_neg'] = torch.stack([test_neg_u, test_neg_v]).t()

        data = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                data[key] = [value] if i == 0 else data[key] + [value]

        for key, item in data.items():
            item = np.array(item).astype(np.float32)
            data[key] = torch.tensor(item)

        self.graph.ndata['feat'] = data['feature']


    def save(self):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        torch.save(self.graph,os.path.join(self._save_dir,'graph.pt'),_use_new_zipfile_serialization=False)
        torch.save(self.split_edges, os.path.join(self._save_dir,'edge_split.pt'), _use_new_zipfile_serialization=False)


    def download(self):
        print('Downloading...')
        download_url('http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv',
                     os.path.join(self.raw_dir, self.edge_body_file))
        download_url('http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv',
                     os.path.join(self.raw_dir, self.edge_title_file))
        download_url('http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv',
                     os.path.join(self.raw_dir, self.embeddings_file))

    def has_cache(self):
        graph_path = os.path.join(self._save_dir, 'graph.pt')
        split_path = os.path.join(self._save_dir, 'edge_split.pt')
        if os.path.exists(graph_path) and os.path.exists(split_path):
            return True
        else:
            return False

    def load(self):
        self.graph = torch.load(os.path.join(self._save_dir, 'graph.pt'))
        self.split_edges = torch.load(os.path.join(self._save_dir, 'edge_split.pt'))

    def get_edge_split(self):
        return self.split_edges

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class TwitchDataset(DGLDataset):
    def __init__(self):
        self._num_nodes = 9498
        self.split_edges = {'train': {}, 'valid': {}, 'test': {}}
        super().__init__(name='Twitch',
                         raw_dir='./dataset/twitch/',
                         save_dir='./dataset/twitch/processed/'
                         )

    def process(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        data = pd.read_csv('./dataset/twitch/musae_DE_edges.csv')
        edges = data.values.tolist()
        print(len(edges))
        edges = [list(sorted([int(edge[0]), int(edge[1])])) for edge in edges]
        print(len(edges))
        edges = [edge for edge in edges if edge[0] < edge[1]]  # remove self loops
        print(len(edges))

        assert np.all([ab[0] < ab[1] for ab in edges])

        num_edges = len(edges)
        print('total amount of edges:', num_edges)


        g = dgl.graph((torch.tensor(edges)[:, 0], torch.tensor(edges)[:, 1]))
        u, v = g.edges()
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_u = torch.tensor(neg_u)
        neg_v = torch.tensor(neg_v)
        neg_eids = np.random.choice(len(neg_u), int(0.2 * num_edges))
        test_neg_u, test_neg_v = neg_u[neg_eids[:int(0.1 * num_edges)]], neg_v[neg_eids[:int(0.1 * num_edges)]]
        valid_neg_u, valid_neg_v = neg_u[neg_eids[int(0.1 * num_edges):]], neg_v[neg_eids[int(0.1 * num_edges):]]

        random.shuffle(edges)
        train_edges = edges[:int(0.8 * num_edges)]
        print('train amount:', len(train_edges))
        val_edges = edges[int(0.8 * num_edges):int(0.9 * num_edges)]
        print('val amount:', len(val_edges))
        test_edges = edges[int(0.9 * num_edges):]
        print('test amount:', len(test_edges))
        g = dgl.graph((torch.tensor(train_edges)[:, 0], torch.tensor(train_edges)[:, 1]), num_nodes=g.number_of_nodes())
        self.graph = dgl.to_bidirected(g)

        with open(f"./dataset/twitch/musae_DE_features.json", 'r') as f:
            j = json.load(f)
        n = self.graph.num_nodes()
        features = np.zeros((n, 4714))
        for node, feats in j.items():
            if int(node) >= n:
                continue
                print('continued')
            features[int(node), np.array(feats, dtype=int)] = 1
        features = features[:, np.sum(features, axis=0) != 0]
        self.graph.ndata['feat'] = torch.tensor(features).float()

        self.split_edges['train']['edge'] = torch.tensor(train_edges)
        self.split_edges['valid']['edge'] = torch.tensor(val_edges)
        self.split_edges['valid']['edge_neg'] = torch.stack([valid_neg_u, valid_neg_v]).t()
        self.split_edges['test']['edge'] = torch.tensor(test_edges)
        self.split_edges['test']['edge_neg'] = torch.stack([test_neg_u, test_neg_v]).t()

    def save(self):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        torch.save(self.graph,os.path.join(self._save_dir,'graph.pt'),_use_new_zipfile_serialization=False)
        torch.save(self.split_edges, os.path.join(self._save_dir,'edge_split.pt'), _use_new_zipfile_serialization=False)
        # save_graphs(os.path.join(self._save_dir,'/graph.pt'),self.graph)


    def has_cache(self):
        graph_path = os.path.join(self._save_dir, 'graph.pt')
        split_path = os.path.join(self._save_dir, 'edge_split.pt')
        if os.path.exists(graph_path) and os.path.exists(split_path):
            return True
        else:
            return False

    def load(self):
        self.graph = torch.load(os.path.join(self._save_dir,'graph.pt'))
        self.split_edges = torch.load(os.path.join(self._save_dir,'edge_split.pt'))
        # self.graphs = load_graphs(os.path.join(self._save_dir,'/graph.pt'))

    def get_edge_split(self):
        return self.split_edges


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def download_url(url, filename):
    get_response = requests.get(url,stream=True)
    with open(filename, 'wb') as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



a = EmailDataset()
print('dd')








import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from tqdm.autonotebook import tqdm


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


# def load_data_v1(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))

#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))

#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)

#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)

#     return adj, features, labels, idx_train, idx_val, idx_test

def load_data():
    print('Loading dataset...')
    # generate_adj_feature() is used to generate adjacency matrix and feature vectors
    # But here we skip this step, and directly read from the pickle we stored before.
    adj, features = generate_adj_feature()

    adj = pd.read_pickle('../data/author_matrix.pickle')
    labels = pd.read_csv("../data/DBLP_four_area/author_label.txt", sep = "\t", names=["authorID", "Label"])
    features = pd.read_pickle('../data/author_feature.pickle')

    features = features.loc[labels['authorID'], :]
    adj = adj.loc[labels['authorID'], labels['authorID']]
    adj = np.array(adj, dtype=float)

    for idx, row in enumerate(adj):
        row[idx] = 1
        for idy, col in enumerate(row):
            if row[idy] != 0:
                row[idy] = 1

    labels = np.array(labels['Label'])
    labels = encode_onehot(labels)
    features = normalize(features)
    adj = normalize(adj)


    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sp.csr_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

def generate_adj_feature():

    #### generate author's features: "features" ####

    # features is a dataframe,
    # with column header being 'label', row index being 'author id',
    # and cell value being the number of papers that the coresponding author has written,
    # for the corresponding labeled venue.
    # Example: Cell [1, 1] = 93, meaning author 1 has written 93 paper belonging to label 1 venues.
    author_df = pd.read_csv("../data/four_area/author.txt", sep="\t", names=["ID", "Author name"], encoding='utf8')
    conf_df = pd.read_csv("../data/four_area/conf.txt", sep="\t", names=["ID", "Conference name"])
    paper_df = pd.read_csv("../data/four_area/paper.txt", sep="\t", names=["ID", "Paper title"])
    term_df = pd.read_csv("../data/four_area/term.txt", sep="\t", names=["ID", "Term"])
    paper_author = pd.read_csv("../data/four_area/paper_author.txt", sep="\t", names=["paperID", "authorID"])
    paper_conf = pd.read_csv("../data/four_area/paper_conf.txt", sep="\t", names=["paperID", "confID"])
    paper_term = pd.read_csv("../data/four_area/paper_term.txt", sep="\t", names=["paperID", "termID"])
    author_dict = pd.read_csv("../data/DBLP_four_area/cleaned_author_dict.txt", sep="\t", names=["ID", "Author name"],
                              encoding='utf8')
    conf_dict = pd.read_csv("../data/DBLP_four_area/conf_dict.txt", sep="\t", names=["ID", "Conference name"])
    term_dict = pd.read_csv("../data/DBLP_four_area/term_dict.txt", sep="\t", names=["ID", "Term"])
    author_label = pd.read_csv("../data/DBLP_four_area/author_label.txt", sep="\t", names=["authorID", "Label"])
    conf_label = pd.read_csv("../data/DBLP_four_area/conf_label.txt", sep="\t",
                             names=["confID", "Conference name", "Label"])

    # Merging dataframes together
    conf_dict_m = pd.merge(conf_dict, conf_df, on='Conference name')
    author_dict_m = pd.merge(author_dict, author_df, on='Author name')
    paper_conf_m = pd.merge(conf_dict_m, paper_conf, left_on='ID_y', right_on='confID')
    paper_conf_m = paper_conf_m.drop(columns=['Conference name', 'ID_y', 'confID'])
    paper_label_m = pd.merge(paper_conf_m, conf_label, left_on='ID_x', right_on='confID')
    paper_label_m = paper_label_m.drop(columns=['Conference name', 'ID_x', 'confID'])

    author_paper_label_m = pd.merge(paper_label_m, paper_author, on='paperID')

    features= pd.DataFrame(np.zeros(shape=(28702,4)),
                                  columns=[1,2,3,4],
                                  index=author_dict['ID'].unique()
                             )
    for author in tqdm(author_paper_label_m['authorID'].unique()):
        author_dict_ID = int(author_dict_m[author_dict_m["ID_y"] == author]['ID_x'].to_string(index=False).strip())
        value_count = author_paper_label_m[author_paper_label_m['authorID'] == author]['Label'].value_counts()
        for vc in value_count.iteritems():
            label = vc[0]
            count = vc[1]
            features.at[author_dict_ID, label]=count


    #### generate adjacency matrix among authors: "adj" ####

    adj = pd.DataFrame(np.zeros(shape=(28702, 28702)),
                              columns=np.arange(1, 28703),
                              index=np.arange(1, 28703))
    for paper in tqdm(paper_author['paperID']):
        authors = list(paper_author[paper_author['paperID'] == paper]['authorID'])
        if (len(authors) > 1):
            for i in range(len(authors)):
                authorID1 = authors[i]
                authorDictID1 = author_dict_m[author_dict_m["ID_y"] == authorID1]['ID_x'].to_string(index=False).strip()
                for j in range(i + 1, len(authors)):
                    authorID2 = authors[j]
                    authorDictID2 = author_dict_m[author_dict_m["ID_y"] == authorID2]['ID_x'].to_string(
                        index=False).strip()
                    adj.at[int(authorDictID1), int(authorDictID2)] += 1
                    adj.at[int(authorDictID2), int(authorDictID1)] += 1
        else:
            authorID1 = authors[0]
            authorDictID1 = author_dict_m[author_dict_m["ID_y"] == authorID1]['ID_x'].to_string(index=False).strip()
            adj.at[int(authorDictID1), int(authorDictID1)] = 1
    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

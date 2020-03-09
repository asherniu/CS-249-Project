import pickle
import numpy as np
import pandas as pd
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM



with open('af_py2.pickle', 'rb') as f:
    raw_author_feature = pickle.load(f)

author_label = pd.read_csv("../data/DBLP_four_area/author_label.txt", 
                            sep = "\t", names=["authorID", "Label"])
pd_labelled_author_features = raw_author_feature.loc[author_label['authorID'],:]
unnormed_af = pd_labelled_author_features.values

cleaned_author_features = unnormed_af/np.linalg.norm(unnormed_af, ord=2, axis=1, keepdims=True)
cleaned_author_labels = author_label.values[:,1]-1 # pystruct requires label start from 0
with open('labelled_author_adj_matrix.pickle', 'rb') as f:
    cleaned_author_adj = pickle.load(f).values




def get_dataset_portion(cleaned_author_features, cleaned_author_labels, 
                        cleaned_author_adj, is_in_set):
    """return a portion of the whole dataset"""
    new_author_features = []
    new_author_labels = []
    new_edges = []

    total_num = len(cleaned_author_features)
    selected_old_ids = []
    for old_id in range(total_num):
        if is_in_set[old_id]:
            selected_old_ids.append(old_id)
    
    # build new author id and set up edges
    for new_id, old_id in enumerate(selected_old_ids):
        new_author_features.append(cleaned_author_features[old_id])
        new_author_labels.append(cleaned_author_labels[old_id])
        for offset_to_new_id, old_id_peer in enumerate(selected_old_ids[new_id+1:]):
            if cleaned_author_adj[old_id][old_id_peer] > 0:
                new_edges.append([new_id, new_id+offset_to_new_id+1]) 

    f, l, e = np.array(new_author_features), np.array(new_author_labels), np.array(new_edges) 
    return f, l, e




def graph_crf(train_set_portion=0.9):
    print('train_set_portion %s' % train_set_portion)
    rand_arr = np.random.rand(len(cleaned_author_labels))
    features_train, labels_train, edges_train = \
        get_dataset_portion(cleaned_author_features, cleaned_author_labels,
                            cleaned_author_adj, rand_arr < train_set_portion)
    features_test, labels_test, edges_test = \
        get_dataset_portion(cleaned_author_features, cleaned_author_labels,
                            cleaned_author_adj, rand_arr >= train_set_portion)
        
    X_train = [(features_train, edges_train)]
    Y_train = [labels_train]
    X_test = [(features_test, edges_test)]
    Y_test = [labels_test]


    print('X_train[0][0] shape', X_train[0][0].shape)
    print('Y_train[0] shape', Y_train[0].shape)

    model = GraphCRF(directed=False, inference_method="max-product")
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10, verbose=3)
    print('training')
    ssvm.fit(X_train, Y_train) 
    score = ssvm.score(X_test, Y_test) 
    print('score: %.4f' % score)




if __name__ == '__main__':
    graph_crf(0.03)
    for i in range(10):
        graph_crf(0.2+i/10.0)

    
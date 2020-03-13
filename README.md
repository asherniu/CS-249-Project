# CS-249-Project

### Download the adjacency matrix and feature vector here https://drive.google.com/drive/folders/1ZdROyFY3KTh7ARqleMKwfYFL8BoWBsLD?usp=sharing or run the generation codes in model 2 or 3 (about 10-15 min to generate) 

# Model 1 Naive Bayes: 
You can find the codes in the NaiveBayes_Jupyter.ipynb. It will require author_feature.pickle and author_features.pickle. Both can be generated, downloaded and came with the repo. You can also run NaiveBayes_Python.py if you prefer python codes. 



# Model 2 Pairwise Conditional Random Field:
The codes are in the ModelPairwiseCRF directory. 
Implemented with [PyStruct](https://pystruct.github.io/index.html)

## Prerequisite
1. numpy
2. pandas
3. [pystruct](https://pystruct.github.io/installation.html)
(note that pystruct is only supported on Python2, Python3.6 or less,
we use Python2.7 to test the code)
4. The model requires `af_py2.pickle` which stores the author feature matrix with 
pickle protocol 2 (since we use Python2). We provide the pickle file in the directory, 
or you can re-generate the pickle file, run `gen_author_feature_py2_pickle.py`.

## Train and Evaluate
`python crf.py`

## Experiment Results
We have run the model with `python crf.py > crf.log`, 
you can either rerun the model or directly check our results in `crf.log`




# Model3 Graph Convolutional Networks in PyTorch:


PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](gcn-figure.png)


## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## Dataset: 

```
gcn/data/DBLP_four_area/ and gcn/data/four_area/ 
(note: we directly read dataset from author_matrix.pickle and author_feature.pickle, which were generated from the dataset through function "generate_adj_feature()" in gcn/pygcn/utils.py)
You can generate the pickle files or download them, and modify the path of reading function "load_data()" in gcn/pygcn/utils.py.
```

## Model Design:
```
Data structure: authors as the nodes in the graph, Adjacency matrix between authors, features of authors, labels of authors
Generate our own dataset: "load_data()" in gcn/pygcn/utils.py.
```
## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)


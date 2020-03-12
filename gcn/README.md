Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](figure.png)


## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## Dataset: data/DBLP_four_area/ and data/four_area/ (note: author_feature.pickle was generated from the dataset)

## Model Design:

	### Data structure: authors as the nodes in the graph, Adjacency matrix between authors, features of authors, labels of authors
	### Generate our own dataset: "load_data_v1()" in pygcn/utils.py.

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

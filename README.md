# HDC_GraphHD

- Following introduction of **GraphHD** in the paper [GraphHD](https://arxiv.org/abs/2205.07826), this repository shows a simple implemenation of **GraphHD** in Python.
- The other implementation of **GraphHD** is [Graph classification notebook](https://colab.research.google.com/drive/1NrmCc99GrkmHm_VLs5nv9Q7BCbCLs0ar?usp=sharing), in which the author provides the implementation in **Python Boolean Hyper-Vectors(BHV)**, However this repo uses **TorchHD** instead.

=================================================================================


### Requirements
```bash
pip install -r requirements.txt
# !pip install torch-hd
```

=================================================================================

### Introduction
- **GraphHD** is a novel graph representation learning method that uses **Hyperdimensional Computing(HDC)** to encode graph structures into high-dimensional vectors.
- The goal of the paper is to evaluate GraphHD on real-world graph classification problems. 
- In general graph shows a more complex representation in which the information about such enetites and the relationships between them is non-Euclidean.
- In this manner GraphHD take advantage of HDC to represent the graph structure in a high-dimensional space, which are typically dimensionality independent and then perform a graph classification task. 


- The procedure of **GraphHD** is as follows:
  - **Graph Encoding**: The graph is encoded into a high-dimensional vector using HDC.
  - **Graph Classification**: The encoded graph is then used to train a classifier for graph classification tasks.
  - **Graph Decoding**: The classifier is used to predict the class of a new graph by decoding the high-dimensional vector into a class label.

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

=================================================================================
#### Graph Encoding
- In the graph structure, the nodes and edges are represented as high-dimensional vectors. However, an identifier has to be determined for each node and edge in the graph to perform a correspondence between the vertices and edges of the graph and the high-dimensional vectors.
- The identifier in this paper is _PageRank_ centrality, which is a measure of the importance of a node in a graph, which works as follows, algorithm receives a graph as input and returns, for each vertex, $v_i \in V$, a value $c(v_i) \in [0, 1]$ that measures its “importance” in the graph.
- Bt this way a meaningful identifier is published between vertices in different graphs. Accordingly, vertices of different graphs, but with the same centrality rank, are encoded to the same random hypervector from the basis set.
- After creating the hypervectors for each vertex, GraphHD makes use of these representations to also encode each edge $(vi,vj) \in E(G)$. The edge encoding function $Enc_e$ is defined as follows:

$$  Enc_e(v_i, v_j) = Enc_v(v_i) \cross Enc_v(v_j) $$

- The $\times$ symbol represents the binding operation in HDC, which is the standard operation to represent an association between a pair of hypervectors, similar to the role of an edge in a graph. The result of the binding operation is a third vector, statistically quasi-orthogonal to the operand vectors, which we name edge-hypervectors.

=================================================================================

#### Training Procedure

The following is the training procedure used in the GraphHD approach for hyperdimensional computing (HDC) model development:

```plaintext
Algorithm 1: GraphHD training procedure

1. GraphHD_Training (GL);
   - Input: A training set of graphs GL with their respective labels and a set of random vertex-hypervectors Hv.
   - Output: A trained HDC model M consisting of the class vectors {C1, ..., Ck}

2. for each class label i ∈ {1, ..., k} do
3.     Hℓ ← ∅
4.     for each graph G ∈ GL such that ℓ(G) = i do
5.         HG ← ∅
6.         for each edge e ∈ E(G) do
7.             HG ← HG ∪ Enc(e)
8.         Hℓ ← Hℓ ∪ bundle(HG)
9.     Ci ← bundle(Hℓ)
10. return {C1, ..., Ck}
```
=================================================================================

Useful links:
- [BHV](https://github.com/Adam-Vandervorst/PyBHV?tab=readme-ov-file)
- [HD-Computing](https://www.hd-computing.com/)
- [dataset](https://chrsmrrs.github.io/datasets/docs/datasets/)


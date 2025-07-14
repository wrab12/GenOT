import torch
import torch.nn as nn
from typing import List, Optional, Union, Any
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, GCNConv

# class Discriminator(nn.Module):
#     def __init__(self, n_h):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#         c_x = c.expand_as(h_pl)
#
#         sc_1 = self.f_k(h_pl, c_x)
#         sc_2 = self.f_k(h_mi, c_x)
#
#         if s_bias1 is not None:
#             sc_1 += s_bias1
#         if s_bias2 is not None:
#             sc_2 += s_bias2
#
#         logits = torch.cat((sc_1, sc_2), 1)
#
#         return logits
#
#
#
#
# class AvgReadout(nn.Module):
#     def __init__(self):
#         super(AvgReadout, self).__init__()
#
#     def forward(self, emb, mask=None):
#         vsum = torch.mm(mask, emb)
#         row_sum = torch.sum(mask, 1)
#         row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
#         global_emb = vsum / row_sum
#
#         return F.normalize(global_emb, p=2, dim=1)
#
#
#
#
# def sym_norm(edge_index: torch.Tensor,
#              num_nodes: int,
#              edge_weight: Optional[Union[Any, torch.Tensor]] = None,
#              improved: Optional[bool] = False,
#              dtype: Optional[Any] = None
#              ) -> List:
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
#
#
#     edge_index = edge_index.long()
#
#     fill_value = 1 if not improved else 2
#     edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
#
#     row, col = edge_index
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#     return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#
#
# class CombUnweighted(MessagePassing):
#     def __init__(self, K: Optional[int] = 1,
#                  cached: Optional[bool] = False,
#                  bias: Optional[bool] = True,
#                  **kwargs):
#         super(CombUnweighted, self).__init__(aggr='add', **kwargs)
#         self.K = K
#
#     def forward(self, x: torch.Tensor,
#                 edge_index: torch.Tensor,
#                 edge_weight: Union[torch.Tensor, None] = None):
#         edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
#                                     dtype=x.dtype)
#
#         xs = [x]
#         for k in range(self.K):
#             xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
#         return torch.cat(xs, dim=1)
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
#
#     def __repr__(self):
#         return '{}(K={})'.format(self.__class__.__name__, self.K)
#
#
#
# class MGCN(nn.Module):
#     def __init__(self, in_features, out_features, graph_neigh, dropout=0.3, act=F.leaky_relu):
#         super(MGCN, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.graph_neigh = graph_neigh
#         self.dropout = dropout
#         self.act = act
#
#
#         # GATConv layers with CombUnweighted integration
#         self.conv1 = CombUnweighted(K=12)
#         self.gat_conv1 = GCNConv(in_features * (12 + 1), out_features)
#         self.gat_conv2 = GCNConv(out_features, in_features)
#
#
#         # Graph Normalization layers
#
#
#         # Other components
#         self.disc = Discriminator(self.out_features)
#         self.sigm = nn.Sigmoid()
#         self.read = AvgReadout()
#
#     def forward(self, feat, feat_a, adj):
#         edge_index, edge_weight = dense_to_sparse(adj)
#         # Main input processing
#         z = F.dropout(feat, self.dropout, self.training)
#         z = self.conv1(z, edge_index)
#         z = self.gat_conv1(z, edge_index, edge_weight)
#         z = self.act(z)
#
#         hiden_emb = z
#
#         h = self.gat_conv2(z, edge_index, edge_weight)
#
#         # Auxiliary input processing
#         z_a = F.dropout(feat_a, self.dropout, self.training)
#         z_a = self.conv1(z_a, edge_index, edge_weight) # CombUnweighted for auxiliary input
#         z_a = self.gat_conv1(z_a, edge_index, edge_weight)
#         z_a = self.act(z_a)
#
#         # Readout graph features
#         g = self.read(z, self.graph_neigh)
#         g = self.sigm(g)
#
#         g_a = self.read(z_a, self.graph_neigh)
#         g_a = self.sigm(g_a)
#
#         # Discriminator
#         ret = self.disc(g, z, z_a)
#         ret_a = self.disc(g_a, z_a, z)
#
#         return hiden_emb, h, ret, ret_a
#
#
# class MGCN2(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.3, act=F.leaky_relu):
#         super(MGCN2, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.act = act
#
#         # 共享的图卷积层
#         self.conv1 = CombUnweighted(K=12)
#         self.gat_conv1 = GCNConv(in_features * (12 + 1), out_features)
#         self.gat_conv2 = GCNConv(out_features, in_features)
#
#         # 共享的判别器和读取层
#         self.disc = Discriminator(out_features)
#         self.sigm = nn.Sigmoid()
#         self.read = AvgReadout()  # 需修改为支持动态邻域
#
#     def forward(self, feat, feat_a, adj, graph_neigh):
#         """新增graph_neigh参数支持不同数据集"""
#         edge_index, edge_weight = dense_to_sparse(adj)
#
#         # 主路径
#         z = F.dropout(feat, self.dropout, self.training)
#         z = self.conv1(z, edge_index)
#         z = self.gat_conv1(z, edge_index, edge_weight)
#         z = self.act(z)
#
#         # 辅助路径
#         z_a = F.dropout(feat_a, self.dropout, self.training)
#         z_a = self.conv1(z_a, edge_index)
#         z_a = self.gat_conv1(z_a, edge_index, edge_weight)
#         z_a = self.act(z_a)
#
#         # 动态读取（关键修改）
#         g = self.read(z, graph_neigh)  # 使用传入的graph_neigh
#         g = self.sigm(g)
#         g_a = self.read(z_a, graph_neigh)
#         g_a = self.sigm(g_a)
#
#         # 重构与判别
#         h = self.gat_conv2(z, edge_index, edge_weight)
#         ret = self.disc(g, z, z_a)
#         ret_a = self.disc(g_a, z_a, z)
#
#         return z, h, ret, ret_a



class Discriminator(nn.Module):
    """
    A neural network-based discriminator, typically used in contrastive learning
    setups to distinguish between positive and negative sample pairs.
    It uses a bilinear layer to compute scores based on global context (c)
    and node-level features (h_pl/h_mi).

    Parameters
    ----------
    n_h : int
        Dimension of the input hidden features (for both context and node features).
    """
    def __init__(self, n_h: int):
        super(Discriminator, self).__init__()
        # Bilinear layer for scoring the compatibility between context and node features
        # f_k(h, c) = h^T W c
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        # Initialize weights
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m: nn.Module):
        """
        Initializes the weights of the bilinear layer.
        Xavier uniform initialization is used for weights, and biases are set to zero.

        Parameters
        ----------
        m : nn.Module
            The module to initialize (expected to be nn.Bilinear).
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c: torch.Tensor, h_pl: torch.Tensor, h_mi: torch.Tensor,
                s_bias1: Optional[torch.Tensor] = None, s_bias2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the discriminator. It computes scores for positive
        and negative pairs given a global context.

        Parameters
        ----------
        c : torch.Tensor
            Global context vector (readout of the graph or part of it).
            Shape: (batch_size, n_h)
        h_pl : torch.Tensor
            Positive sample node features. These are features from the original graph.
            Shape: (num_nodes, n_h)
        h_mi : torch.Tensor
            Negative sample node features. These are features from a corrupted/permuted graph.
            Shape: (num_nodes, n_h)
        s_bias1 : Optional[torch.Tensor], default None
            Optional bias term for positive scores.
        s_bias2 : Optional[torch.Tensor], default None
            Optional bias term for negative scores.

        Returns
        -------
        logits : torch.Tensor
            Concatenated scores for positive and negative pairs.
            Shape: (num_nodes, 2) where column 0 are positive scores and column 1 are negative scores.
        """
        # Expand the context vector to match the shape of node features for element-wise operation
        # c_x will have shape (num_nodes, n_h) by replicating 'c' for each node.
        c_x = c.expand_as(h_pl)

        # Compute scores for positive (h_pl, c) pairs
        sc_1 = self.f_k(h_pl, c_x)
        # Compute scores for negative (h_mi, c) pairs
        sc_2 = self.f_k(h_mi, c_x)

        # Add optional bias terms
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # Concatenate positive and negative scores along dimension 1
        # This forms the logits for a binary classification task (positive vs negative)
        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    """
    Computes a global graph-level embedding by averaging node embeddings.
    It takes a mask (adjacency matrix or neighborhood graph) to define
    which nodes contribute to which global embedding.
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the average readout layer.

        Parameters
        ----------
        emb : torch.Tensor
            Node embeddings. Shape: (num_nodes, embedding_dim)
        mask : Optional[torch.Tensor], default None
            A mask matrix, typically an adjacency matrix or a neighborhood graph.
            If a standard adjacency matrix, it implies a simple sum across neighbors.
            If a batch mask, it aggregates nodes within each graph in a batch.
            Shape: (num_graphs_in_batch, num_nodes) or (num_nodes, num_nodes) for a single graph.

        Returns
        -------
        global_emb : torch.Tensor
            Global graph-level embeddings, normalized to L2 norm.
            Shape: (num_graphs_in_batch, embedding_dim) or (1, embedding_dim) for a single graph.
        """
        # Perform masked summation: vsum[i, k] = sum_j(mask[i, j] * emb[j, k])
        # This effectively sums embeddings of nodes selected by the mask for each "graph" or context.
        vsum = torch.mm(mask, emb)

        # Calculate the sum of rows in the mask to get normalization factors
        # row_sum[i] = sum_j(mask[i, j])
        row_sum = torch.sum(mask, 1)
        # Expand row_sum to match the dimensions of vsum for element-wise division
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T

        # Compute the average: global_emb[i, k] = vsum[i, k] / row_sum[i]
        global_emb = vsum / row_sum

        # L2 normalize the global embeddings
        return F.normalize(global_emb, p=2, dim=1)


def sym_norm(edge_index: torch.Tensor,
             num_nodes: int,
             edge_weight: Optional[Union[Any, torch.Tensor]] = None,
             improved: Optional[bool] = False,
             dtype: Optional[Any] = None
             ) -> List[torch.Tensor]:
    """
    Computes the symmetrically normalized adjacency matrix.
    This normalization method is commonly used in Graph Convolutional Networks (GCNs).
    A_sym = D^(-0.5) * A * D^(-0.5), where A is adjacency and D is degree matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        The edge indices (COO format) of the graph. Shape: (2, num_edges)
    num_nodes : int
        The total number of nodes in the graph.
    edge_weight : Optional[Union[Any, torch.Tensor]], default None
        Edge weights. If None, all edge weights are assumed to be 1.
    improved : Optional[bool], default False
        If set to True, adds self-loops with a value of 2 instead of 1.
        This is a common variant in some GCN implementations.
    dtype : Optional[Any], default None
        Desired data type of the edge weights.

    Returns
    -------
    List[torch.Tensor]
        A list containing:
        - edge_index: The modified edge indices (with self-loops).
        - norm: The computed symmetrically normalized edge weights.
    """
    if edge_weight is None:
        # If no edge weights provided, assume all weights are 1.0
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

    # Ensure edge_index is of type long (required by torch_geometric utils)
    edge_index = edge_index.long()

    # Add self-loops to the adjacency matrix
    # fill_value = 1 for standard GCN, 2 for "improved" GCN
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    # Extract row (source) and column (target) indices from edge_index
    row, col = edge_index
    # Calculate degree for each node: sum of edge_weights connected to each node
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # Calculate D^(-0.5)
    deg_inv_sqrt = deg.pow(-0.5)
    # Handle cases where degree is zero (resulting in infinity), set them to 0
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Compute the normalized edge weights: D^(-0.5)_row * edge_weight * D^(-0.5)_col
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class CombUnweighted(MessagePassing):
    """
    A Graph Convolutional Network (GCN) layer that combines features from
    multiple propagation steps (K-hop neighborhood). This effectively captures
    information from a wider receptive field without stacking many GCN layers.

    Parameters
    ----------
    K : Optional[int], default 1
        The number of propagation steps (hops) to perform. The final output
        concatenates features from 0 to K hops.
    cached : Optional[bool], default False
        If set to True, the normalization constants will be cached.
    bias : Optional[bool], default True
        If set to True, the layer will learn an additive bias.
    **kwargs : Any
        Additional arguments passed to the base MessagePassing class.
    """
    def __init__(self, K: Optional[int] = 1,
                 cached: Optional[bool] = False, # Note: `cached` not used in this specific implementation.
                 bias: Optional[bool] = True, # Note: `bias` not used in this specific implementation, as no linear transformation is applied.
                 **kwargs):
        super(CombUnweighted, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Forward pass for the K-hop propagation layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features. Shape: (num_nodes, num_features)
        edge_index : torch.Tensor
            Graph connectivity in COO format. Shape: (2, num_edges)
        edge_weight : Union[torch.Tensor, None], default None
            Edge weights. If None, all edges are assumed to have weight 1.

        Returns
        -------
        torch.Tensor
            Concatenated node features from 0 to K hops.
            Shape: (num_nodes, num_features * (K + 1))
        """
        # Compute symmetric normalization constants for edges
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
                                    dtype=x.dtype)

        # Store features from each hop
        xs = [x] # xs[0] contains original features (0-hop)
        for k in range(self.K):
            # Propagate features for one step using the current last hop's features
            # This calls `message` and `aggregate`
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        # Concatenate features from all hops (0 to K) along the feature dimension
        return torch.cat(xs, dim=1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages from neighboring nodes `x_j`.
        Each neighbor's feature `x_j` is weighted by the pre-computed `norm`.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of neighboring nodes.
        norm : torch.Tensor
            Symmetrically normalized edge weights for the message passing.

        Returns
        -------
        torch.Tensor
            Weighted messages to be aggregated.
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        """Returns a string representation of the module."""
        return '{}(K={})'.format(self.__class__.__name__, self.K)

# Helper function to convert dense adjacency matrix to sparse edge_index and edge_weight
# This is a common utility for PyTorch Geometric, assumed to be defined elsewhere or implemented here.
def dense_to_sparse(adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a dense adjacency matrix to edge_index and edge_weight suitable for PyTorch Geometric.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        A dense adjacency matrix. Shape: (num_nodes, num_nodes)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - edge_index: Graph connectivity in COO format. Shape: (2, num_edges)
        - edge_weight: Corresponding edge weights. Shape: (num_edges,)
    """
    # Find the indices where the adjacency matrix has non-zero values
    edge_index = adj_matrix.nonzero(as_tuple=True)
    # Get the corresponding edge weights (values)
    edge_weight = adj_matrix[edge_index]
    return torch.stack(edge_index, dim=0), edge_weight


class MGCN(nn.Module):
    """
    Multi-modal Graph Convolutional Network (MGCN) for single-modal graph representation learning.
    It incorporates K-hop propagation and a discriminator for contrastive learning.

    Parameters
    ----------
    in_features : int
        Dimension of input node features.
    out_features : int
        Dimension of the output node embeddings after the first GCN layer.
        Also the dimension of the global context for the Discriminator.
    graph_neigh : torch.Tensor
        The neighborhood graph / mask used by the AvgReadout layer to compute global context.
        This is typically a dense adjacency matrix.
    dropout : float, default 0.3
        Dropout rate applied to input features.
    act : callable, default F.leaky_relu
        Activation function to use after the first GCN layer.
    """
    def __init__(self, in_features: int, out_features: int, graph_neigh: torch.Tensor,
                 dropout: float = 0.3, act=F.leaky_relu):
        super(MGCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh # This is a static mask/neighborhood graph
        self.dropout = dropout
        self.act = act

        # CombUnweighted layer for K-hop feature aggregation (K=12 means 12 hops + original features)
        self.conv1 = CombUnweighted(K=12)
        # First GCNConv layer, takes (in_features * (K+1)) from CombUnweighted
        self.gat_conv1 = GCNConv(in_features * (12 + 1), out_features)
        # Second GCNConv layer for reconstruction/embedding
        self.gat_conv2 = GCNConv(out_features, in_features) # Outputs back to original feature dimension

        # Discriminator for contrastive learning
        self.disc = Discriminator(self.out_features)
        # Sigmoid activation for global context (g)
        self.sigm = nn.Sigmoid()
        # Average Readout layer to get global graph context
        self.read = AvgReadout()

    def forward(self, feat: torch.Tensor, feat_a: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MGCN model.

        Parameters
        ----------
        feat : torch.Tensor
            Original node features. Shape: (num_nodes, in_features)
        feat_a : torch.Tensor
            Augmented (e.g., permuted) node features for negative samples.
            Shape: (num_nodes, in_features)
        adj : torch.Tensor
            Dense adjacency matrix. Shape: (num_nodes, num_nodes)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - hiden_emb: Hidden embeddings after the first GCN layer (z before reconstruction).
                         Shape: (num_nodes, out_features)
            - h: Reconstructed features after the second GCN layer.
                 Shape: (num_nodes, in_features)
            - ret: Discriminator logits for positive pairs (global context vs original features).
                   Shape: (num_nodes, 2)
            - ret_a: Discriminator logits for negative pairs (global context vs augmented features).
                     Shape: (num_nodes, 2)
        """
        # Convert dense adjacency matrix to sparse format for PyTorch Geometric layers
        edge_index, edge_weight = dense_to_sparse(adj)

        # --- Main Input Processing Path (for original features) ---
        z = F.dropout(feat, self.dropout, self.training) # Apply dropout
        z = self.conv1(z, edge_index, edge_weight) # K-hop feature aggregation
        z = self.gat_conv1(z, edge_index, edge_weight) # First GCN layer
        z = self.act(z) # Apply activation function

        hiden_emb = z # Store hidden embedding (output of first GCN)

        # Second GCN layer for reconstruction or further processing
        h = self.gat_conv2(z, edge_index, edge_weight)

        # --- Auxiliary Input Processing Path (for augmented features) ---
        z_a = F.dropout(feat_a, self.dropout, self.training) # Apply dropout
        z_a = self.conv1(z_a, edge_index, edge_weight) # K-hop feature aggregation for augmented data
        z_a = self.gat_conv1(z_a, edge_index, edge_weight) # First GCN layer for augmented data
        z_a = self.act(z_a) # Apply activation function

        # --- Readout Global Graph Features ---
        # Compute global context 'g' from main path embeddings 'z' using the static graph_neigh
        g = self.read(z, self.graph_neigh)
        g = self.sigm(g) # Apply sigmoid activation to global context

        # Compute global context 'g_a' from auxiliary path embeddings 'z_a'
        g_a = self.read(z_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        # --- Discriminator for Contrastive Learning ---
        # Discriminator scores for (global context 'g', original features 'z', augmented features 'z_a')
        ret = self.disc(g, z, z_a)
        # Discriminator scores for (global context 'g_a', augmented features 'z_a', original features 'z')
        ret_a = self.disc(g_a, z_a, z)

        return hiden_emb, h, ret, ret_a


class MGCN2(nn.Module):
    """
    Multi-modal Graph Convolutional Network (MGCN) for dual-modal integration,
    designed to handle varying neighborhood graphs for different datasets.
    It shares GCN layers, discriminator, and readout but allows dynamic `graph_neigh`.

    Parameters
    ----------
    in_features : int
        Dimension of input node features.
    out_features : int
        Dimension of the output node embeddings after the first GCN layer.
        Also the dimension of the global context for the Discriminator.
    dropout : float, default 0.3
        Dropout rate applied to input features.
    act : callable, default F.leaky_relu
        Activation function to use after the first GCN layer.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, act=F.leaky_relu):
        super(MGCN2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        # Shared Graph Convolutional layers
        self.conv1 = CombUnweighted(K=12)
        self.gat_conv1 = GCNConv(in_features * (12 + 1), out_features)
        self.gat_conv2 = GCNConv(out_features, in_features)

        # Shared Discriminator and Readout layer
        self.disc = Discriminator(out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout() # AvgReadout now supports dynamic mask/neighborhood input in forward pass

    def forward(self, feat: torch.Tensor, feat_a: torch.Tensor, adj: torch.Tensor,
                graph_neigh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MGCN2 model, designed for dual-modal integration.

        Parameters
        ----------
        feat : torch.Tensor
            Original node features. Shape: (num_nodes, in_features)
        feat_a : torch.Tensor
            Augmented (e.g., permuted) node features for negative samples.
            Shape: (num_nodes, in_features)
        adj : torch.Tensor
            Dense adjacency matrix. Shape: (num_nodes, num_nodes)
        graph_neigh : torch.Tensor
            The neighborhood graph / mask used by the AvgReadout layer. This parameter
            allows different neighborhood structures for different datasets.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - z: Hidden embeddings after the first GCN layer (main path).
                 Shape: (num_nodes, out_features)
            - h: Reconstructed features after the second GCN layer.
                 Shape: (num_nodes, in_features)
            - ret: Discriminator logits for positive pairs (global context vs original features).
                   Shape: (num_nodes, 2)
            - ret_a: Discriminator logits for negative pairs (global context vs augmented features).
                     Shape: (num_nodes, 2)
        """
        # Convert dense adjacency matrix to sparse format for PyTorch Geometric layers
        edge_index, edge_weight = dense_to_sparse(adj)

        # --- Main Path (Original Features) ---
        z = F.dropout(feat, self.dropout, self.training)
        z = self.conv1(z, edge_index, edge_weight) # Note: `edge_weight` was missing here in original `MGCN2` implementation
        z = self.gat_conv1(z, edge_index, edge_weight)
        z = self.act(z)

        # --- Auxiliary Path (Augmented Features) ---
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = self.conv1(z_a, edge_index, edge_weight) # Note: `edge_weight` was missing here in original `MGCN2` implementation
        z_a = self.gat_conv1(z_a, edge_index, edge_weight)
        z_a = self.act(z_a)

        # --- Dynamic Readout (Key Modification) ---
        # Use the passed `graph_neigh` for readout, enabling different graphs per dataset
        g = self.read(z, graph_neigh)
        g = self.sigm(g)
        g_a = self.read(z_a, graph_neigh)
        g_a = self.sigm(g_a)

        # --- Reconstruction and Discrimination ---
        # Second GCN layer for reconstruction
        h = self.gat_conv2(z, edge_index, edge_weight)
        # Discriminator scores for positive and negative pairs
        ret = self.disc(g, z, z_a)
        ret_a = self.disc(g_a, z_a, z)

        return z, h, ret, ret_a



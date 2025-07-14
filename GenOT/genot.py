import torch
from GenOT.preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, add_contrastive_label, get_feature,get_feature2,permutation, fix_seed, feature_reconstruct_loss
import numpy as np
from .model import MGCN,MGCN2
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scipy.sparse import issparse
from scipy.sparse import csc_matrix, csr_matrix


class Encoder():
    def __init__(self,
                 adata,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=700,
                 pca_n=200,
                 dim_input=3000,# Initial placeholder; the actual input dimension will be determined dynamically based on the data.
                 dim_output=64,
                 random_seed=41,
                 alpha=10,
                 beta=1,
                 theta=0.1,
                 lamda1=10,
                 lamda2=1,
                 datatype='10X'
                 ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning.
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning.
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning.
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning.
            The default is 1.
        datatype : string, optional
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data.
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pca_n = pca_n
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.datatype = datatype

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata, n_components=self.pca_n)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            # standard version
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)



    def train_encoder(self):
        self.model = MGCN(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train ...')
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)

            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)
            # loss = self.alpha * self.loss_feat
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(" finished!")

        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec

            return self.adata





class DualEncoder():
    """Dual encoder model for integrating spatial transcriptomics datasets

    Attributes:
        adata1: AnnData object for first spatial transcriptomics dataset
        adata2: AnnData object for second spatial transcriptomics dataset
        device: Torch device for computation (CPU/GPU)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay coefficient for regularization
        epochs: Number of training epochs
        pca_n: Number of PCA components for feature extraction
        dim_output: Dimension of output embeddings
        random_seed: Random seed for reproducibility
        alpha: Weight coefficient for reconstruction loss
        beta: Weight coefficient for contrastive loss
        theta: Threshold parameter for spatial neighborhood
        datatype: Dataset type identifier ('10X', 'Stereo', etc.)
    """

    def __init__(self,
                 adata1,
                 adata2,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=700,
                 pca_n=200,
                 dim_output=64,
                 random_seed=41,
                 alpha=10,
                 beta=1,
                 theta=0.1,
                 datatype='10X'):
        """Initialize dual encoder with datasets and parameters"""

        # Store input datasets and parameters
        self.adata1 = adata1
        self.adata2 = adata2
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pca_n = pca_n
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.datatype = datatype

        # Ensure reproducibility
        self._fix_seed()

        # Preprocess datasets (feature extraction, neighborhood construction)
        self._prepare_datasets()

        # Initialize MGCN model and optimizer
        self.dim_input = self.adata1.obsm['feat'].shape[1]  # Feature dimension
        self.model = MGCN2(self.dim_input, dim_output).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        self.loss_CSL = nn.BCEWithLogitsLoss()  # Contrastive loss function

    def _fix_seed(self):
        """Fix random seeds for reproducibility"""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def _prepare_datasets(self):
        """Preprocess both datasets with standardized pipeline"""
        combined_X = np.vstack([
            self.adata1.X.toarray() if issparse(self.adata1.X) else self.adata1.X,
            self.adata2.X.toarray() if issparse(self.adata2.X) else self.adata2.X
        ])

        Dualpca = PCA(n_components=self.pca_n)
        Dualpca.fit(combined_X)
        self.dual_pca_model = Dualpca
        for i, adata in enumerate([self.adata1, self.adata2]):
            # Perform basic preprocessing if not done
            if 'highly_variable' not in adata.var.keys():
                preprocess(adata)

            # Construct spatial neighborhood graph
            if 'adj' not in adata.obsm.keys():
                if self.datatype in ['Stereo', 'Slide']:
                    construct_interaction_KNN(adata)
                else:
                    construct_interaction(adata)

            # Add contrastive learning labels
            if 'label_CSL' not in adata.obsm.keys():
                add_contrastive_label(adata)

            # Extract features using PCA
            if 'feat' not in adata.obsm.keys():
                get_feature2(adata, pca_model=Dualpca)

            # Convert numpy arrays to PyTorch tensors
            self._convert_to_tensor(adata, f'dataset{i + 1}')

    def _convert_to_tensor(self, adata, dataset_name):
        """Convert AnnData features to PyTorch tensors

        Args:
            adata: AnnData object containing features
            dataset_name: Identifier for dataset ('dataset1' or 'dataset2')
        """
        # Convert features and augmented features
        setattr(self, f'features_{dataset_name}',
                torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device))
        setattr(self, f'features_a_{dataset_name}',
                torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device))

        # Convert contrastive labels
        setattr(self, f'label_CSL_{dataset_name}',
                torch.FloatTensor(adata.obsm['label_CSL']).to(self.device))

        # Process adjacency matrix based on data type
        adj = adata.obsm['adj']
        if self.datatype in ['Stereo', 'Slide']:
            adj = preprocess_adj_sparse(adj).to(self.device)
        else:
            adj = torch.FloatTensor(preprocess_adj(adj)).to(self.device)
        setattr(self, f'adj_{dataset_name}', adj)

        # Convert neighborhood graph with self-connections
        graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'] + np.eye(adj.shape[0])).to(self.device)
        setattr(self, f'graph_neigh_{dataset_name}', graph_neigh)

    def _compute_loss(self, features, features_a, adj, graph_neigh, label_CSL):
        """Calculate combined loss for model update

        Args:
            features: Original features tensor
            features_a: Augmented features tensor
            adj: Adjacency matrix tensor
            graph_neigh: Neighborhood graph tensor
            label_CSL: Contrastive learning labels

        Returns:
            Combined loss value (reconstruction + contrastive)
        """
        # Forward pass through model
        hiden_feat, emb, ret, ret_a = self.model(features, features_a, adj, graph_neigh)

        # Reconstruction loss (MSE between input and reconstructed features)
        loss_feat = F.mse_loss(features, emb)

        # Contrastive loss (Binary cross-entropy for both original and augmented features)
        loss_sl = (self.loss_CSL(ret, label_CSL) +
                   self.loss_CSL(ret_a, label_CSL))

        # Weighted combination of losses
        return self.alpha * loss_feat + self.beta * loss_sl

    def train_encoder(self):
        """Main training loop for joint dataset integration"""
        print(f"Training DualEncoder on 2 datasets with {self.epochs} epochs...")
        self.model.train()

        # Training progress bar
        for epoch in tqdm(range(self.epochs), desc="Training"):
            total_loss = 0

            # Process dataset 1 with random feature permutation
            features_a1 = permutation(self.features_dataset1)
            loss1 = self._compute_loss(self.features_dataset1, features_a1,
                                       self.adj_dataset1, self.graph_neigh_dataset1,
                                       self.label_CSL_dataset1)

            # Process dataset 2 with random feature permutation
            features_a2 = permutation(self.features_dataset2)
            loss2 = self._compute_loss(self.features_dataset2, features_a2,
                                       self.adj_dataset2, self.graph_neigh_dataset2,
                                       self.label_CSL_dataset2)

            # Combine losses and update model
            total_loss = loss1 + loss2
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Save final embeddings to AnnData objects
        self._save_embeddings()
        print("Training completed!")
        return self.adata1, self.adata2

    def _save_embeddings(self):
        """Save learned embeddings back to AnnData objects"""
        with torch.no_grad():
            self.model.eval()

            # Generate embeddings for dataset 1
            _, emb1, _, _ = self.model(self.features_dataset1, self.features_a_dataset1,
                                       self.adj_dataset1, self.graph_neigh_dataset1)
            self.adata1.obsm['emb'] = self._postprocess_embedding(emb1)

            # Generate embeddings for dataset 2
            _, emb2, _, _ = self.model(self.features_dataset2, self.features_a_dataset2,
                                       self.adj_dataset2, self.graph_neigh_dataset2)
            self.adata2.obsm['emb'] = self._postprocess_embedding(emb2)

    def _postprocess_embedding(self, embedding):
        """Post-process embeddings based on data type

        Args:
            embedding: Raw embedding tensor from model

        Returns:
            Processed embedding numpy array
        """
        if self.datatype in ['Stereo', 'Slide']:
            # L2 normalization for specific data types
            return F.normalize(embedding, p=2, dim=1).cpu().numpy()
        return embedding.cpu().numpy()


class Decoder(nn.Module):
    r"""
    Advanced Data Reconstruction Network

    Parameters
    ----------
    input_size : int
        Input dimension.
    output_size : int
        Output dimension (feature dimension of input data).
    hidden_size1 : int
        Size of the first hidden layer.
    hidden_size2 : int, optional
        Size of the second hidden layer (default is 256).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.5).
    activation : str, optional
        Activation function to use ('relu', 'tanh', 'leaky_relu', etc.) (default is 'relu').
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size1: int = 512,
            hidden_size2: int = 256,
            dropout_rate: float = 0.5,
            activation: str = 'relu',
    ):
        super(Decoder, self).__init__()

        # Define layers
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, output_size)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'leaky_relu': F.leaky_relu,
            'sigmoid': torch.sigmoid,
            'softplus': F.softplus,
        }
        self.activation = activations.get(activation, F.relu)

    def forward(self, input_embd: torch.Tensor):
        # Forward pass with dropout and batch normalization
        x = self.hidden1(input_embd)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output(x)
        return x

    def train_decoder(
            self,
            adata1,
            adata2,
            decoder,
            epochs=100,
            batch_size=32,
            learning_rate=1e-2,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        """
        Train a shared decoder using embeddings from two adata objects.

        Parameters
        ----------
        adata1 : AnnData
            The first AnnData object with embeddings in `obsm['emb']`.
        adata2 : AnnData
            The second AnnData object with embeddings in `obsm['emb']`.
        decoder : nn.Module
            The shared decoder to train.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        learning_rate : float
            Learning rate for the optimizer.
        device : str
            Device to use for training ('cuda' or 'cpu').

        Returns
        -------
        decoder : nn.Module
            The trained decoder.
        """

        # Extract embeddings and original features
        embedding1 = torch.tensor(adata1.obsm["emb"], dtype=torch.float32)
        embedding2 = torch.tensor(adata2.obsm["emb"], dtype=torch.float32)
        raw_features1 = torch.tensor(adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X, dtype=torch.float32)
        raw_features2 = torch.tensor(adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X, dtype=torch.float32)

        # Create datasets and dataloaders
        dataset1 = TensorDataset(embedding1, raw_features1)
        dataset2 = TensorDataset(embedding2, raw_features2)
        loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
        loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

        # Move decoder to device
        decoder = decoder.to(device)

        # Define optimizer and loss function
        optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in tqdm(range(epochs)):
            decoder.train()
            total_loss = 0.0

            # Train on both datasets
            for loader in [loader1, loader2]:
                for embeddings, raw_features in loader:
                    embeddings, raw_features = embeddings.to(device), raw_features.to(device)

                    # Forward pass
                    reconstructed_features = decoder(embeddings)

                    # Compute loss
                    loss = criterion(reconstructed_features, raw_features)
                    total_loss += loss.item()

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


        return decoder


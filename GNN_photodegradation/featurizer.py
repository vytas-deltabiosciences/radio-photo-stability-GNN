import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, Batch
from GNN_photodegradation.data_utils import smiles_to_graph
from GNN_photodegradation.get_logger import get_logger

logger = get_logger()

class Create_Dataset(Dataset):
    def __init__(self, dataframe, numerical_features, scaler=None, inference=False):
        print("✅ Using featurizer.py")
        self.dataframe = dataframe.reset_index(drop=True)
        self.smiles = self.dataframe['SMILES']

        # Extract numerical features
        self.experimental_feats = self.dataframe[numerical_features].astype(np.float32)
        
        # Targets (only during training)
        self.targets = None
        if not inference and 'mlogk' in self.dataframe.columns:
            self.targets = self.dataframe['mlogk'].values.astype(np.float32)

        # Standardize with feature names
        if scaler is None:
            self.scaler = StandardScaler()
            self.experimental_feats = self.scaler.fit_transform(self.experimental_feats)
        else:
            self.scaler = scaler
            self.experimental_feats = self.scaler.transform(self.experimental_feats)

        # Convert to torch tensors
        self.experimental_feats = torch.tensor(self.experimental_feats, dtype=torch.float)
        if self.targets is not None:
            self.targets = torch.tensor(self.targets, dtype=torch.float).unsqueeze(1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        graph = smiles_to_graph(smiles, logger)
        if graph is None:
            graph = Data(
                x=torch.zeros((1, 22)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 6))
            )

        experimental_feat = self.experimental_feats[idx]
        target = self.targets[idx] if self.targets is not None else torch.tensor([0.0], dtype=torch.float)

        return graph, experimental_feat, target

# ----------------------------- Collate Function ----------------------------- #
def collate_fn(batch):
    graphs, experimental_feats, targets = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    experimental_feats = torch.stack(experimental_feats)
    targets = torch.stack(targets)
    return batch_graph, experimental_feats, targets

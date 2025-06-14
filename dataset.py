import torch
from torch_geometric.datasets import Planetoid

class PyGDatasetLoader:
    """
    Lightweight wrapper to load PyG datasets (e.g., Planetoid) with optional verbose logging.
    """
    def __init__(self, root='/tmp/Cora', name='Cora', verbose=True):
        """
        Args:
            root (str): Root directory for dataset storage.
            name (str): Dataset name, e.g., 'Cora', 'Citeseer', 'Pubmed'.
            verbose (bool): Whether to print dataset information.
        """
        self.root = root
        self.name = name
        self.verbose = verbose
        self.dataset = None
        self.data = None

    def load(self):
        """Loads the dataset and prints information if verbose is enabled."""
        self.dataset = Planetoid(root=self.root, name=self.name)
        self.data = self.dataset[0]

        if self.verbose:
            self._print_info()

        return self.dataset, self.data

    def _print_info(self):
        """Prints basic dataset and graph statistics."""
        print(f"\nDataset: {self.name}")
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of classes: {self.dataset.num_classes}")
        print("\nGraph Info:")
        print(f"  Nodes: {self.data.num_nodes}")
        print(f"  Features per node: {self.data.num_node_features}")
        print(f"  Edges: {self.data.num_edges}")
        print(f"  Train nodes: {self.data.train_mask.sum().item()}")
        print(f"  Val nodes: {self.data.val_mask.sum().item()}")
        print(f"  Test nodes: {self.data.test_mask.sum().item()}")


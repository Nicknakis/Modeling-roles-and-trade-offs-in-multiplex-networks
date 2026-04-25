import numpy as np
import torch
from sklearn.decomposition import NMF


# Module-level N is set by main.py before calculate_nmf_for_layer is invoked,
# preserving the original global-reference behavior of the function body.
N = None


def calculate_nmf_for_layer(layer, n_components):
    """
    Calculate NMF for the adjacency matrix of a single graph layer.
    
    Parameters:
        layer (nx.DiGraph): A single graph layer.
        n_components (int): Number of components for the NMF decomposition.
        
    Returns:
        tuple: (W, H), the factor matrices from the NMF decomposition.
    """
    # Convert graph to adjacency matrix
    adjacency_matrix = np.zeros((N,N))
    i=np.array(list(layer.edges()))[:,0]
    j=np.array(list(layer.edges()))[:,1]
    adjacency_matrix[i,j]=1
    # Perform NMF
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500, tol=1e-4)
    W_ = torch.from_numpy(model.fit_transform(adjacency_matrix)).float()
    H_ = torch.from_numpy(model.components_.T).float()
    

    
    

    
    return W_+1e-04, H_+1e-04

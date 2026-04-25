import numpy as np
import scipy.sparse as sparse


def multiplex_sbm_edges_np(
    sizes,
    layer_block_mats,
    *,
    directed=True,
    self_loops=False,
    interlayer_coupling=True,
    interlayer_weight=1.0,
    seed=None
):
    """
    Generate a directed multiplex SBM with L layers, returning NumPy edge arrays.

    Returns
    -------
    {
      "edges_np": [E0, E1, ..., EL-1],   # each a numpy array of shape (E, 2)
      "partition": np.ndarray,           # node-to-community array
      "interlayer_edges": list,          # [(u,ell),(u,m)] tuples
      "supraadj": scipy.sparse.csr_matrix
    }
    """
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed

    K = len(sizes)
    N = sum(sizes)
    L = len(layer_block_mats)

    # checks
    for B in layer_block_mats:
        if B.shape != (K, K):
            raise ValueError("Each block matrix must be K×K.")
        if (B < 0).any() or (B > 1).any():
            raise ValueError("Probabilities must be within [0, 1].")

    # community assignment
    partition = np.repeat(np.arange(K), sizes)
    comm_nodes = []
    start = 0
    for s in sizes:
        comm_nodes.append(np.arange(start, start + s))
        start += s

    # sample each layer
    edges_np = []
    for B in layer_block_mats:
        edges = []
        for a in range(K):
            Ia = comm_nodes[a]
            na = len(Ia)
            for b in range(K):
                Ib = comm_nodes[b]
                nb = len(Ib)
                p = B[a, b]
                if p <= 0:
                    continue

                if directed:
                    mask = rng.random((na, nb)) < p
                    if not self_loops and a == b:
                        np.fill_diagonal(mask, False)
                    src, dst = np.where(mask)
                    if len(src):
                        edges.append(np.column_stack((Ia[src], Ib[dst])))
                else:
                    if a == b:
                        tri = rng.random((na, na))
                        mask = np.triu(tri, k=1 if not self_loops else 0) < p
                        r, c = np.where(mask)
                        if len(r):
                            pairs = np.column_stack((Ia[r], Ia[c]))
                            edges.append(np.vstack((pairs, pairs[:, [1, 0]])))
                    else:
                        mask = rng.random((na, nb)) < p
                        r, c = np.where(mask)
                        if len(r):
                            pairs = np.column_stack((Ia[r], Ib[c]))
                            edges.append(np.vstack((pairs, pairs[:, [1, 0]])))
        edges_np.append(np.vstack(edges) if edges else np.zeros((0, 2), dtype=int))

    # build supra adjacency
    LN = L * N
    supra_blocks = [[sparse.csr_matrix((N, N)) for _ in range(L)] for _ in range(L)]
    for ell, arr in enumerate(edges_np):
        if arr.size:
            data = np.ones(len(arr))
            supra_blocks[ell][ell] = sparse.csr_matrix((data, (arr[:, 0], arr[:, 1])), shape=(N, N))

    interlayer_edges = []
    if interlayer_coupling and L > 1:
        for ell in range(L):
            for m in range(L):
                if ell == m:
                    continue
                rows = np.arange(N)
                cols = np.arange(N)
                data = np.full(N, interlayer_weight)
                supra_blocks[ell][m] = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        for u in range(N):
            for ell in range(L):
                for m in range(ell + 1, L):
                    interlayer_edges.append(((u, ell), (u, m)))

    supra = sparse.bmat(supra_blocks, format="csr")

    return {
        "edges_np": edges_np,
        "partition": partition,
        "interlayer_edges": interlayer_edges,
        "supraadj": supra,
    }

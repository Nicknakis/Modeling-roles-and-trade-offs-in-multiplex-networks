import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import timeit
import scipy.sparse as sparse
import math
import scipy
from sklearn.manifold import MDS
import torch.nn.functional as F
from sklearn.decomposition import NMF





start = timeit.default_timer()
CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:        
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

print(device)
undirected=1




class Spectral_clustering_init():
    def __init__(self,num_of_eig=10,method='Adjacency',device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        self.num_of_eig=num_of_eig
        self.method=method
        self.device=device

    
    def spectral_clustering(self):
        
        sparse_i=self.sparse_i_idx.cpu().numpy()
        sparse_j=self.sparse_j_idx.cpu().numpy()
        idx_shape=sparse_i.shape[0]
        if (sparse_i<sparse_j).sum()==idx_shape:
            sparse_i_new=np.concatenate((sparse_i,sparse_j))
            sparse_j_new=np.concatenate((sparse_j,sparse_i))
            
            sparse_i=sparse_i_new
            sparse_j=sparse_j_new
            
        V=np.ones(sparse_i.shape[0])
   
        Affinity_matrix=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.input_size,self.input_size))
        if self.missing_data:
            sparse_i_rem=self.sparse_i_idx_removed.cpu().numpy()
            sparse_j_rem=self.sparse_j_idx_removed.cpu().numpy()
            idx_shape_rem=sparse_i_rem.shape[0]
            if (sparse_i_rem<sparse_j_rem).sum()==idx_shape_rem:
                sparse_i_new=np.concatenate((sparse_i_rem,sparse_j_rem))
                sparse_j_new=np.concatenate((sparse_j_rem,sparse_i_rem))
                
            sparse_i_rem=sparse_i_new
            sparse_j_rem=sparse_j_new
            
            V=np.ones(sparse_i_rem.shape[0])
            temp_links=sparse.coo_matrix((V,(sparse_i_rem,sparse_j_rem)),shape=(self.input_size,self.input_size))
            Affinity_matrix=Affinity_matrix-temp_links
       
        
       
        
        if self.method=='Adjacency':
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(Affinity_matrix,self.num_of_eig,which='LM')
            X = eig_vect.real
            rows_norm = np.linalg.norm(X, axis=1, ord=2)
            U_norm = (X.T / rows_norm).T
             
             

            
        elif self.method=='Normalized_sym':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
            tem=DH @ (L @ DH)
            eig_val, eig_vect = scipy.sparse.linalg.eigs(tem,self.num_of_eig,which='SR')
            X = eig_vect.real
            self.X=X
            rows_norm = np.linalg.norm(X, axis=1,ord=2)
            U_norm =(X.T / rows_norm).T
            
                
                
        elif self.method=='Normalized':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_inv = 1.0 /diags
            diags_inv[np.isinf(diags_inv)] = 0
            DH = sparse.spdiags(diags_inv, [0], m, n, format="csr")
            tem=DH @L
            eig_val, eig_vect = scipy.sparse.linalg.eigs(tem,self.num_of_eig,which='SR')
    
            X = eig_vect.real
            self.X=X
            U_norm =X
            
        elif self.method=='MDS':
            n, m = Affinity_matrix.shape

            G= nx.Graph(Affinity_matrix)

            max_l=0
            N = G.number_of_nodes()
            pmat = np.zeros((N, N))+np.inf
            paths = nx.all_pairs_shortest_path_length(G)
            for node_i, node_ij in paths:
                for node_j, length_ij in node_ij.items():
                    pmat[node_i, node_j] = length_ij
                    if length_ij>max_l:
                        max_l=length_ij

            pmat[pmat == np.inf] = max_l+1
            print('shortest path done')
            
            embedding = MDS(n_components=self.num_of_eig,dissimilarity='precomputed')   
            U_norm = embedding.fit_transform(pmat)
            
        else:
            print('Invalid Spectral Clustering Method')


        
        return torch.from_numpy(U_norm).float().to(self.device)
            
        






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


#H,W=calculate_nmf_for_layer(pruned_layers[1],latent_dim)





start = timeit.default_timer()
CUDA = torch.cuda.is_available()



undirected=1

  


class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,weights,tree_height,Ws,Hs,sparse_is,sparse_js,sparse_i,sparse_j, input_size1,input_size2,num_layers):
        super(LSM, self).__init__()
        # initialization
        self.input_size1=N1
        self.input_size2=N2
        self.tree_height=tree_height


        
        
        self.p_k_1=nn.Parameter(torch.log(torch.ones(self.tree_height+1,device=device)/(self.tree_height+1)))
        self.p_k_2=nn.Parameter(torch.log(torch.ones(self.tree_height+1,device=device)/(self.tree_height+1)))
        self.p_k_3=nn.Parameter(torch.log(torch.ones(self.tree_height+1,device=device)/(self.tree_height+1)))

        
        self.soft_0=nn.Softmax(0)

        self.weights=weights
        
        
        self.full_k=False

       
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.softmax_loc=nn.Softmax(dim=-1)
        self.us=nn.ParameterList([nn.Parameter(torch.rand(N1, 2**(self.tree_height),device=device))  for i in range(3)])
        self.vs=nn.ParameterList([nn.Parameter(torch.rand(N2, 2**(self.tree_height),device=device))  for i in range(3)])

        
        
        
        
        
       
        self.intra_dim=intra_dim

        self.gamma=nn.Parameter(torch.randn(input_size1,3,device=device))
        self.delta=nn.Parameter(torch.randn(input_size2,3,device=device))

        
        self.scaling=1

        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.Softmax=nn.Softmax(1)

        self.sparse_is=sparse_is
        self.sparse_js=sparse_js
        

            
        self.softplus=nn.Softplus()
       
        
        
        self.torch_pi=torch.tensor(math.pi)
        

        self.L_=nn.Parameter(5*torch.ones(1))
        self.g_l=nn.ParameterList([nn.Parameter(torch.rand(1,device=device)) for i in range(3)])
        self.d_l=nn.ParameterList([nn.Parameter(torch.rand(1,device=device)) for i in range(3)])

       # self.latent_z1=nn.Parameter(torch.log((init_z+0.1)/1.3))
       # self.latent_w1=nn.Parameter(torch.log((init_w+0.1)/1.3))


        self.latent_z1=nn.Parameter(torch.rand(input_size1,3,device=device))

        self.latent_w1=nn.Parameter(torch.rand(input_size2,3,device=device))


        
    
    
   
    
    
    

    
    
    
    def LSM_likelihood_bias(self,epoch):
        '''
        Bernoulli log-likelihood ignoring the log(k!) constant
        
        '''
        self.epoch=epoch
      
       
        self.latent_z=self.Softmax(self.latent_z1)
        self.latent_w=self.Softmax(self.latent_w1)
        
       
      
        if self.scaling:
            
            for layer in range(len(self.sparse_is)):
                

               
                self.g=self.gamma[:,layer]
                self.d=self.delta[:,layer]

                sparse_i_=self.sparse_is[layer].long()
                sparse_j_=self.sparse_js[layer].long()
                
                
                mat=self.softplus(self.g.unsqueeze(-1)+self.d)
                
                if self.input_size1==self.input_size2:
                    mat=(mat-torch.diag(torch.diagonal(mat)))


                
                z_pdist1=mat.sum()#(mat-torch.diag(torch.diagonal(mat))).sum()

                
                z_pdist2=((self.g[sparse_i_]+self.d[sparse_j_])).sum()

                
    
                if layer==0:
                    log_likelihood_sparse=z_pdist2-z_pdist1
                else:
                    log_likelihood_sparse=log_likelihood_sparse+z_pdist2-z_pdist1
           
            

    
            if self.epoch==2000:
                self.scaling=0
                
            return log_likelihood_sparse

        else:
            

            

            self.L=self.softplus(self.L_)#
            self.z_mem_l=[]
            self.w_mem_l=[]
            
            
            
            

            for layer in range(len(self.sparse_is)):
                
               
                
                if layer==0:
                    self.p_k=self.soft_0(self.p_k_1)
                    
                if layer==1:
                    self.p_k=self.soft_0(self.p_k_2)
                if layer==2:
                    self.p_k=self.soft_0(self.p_k_3)
                
                w_l=self.weights[layer]

                self.g=self.gamma[:,layer]
                self.d=self.delta[:,layer]
                
                
                
                self.emb_z=self.Softmax(self.us[layer])
                self.emb_w=self.Softmax(self.vs[layer])
                self.z_mem=[]
                self.w_mem=[]
                

                interdependence=self.p_k[0]*torch.ones(self.input_size1,self.input_size2)
                
                
                for k in range(1,(self.tree_height)):
                    
                    if k==1:
                        z_k = self.emb_z.view(self.input_size1, -1, 2).sum(dim=2)
                        w_k = self.emb_w.view(self.input_size2, -1, 2).sum(dim=2)
                        self.z_mem.append(z_k)
                        self.w_mem.append(w_k)
                    
                  
                        
                    else:
                        z_k = self.z_mem[-1].view(self.input_size1, -1, 2).sum(dim=2)
                        w_k = self.w_mem[-1].view(self.input_size2, -1, 2).sum(dim=2)
                        self.z_mem.append(z_k)
                        self.w_mem.append(w_k)
                        
                    
                    
                    interdependence=interdependence+ self.p_k[k]*((z_k).unsqueeze(1)*(w_k+1e-06)).sum(-1)
                interdependence=interdependence+ self.p_k[-1]*((self.emb_z).unsqueeze(1)*(self.emb_w+1e-06)).sum(-1)
                

                    

                
                mat_1=self.g.unsqueeze(-1)+self.d

             
                mat_0=((self.latent_z[:,layer].unsqueeze(1))*self.L*(self.latent_w[:,layer]+1e-06))*interdependence
                
                    
                
                sparse_i_=self.sparse_is[layer].long()
                sparse_j_=self.sparse_js[layer].long()   
                
              
                mat=self.softplus(mat_0+mat_1)
                if self.input_size1==self.input_size2:
                    mat=(mat-torch.diag(torch.diagonal(mat)))
                

                
                z_pdist1=mat.sum()
                
                z_pdist2=((mat_0[sparse_i_,sparse_j_]+mat_1[sparse_i_,sparse_j_])).sum()
            
               
                if layer==0:
                    log_likelihood_sparse=z_pdist2-z_pdist1

                else:
                    log_likelihood_sparse=log_likelihood_sparse+z_pdist2-z_pdist1

        
            return log_likelihood_sparse
        
    def negative_log_prior(self):
        num_dyads=(self.input_size1*(self.input_size2-1))  # global prior (no normalization)
        sigma = F.softplus(self.log_sigma) + 1e-6
        prior_delta = (self.L ** 2) / (2 * sigma ** 2) + torch.log(sigma)
        prior_sigma = torch.log(1 + (sigma ** 2) / (self.gamma_cauchy ** 2))
        # CRITICAL CHANGE: Normalize explicitly per dyad
        return (prior_delta + prior_sigma / num_dyads)
    
    
    


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






parser = argparse.ArgumentParser(description='MLT Model')


args = parser.parse_args()

if __name__ == "__main__":
    
    # GENERATE RANDOM NETWORK
    sizes = [30, 40, 50]
    B1 = np.random.rand(3, 3) * 0.05  # for example
    B2 = np.random.rand(3, 3) * 0.05
    B3 = np.random.rand(3, 3) * 0.05


    out = multiplex_sbm_edges_np(sizes, [B1, B2, B3], directed=True, seed=123)


        
    edges_i_soc_all=out['edges_np'][0][:,0]
    edges_j_soc_all=out['edges_np'][0][:,1]
        
    edges_i_health_all=out['edges_np'][1][:,0]
    edges_j_health_all=out['edges_np'][1][:,1]

    edges_i_econ_all=out['edges_np'][2][:,0]
    edges_j_econ_all=out['edges_np'][2][:,1]
   


    



        

   

    #for vil_id in range(len(edges_i_econ_all)):
    for vil_id in range(1):
        

        final_i_soc=torch.from_numpy(edges_i_soc_all).long()
        final_j_soc=torch.from_numpy(edges_j_soc_all).long()
        w_soc=np.ones(final_i_soc.shape[0])

        final_i_heal=torch.from_numpy(edges_i_health_all).long()
        final_j_heal=torch.from_numpy(edges_j_health_all).long()
        w_heal=np.ones(final_i_heal.shape[0])

        final_i_econ=torch.from_numpy(edges_i_econ_all).long()
        final_j_econ=torch.from_numpy(edges_j_econ_all).long()
        w_econ=np.ones(final_i_econ.shape[0])
        
        weights=[torch.from_numpy(w_soc),torch.from_numpy(w_heal),torch.from_numpy(w_econ)]


       

        sparse_is=[final_i_soc,final_i_heal,final_i_econ]
        sparse_js=[final_j_soc,final_j_heal,final_j_econ]

        sparse_i=torch.cat(sparse_is)
        sparse_j=torch.cat(sparse_js)

        N=int(max(sparse_i.max(),sparse_j.max())+1)
        
        
        
        print(N)
        


    
        
        
        N1=int(sparse_i.max()+1)
        N2=int(sparse_j.max()+1)
        
        
        
        



            
        
      
        sparse_is=[final_i_soc,final_i_heal,final_i_econ]
        sparse_js=[final_j_soc,final_j_heal,final_j_econ]

        sparse_i=torch.cat(sparse_is)
        sparse_j=torch.cat(sparse_js)
        
        
        A1=torch.zeros(N,N)
        A2=torch.zeros(N,N)
        A3=torch.zeros(N,N)

        A1[final_i_soc,final_j_soc]=1
        A2[final_i_heal,final_j_heal]=1
        A3[final_i_econ,final_j_econ]=1

        A=A1+A2+A3
        plt.spy(A.cpu().numpy())
        plt.show()


    
    
        layer1 = nx.DiGraph()
        layer1.add_edges_from(zip(final_i_soc.cpu().numpy(), final_j_soc.cpu().numpy()))

        layer2 = nx.DiGraph()
        layer2.add_edges_from(zip(final_i_heal.cpu().numpy(), final_j_heal.cpu().numpy()))

        layer3 = nx.DiGraph()
        layer3.add_edges_from(zip(final_i_econ.cpu().numpy(), final_j_econ.cpu().numpy()))


        layers = [layer1, layer2, layer3]

        N=int(max(sparse_i.max(),sparse_j.max())+1)
       
        
        # total number of RUNS
        runs=1
        
        # number of runs to avoid bad local minimas
        inner_runs=5
        
        soc_z_dim_runs=[]
        heal_z_dim_runs=[]
        econ_z_dim_runs=[]

        model_pks=[]
        soc_w_dim_runs=[]
        heal_w_dim_runs=[]
        econ_w_dim_runs=[]
        
        for run in range(runs):
            min_loss=1e06
            min_loss_RE=1e06

            for inner_run in range(inner_runs):
                print(f'RUN: {run}')



                print("RUN number:",run)
                epoch_num=15001
                intra_dim=1
                num_layers=3

                in_deg=torch.cat((A1.sum(1).reshape(-1,1)*torch.ones(N1,intra_dim),A2.sum(1).reshape(-1,1)*torch.ones(N1,intra_dim),A3.sum(1).reshape(-1,1)*torch.ones(N1,intra_dim)),1)
                out_deg=torch.cat((A1.sum(0).reshape(-1,1)*torch.ones(N2,intra_dim),A2.sum(0).reshape(-1,1)*torch.ones(N2,intra_dim),A3.sum(0).reshape(-1,1)*torch.ones(N2,intra_dim)),1)
                
                
                i=torch.where(in_deg.sum(1)==0)[0]
                if i.shape[0]>0:
                    in_deg[i]=in_deg[i]+1


                i=torch.where(out_deg.sum(1)==0)[0]
                if i.shape[0]>0:
                    out_deg[i]=out_deg[i]+1

                
                
                init_z=in_deg/in_deg.sum(-1).view(-1,1)
                init_w=out_deg/out_deg.sum(-1).view(-1,1)

                W_soc, H_soc=calculate_nmf_for_layer(layers[0],n_components=num_layers)
                W_heal, H_heal=calculate_nmf_for_layer(layers[1],n_components=num_layers)
                W_econ, H_econ=calculate_nmf_for_layer(layers[2],n_components=num_layers)
                Ws=[W_soc,W_heal,W_econ]
                Hs=[H_soc,H_heal,H_econ]
                
                
                # Height of the tree
                tree_height=round(np.log(N1))
                # Missing_data should be set to False for link_prediction since we do not consider these interactions as missing but as zeros.
                model1 = LSM(weights,tree_height,Ws,Hs,sparse_is,sparse_js,sparse_i,sparse_j,N1,N2,num_layers=num_layers).to(device)         




                optimizer1 = optim.AdamW(model1.parameters(), 0.1)  
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer1,mode='min', factor=0.5, patience=10, verbose=True)

               
                    
                colors=np.array(["green","blue","red"])
                losses=[]
                #sampling=True    
                for epoch in range(epoch_num):


                    if model1.scaling:


                        loss1=-model1.LSM_likelihood_bias(epoch=epoch)/(N1*(N1-1))


                    else:   



                        loss1=-model1.LSM_likelihood_bias(epoch=epoch)/(N1*(N1-1))




                    #model_pks.append(1*model1.p_k.detach())

                    optimizer1.zero_grad() # clear the gradients.   
                    loss1.backward() # backpropagate
                    optimizer1.step() # update the weights
                    
                    if epoch==1999:
                        
                        if loss1.item()<min_loss_RE:
                            min_loss_RE=1*loss1.item()

                           
                            torch.save(model1.state_dict(), './Bias_model.pth')

                    
                    current_lr = optimizer1.param_groups[0]['lr']
                    #print(f"Epoch {epoch}: Current LR = {current_lr:.5f}")
                    min_lr_threshold=1e-07
                    # Early stop if learning rate is below threshold
                    if current_lr < min_lr_threshold:
                        print(f"Stopping early at epoch {epoch} as learning rate dropped below {min_lr_threshold}")
                        break
                    if not model1.scaling:
                        scheduler.step(loss1)
                    losses.append(loss1.item())

                    



                
                save=True
                if save:
                    
                    if loss1.item()<min_loss:
                        min_loss=1*loss1.item()
                        print('INNER RUN', inner_run)
                      
                        torch.save(model1.state_dict(), './Full_model.pth')

                

        

            
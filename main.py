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

from src import model as _model_mod
from src import nmf_utils as _nmf_mod
from src.model import LSM
from src.nmf_utils import calculate_nmf_for_layer
from src.data_generator import multiplex_sbm_edges_np


start = timeit.default_timer()
CUDA = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if CUDA:        
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


CUDA = torch.cuda.is_available()


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

        # Inject module-level globals consumed inside src.nmf_utils and src.model,
        # mirroring how N / N1 / N2 were referenced as globals in the original script.
        _nmf_mod.N = N
        _model_mod.N1 = N1
        _model_mod.N2 = N2
       
        
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

                # Keep src.model.intra_dim in sync with the local intra_dim, as
                # LSM.__init__ reads it as a module-level global in the original code.
                _model_mod.intra_dim = intra_dim

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

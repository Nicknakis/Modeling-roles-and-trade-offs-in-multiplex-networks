import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral import Spectral_clustering_init


# Mirror the original module-level computation of CUDA/device so the
# references inside the class body resolve identically to the source script.
CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# These are populated by main.py before LSM is instantiated, mirroring the
# original script where N1, N2, and intra_dim were module-level globals
# referenced from inside LSM.__init__.
N1 = None
N2 = None
intra_dim = None


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
        Bernoulli log-likelihood
        
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


                
                z_pdist1=mat.sum()

                
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
        return (prior_delta + prior_sigma / num_dyads)

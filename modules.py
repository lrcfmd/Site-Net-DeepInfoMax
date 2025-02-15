import torch
from torch import nn
import torch.nn.functional as F
from itertools import tee
from torch.nn import LayerNorm,InstanceNorm1d
import numpy as np
from torch_scatter import scatter_std,scatter_mean,segment_coo,segment_csr
from torch_scatter.composite import scatter_std
import sys

################################
# Helper Functions and classes #
################################
#epsilon value for giving numerical stability to asymptotic function
eps = 0.0000008
#Creates a sequential module of linear layers + activation function + normalization for the interaction and site features, used in the attention block
class set_seq_af_norm(nn.Module):
    def __init__(self,layers,af,norm):
        super().__init__()
        layer_list = layers
        first_iter = layer_list[:-1]
        second_iter = layer_list[1:]
        self.linear_layers = nn.ModuleList(nn.Linear(i, j) for i,j in zip(first_iter,second_iter))
        self.af_modules = nn.ModuleList(af() for i in second_iter)
        self.norm_layers = nn.ModuleList(norm(i) for i in second_iter)
    def forward(self,x):
        for i,j,k in zip(self.linear_layers,self.af_modules,self.norm_layers):
            x = i(x)
            x = j(x)
            x = k(x)
        return x
#Convinience class for defining independant perceptrons with activation functions and norms inside the model, used in the attention block
class pairwise_seq_af_norm(nn.Module):
    def __init__(self,layers,af,norm):
        super().__init__()
        layer_list = layers
        first_iter = layer_list[:-1]
        second_iter = layer_list[1:]
        self.linear_layers = nn.ModuleList(nn.Linear(i, j) for i,j in zip(first_iter,second_iter))
        self.af_modules = nn.ModuleList(af() for i in second_iter)
        self.norm_layers = nn.ModuleList(norm(i) for i in second_iter)
    def forward(self,x):
        for i,j,k in zip(self.linear_layers,self.af_modules,self.norm_layers):
            x = i(x)
            x = j(x)
            x = k(x)
        return x

#Sets all but the highest k attention coefficients to negative infinity prior to softmax
#Just performs softmax along the requested dimension otherwise
#Not used in paper, but made available regardless
def k_softmax(x,dim,k):
    if k != -1:
        top_k = x.topk(k,dim=dim,sorted=False)[1]
        mask = torch.zeros_like(x)
        mask.scatter_(dim,top_k,True)
        mask = mask.bool()
        x[~mask] = float("-infinity")
    x = F.softmax(x,dim=dim)
    return x

#Simple transparent module for use with the 3 part framework above
class FakeModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
#Allows iterating through a list with a length 2 sliding window
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

#The mish activation function
class mish(nn.Module):
    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x

class pairwise_norm(nn.Module):
    def __init__(self, dim, norm_func):
        super().__init__()
        self.Norm = norm_func(dim)

    def forward(self, x):
        #Batch mask is B*N,E
        return self.Norm(x)

class set_norm(nn.Module):
    def __init__(self, dim, norm_func):
        super().__init__()
        self.Norm = norm_func(dim)
    def forward(self, x):
        #Batch mask is B*N,E
        return self.Norm(x)

#######################
# Helper Dictionaries #
#######################

#Dictionaries of normalization modules
pairwise_norm_dict = {
    "layer": lambda _: pairwise_norm(_,LayerNorm),
    "none": FakeModule,
}

set_norm_dict = {
    "layer": lambda _: set_norm(_,LayerNorm),
    "none": FakeModule,
}
norm_dict = {
    "layer": LayerNorm,
    "instance": InstanceNorm1d,
    "none": FakeModule,
}
#Dictionary of activation modules
af_dict = {"relu": nn.ReLU, "mish": mish,"sigmoid":nn.Sigmoid,"none":FakeModule,"tanh":nn.Tanh,"relu6":nn.ReLU6}

class SiteNetAttentionBlock(nn.Module):
    def __init__(
        self, site_dim, interaction_dim, heads=4, af="relu", set_norm="batch",tdot=False,k_softmax=-1,attention_hidden_layers=[256,256], site_bottleneck = 64
    ):
        super().__init__()
        #Number of attention heads
        self.heads = heads
        #Site feature vector length per attention head
        self.site_dim = site_dim
        #Interaction feature length per attention head
        self.interaction_dim = interaction_dim
        #Activation function to use in hidden layers
        self.af = af_dict[af]()
        #K softmax value, -1 and unused in the paper
        self.k_softmax = k_softmax
        #Hidden layers for calculating the attention weights (g^W)
        self.ije_to_multihead = pairwise_seq_af_norm([2*site_dim*heads + interaction_dim*heads,*attention_hidden_layers],af_dict[af],pairwise_norm_dict[set_norm])
        #Final layer to generate the attention weights, no activation function or normalization is used
        self.pre_softmax_linear = nn.Linear(attention_hidden_layers[-1],heads)
        #Maps the bond feautres to the new interaction features (g^I)
        self.ije_to_Interaction_Features = pairwise_seq_af_norm([(site_dim * 2 + interaction_dim) * heads, interaction_dim * heads],af_dict[af],pairwise_norm_dict[set_norm])
        #Maps the bond features to the attention features (g^A)
        self.ije_to_attention_features = pairwise_seq_af_norm([(site_dim*2 + interaction_dim) * heads,*attention_hidden_layers, site_dim],af_dict[af],pairwise_norm_dict[set_norm])
        #Linear layer on new site features prior to the next attention block / pooling
        self.global_linear = set_seq_af_norm([site_dim * heads, site_bottleneck],af_dict["none"],set_norm_dict["none"])
    @staticmethod
    def head_reshape(x,heads):
        return x.reshape(*x.shape[:-1],x.shape[-1]//heads,heads)

    def forward(self, x, Interaction_Features, Attention_Mask, Batch_Mask, cutoff_mask=None, m=None):
        #Construct the Bond Features x_ije
        x_i = x[Batch_Mask["attention_i"],:]
        x_j = x[Batch_Mask["attention_j"],:]
        x_ije = torch.cat([x_i, x_j, Interaction_Features], axis=2)
        #Construct the Attention Weights
        multi_headed_attention_weights = self.pre_softmax_linear(self.ije_to_multihead(x_ije)) #g^W
        multi_headed_attention_weights[Attention_Mask] = float("-infinity") #Necessary to avoid interfering with the softmax
        if cutoff_mask is not None:
            multi_headed_attention_weights[cutoff_mask] = float("-infinity")
        #Perform softmax on j
        multi_headed_attention_weights = k_softmax(multi_headed_attention_weights, 1,self.k_softmax) #K_softmax is unused in the paper, ability to disable message passing beyond the highest N coefficients, dynamic graph
        #Compute the attention weights and perform attention
        x = torch.einsum(
            "ijk,ije->iek",
            multi_headed_attention_weights,
            self.ije_to_attention_features(x_ije) #g^F
        )
        #Compute the new site features with g^S and append to the global summary
        x= self.global_linear(torch.reshape(x,[x.shape[0],x.shape[1] * x.shape[2],],)) #g^S
        m = torch.cat([m, x], dim=1) if m != None else x #Keep running total of the site features
        #Compute the new interaction features with g^I
        New_interaction_Features = self.ije_to_Interaction_Features(x_ije) #g^I
        return x, New_interaction_Features, m

class SiteNetEncoder(nn.Module):
    def __init__(
        self,
        embedding_size=100,
        site_feature_size=1,
        attention_blocks=4,
        attention_heads=4,
        site_dim_per_head=64,
        pre_pool_layers=[256, 256],
        post_pool_layers=[256, 256],
        activation_function="relu",
        sym_func="mean",
        set_norm="none",
        lin_norm="none",
        interaction_feature_size=3,
        attention_dim_interaction=64,
        tdot=False,
        attention_hidden_layers=[256,256],
        k_softmax=-1,
        distance_cutoff=-1,

        **kwargs,
    ):
        super().__init__()
        #Site Layers
        self.full_elem_token_size = embedding_size + site_feature_size + 1
        self.site_featurization = nn.Linear(
            self.full_elem_token_size, site_dim_per_head * attention_heads
        )
        self.site_featurization_norm = set_norm_dict[set_norm](
            site_dim_per_head * attention_heads
        )
        #Interaction Layers
        self.interaction_featurization = nn.Linear(
            interaction_feature_size,
            attention_dim_interaction * attention_heads,
        )
        self.interaction_featurization_norm = pairwise_norm_dict[set_norm](
            attention_dim_interaction * attention_heads
        )
        self.distance_cutoff=distance_cutoff

        #Attention Layers
        self.Attention_Blocks = nn.ModuleList(
            SiteNetAttentionBlock(
                site_dim_per_head,
                attention_dim_interaction,
                af=activation_function,
                heads=attention_heads,
                set_norm=set_norm,
                tdot=tdot,
                k_softmax=k_softmax,
                attention_hidden_layers = attention_hidden_layers
            )
            for i in range(attention_blocks)
        )
        # Pooling Layers
        self.sym_func = sym_func
        self.pre_pool_layers = nn.ModuleList(
            nn.Linear(i, j)
            for i, j in pairwise(
                (
                    site_dim_per_head
                    * attention_blocks
                    * attention_heads,
                    *pre_pool_layers,
                )
            )
        )
        self.pre_pool_layers_norm = nn.ModuleList(
            set_norm_dict[set_norm](i) for i in pre_pool_layers
        )
        if sym_func == "mean" or sym_func == "max":
            self.post_pool_layers = nn.ModuleList(
                nn.Linear(i, j)
                for i, j in pairwise((pre_pool_layers[-1], *post_pool_layers))
            )
            self.post_pool_layers_norm = nn.ModuleList(
                norm_dict[lin_norm](i) for i in post_pool_layers
            )
        self.af = af_dict[activation_function]()

    def forward(
        self,
        Site_Features,
        Interaction_Features,
        Attention_Mask,
        Batch_Mask
    ):
        # Interaction Feature Dimensions (i,j,Batch,Embedding)
        # Site Feature Dimensions (i,Batch,Embedding)
        # Attention Mask Dimensions (Batch,i), True for padding, False for data

        #Compute Optional attention mask for cut off mode
        if self.distance_cutoff >= 0:
            cutoff_mask = torch.gt(Interaction_Features[:,:,0],self.distance_cutoff)
        else:
            cutoff_mask = None
      
        #We concat the site feautre outputs as we go to global_summary, intialized here so it exists
        global_summary = None
        #Bring Interaction Features to dimensionality expected by the attention blocks
        Interaction_Features = self.interaction_featurization_norm(
            self.af(self.interaction_featurization(Interaction_Features))
        )
        #Bring Site Features to dimensionality expected by the attention blocks
        Site_Features = self.site_featurization_norm(
            self.af(self.site_featurization(Site_Features))
        )
        #Apply the attention blocks and build up the summary site feature vectors accordingly
        for layer in self.Attention_Blocks:
            Site_Features, Interaction_Features, global_summary = layer(
                Site_Features, Interaction_Features, Attention_Mask, Batch_Mask, cutoff_mask=cutoff_mask, m=global_summary
            )        
        #Apply the pre pooling layers
        for pre_pool_layer, pre_pool_layer_norm in zip(
            self.pre_pool_layers, self.pre_pool_layers_norm
        ):
            global_summary = pre_pool_layer_norm(
                self.af(pre_pool_layer(global_summary))
            )
        
        #Apply the symettric aggregation function to get the global representation
        #segment_csr takes the mean or max across the whole batch
        if self.sym_func == "mean":
            Global_Representation = segment_csr(global_summary,Batch_Mask["CSR"],reduce="mean")
        elif self.sym_func == "max":
            Global_Representation = segment_csr(global_summary,Batch_Mask["CSR"],reduce="max")
        else:
            raise Exception()
        #Apply the post pooling layers
        for post_pool_layer, post_pool_layer_norm in zip(
            self.post_pool_layers, self.post_pool_layers_norm
        ):
            Global_Representation = post_pool_layer_norm(
                self.af(post_pool_layer(Global_Representation))
            )
        return Global_Representation

##############################################################################################################################################################################################
#
#  DEEP INFOMAX STUFF STARTS HERE
#
###############################################################################################################################################################################################

class SiteNetDIMGlobal(nn.Module):
    def __init__(
        self,
        embedding_size=100,
        site_feature_size=1,
        attention_blocks=4,
        attention_heads=4,
        site_dim_per_head=64,
        pre_pool_layers=[256, 256],
        post_pool_layers=[256, 256],
        classifier_hidden_layers=[64],
        activation_function="relu",
        sym_func="mean",
        set_norm="none",
        lin_norm="none",
        interaction_feature_size=3,
        attention_dim_interaction=64,
        global_dot_space = 256,
        site_bottleneck = 64,
        distance_cutoff=-1,
        global_dot_hidden_layers=[64],
        **kwargs,
    ):
        super().__init__()
        #Site Layers
        self.full_elem_token_size = embedding_size + site_feature_size + 1
        self.site_featurization = nn.Linear(
            self.full_elem_token_size, site_dim_per_head * attention_heads
        )
        self.site_featurization_norm = set_norm_dict[set_norm](
            site_dim_per_head * attention_heads
        )
        #Interaction Layers
        self.interaction_featurization = nn.Linear(
            interaction_feature_size,
            attention_dim_interaction * attention_heads,
        )
        self.interaction_featurization_norm = pairwise_norm_dict[set_norm](
            attention_dim_interaction * attention_heads
        )
        self.distance_cutoff=distance_cutoff
        # Pooling Layers
        self.sym_func = sym_func
        self.pre_pool_layers = nn.ModuleList(
            nn.Linear(i, j)
            for i, j in pairwise(
                (
                    site_bottleneck,
                    *pre_pool_layers,
                )
            )
        )
        self.pre_pool_layers_norm = nn.ModuleList(
            set_norm_dict[set_norm](i) for i in pre_pool_layers
        )
        if sym_func == "mean" or sym_func == "max":
            post_pool_layers_complete = [pre_pool_layers[-1], *post_pool_layers]
            self.post_pool_layers = nn.ModuleList(
                nn.Linear(i, j)
                for i, j in pairwise((post_pool_layers_complete))
            )
            self.post_pool_layer_std = nn.Linear(post_pool_layers_complete[-2],post_pool_layers_complete[-1])
            self.post_pool_layers_norm = nn.ModuleList(
                norm_dict[lin_norm](i) for i in post_pool_layers
            )
        self.af = af_dict[activation_function]()
        self.localenv_upscale_layers = [site_bottleneck,*global_dot_hidden_layers]
        self.global_upscale_layers = [post_pool_layers[-1],*global_dot_hidden_layers]
        self.localenv_upscale = set_seq_af_norm(self.localenv_upscale_layers,af_dict[activation_function],set_norm_dict["none"])
        self.global_upscale = set_seq_af_norm(self.global_upscale_layers,af_dict[activation_function],set_norm_dict["none"])
        self.localenv_upscale_final_layer = nn.Linear(self.localenv_upscale_layers[-1],global_dot_space)
        self.global_upscale_final_layer = nn.Linear(self.global_upscale_layers[-1],global_dot_space)

    def inference(
        self,
        LocalEnvironment_Features,
        Batch_Mask,
    ):

        #Apply the pre pooling layers
        for pre_pool_layer, pre_pool_layer_norm in zip(
            self.pre_pool_layers, self.pre_pool_layers_norm
        ):
            LocalEnvironment_Features = pre_pool_layer_norm(
                self.af(pre_pool_layer(LocalEnvironment_Features))
            )
        
        #Apply the symettric aggregation function to get the global representation
        if self.sym_func == "mean":
            Global_Representation = segment_csr(LocalEnvironment_Features,Batch_Mask["CSR"],reduce="mean")
        elif self.sym_func == "max":
            Global_Representation = segment_csr(LocalEnvironment_Features,Batch_Mask["CSR"],reduce="max")
        else:
            raise Exception()

        #Apply the post pooling layers
        for idx,(post_pool_layer, post_pool_layer_norm) in enumerate(zip(
            self.post_pool_layers, self.post_pool_layers_norm)
        ):
            if idx == len(self.post_pool_layers)-1:
                Global_Representation = post_pool_layer(Global_Representation)
            else:
                Global_Representation = post_pool_layer_norm(
                    self.af(post_pool_layer(Global_Representation))
                )
        
        return Global_Representation

    def forward(
        self,
        LocalEnvironment_Features,
        False_LocalEnvironment_Features,
        Batch_Mask,
        KL = False
    ):
        #Need a detached copy so gradients don't get propogated on the sample end
        detached_LocalEnvironment_Features = LocalEnvironment_Features.detach().clone()
        
        #Apply the pre pooling layers
        for pre_pool_layer, pre_pool_layer_norm in zip(
            self.pre_pool_layers, self.pre_pool_layers_norm
        ):
            LocalEnvironment_Features = pre_pool_layer_norm(
                self.af(pre_pool_layer(LocalEnvironment_Features))
            )
        
        #Apply the permutation invariant aggregation function to get the global representation
        if self.sym_func == "mean":
            Global_Representation = segment_csr(LocalEnvironment_Features,Batch_Mask["CSR"],reduce="mean")
        elif self.sym_func == "max":
            Global_Representation = segment_csr(LocalEnvironment_Features,Batch_Mask["CSR"],reduce="max")
        else:
            raise Exception()
        
        #Apply the post pooling layers
        for idx,(post_pool_layer, post_pool_layer_norm) in enumerate(zip(
            self.post_pool_layers, self.post_pool_layers_norm)
        ):
            if idx == len(self.post_pool_layers)-1: #Skip activiation function and normalization on the final layer, otherwise KL divergence term breaks
                if KL:
                    Global_Representation_log_var = self.post_pool_layer_std(Global_Representation)
                Global_Representation = post_pool_layer(Global_Representation)
            else:
                Global_Representation = post_pool_layer_norm(
                    self.af(post_pool_layer(Global_Representation))
                )
        #Global_Representation_Sample = Global_Representation+torch.randn_like(Global_Representation_log_var)*torch.exp(Global_Representation_log_var/2)
        if KL:
            Global_Representation_Sample = self.global_upscale_final_layer(self.global_upscale(Global_Representation + torch.exp(0.5*Global_Representation_log_var)*torch.randn_like(Global_Representation)))
        else:
            Global_Representation_Sample = self.global_upscale_final_layer(self.global_upscale(Global_Representation))
        local_env_samples = self.localenv_upscale_final_layer(self.localenv_upscale(detached_LocalEnvironment_Features))
        false_local_env_samples = [self.localenv_upscale_final_layer(self.localenv_upscale(i.detach().clone())) for i in False_LocalEnvironment_Features]

        #Get the false scores for the engineered false samples
        False_Scores = [F.softplus(torch.einsum("ik,ik->i",Global_Representation_Sample[Batch_Mask["COO"]],i)) for i in false_local_env_samples]

        #Roll the batch mask to get the score for the false samples taken from other crystals
        False_Batch_Mask_COO = torch.roll(Batch_Mask["COO"],len(Batch_Mask["COO"])//2,0)
        False_Scores.append(F.softplus(torch.einsum("ik,ik->i",Global_Representation_Sample[False_Batch_Mask_COO],local_env_samples)))
        
        True_Score = F.softplus(-torch.einsum("ik,ik->i",Global_Representation_Sample[Batch_Mask["COO"]],local_env_samples))
        #Get DIM_loss per crystal
        DIM_loss = segment_csr(sum(False_Scores)/len(False_Scores)+True_Score,Batch_Mask["CSR"],reduce="mean").flatten().mean()
        #DIM_loss = segment_csr(False_Score_1+True_Score,Batch_Mask["CSR"],reduce="mean").flatten().mean()
        if KL:
            KL_loss = (0.5*Global_Representation**2+torch.exp(Global_Representation_log_var)-Global_Representation_log_var).flatten().mean()
        else:
            KL_loss = torch.tensor(0,dtype=torch.float)

        return Global_Representation,DIM_loss,KL_loss

class SiteNetDIMAttentionBlock(nn.Module):
    def __init__(
        self, af="relu", set_norm="batch",tdot=False,k_softmax=-1,attention_hidden_layers=[256,256],
        site_dim_per_head = 16, attention_heads = 4, attention_dim_interaction = 16,embedding_size=100, site_bottleneck = 64,
        site_feature_size=1,interaction_feature_size=3, site_dot_space=256,site_dot_hidden_layers=[256],distance_cutoff=-1, extra_false_samples = True,**kwargs
    ):
        super().__init__()
        self.distance_cutoff=distance_cutoff
        self.full_elem_token_size = embedding_size + site_feature_size + 1
        self.heads = attention_heads
        self.site_dim = site_dim_per_head * attention_heads
        self.site_bottleneck = site_bottleneck
        self.interaction_dim = attention_dim_interaction * attention_heads
        self.glob_dim = site_dim_per_head
        self.af = af_dict[af]()
        self.k_softmax = k_softmax
        self.extra_false_samples = extra_false_samples

        self.site_featurization = nn.Linear(
            self.full_elem_token_size, site_dim_per_head * attention_heads
        )
        self.site_featurization_norm = set_norm_dict[set_norm](
            site_dim_per_head * attention_heads
        )
        #Interaction Layers
        self.interaction_featurization = nn.Linear(
            interaction_feature_size,
            attention_dim_interaction * attention_heads,
        )
        self.interaction_featurization_norm = pairwise_norm_dict[set_norm](
            attention_dim_interaction * attention_heads
        )
        self.ije_to_multihead = pairwise_seq_af_norm([2*self.site_dim + self.interaction_dim,*attention_hidden_layers],af_dict[af],pairwise_norm_dict[set_norm])
        self.pre_softmax_linear = nn.Linear(attention_hidden_layers[-1],attention_heads)
        self.ije_to_attention_features = pairwise_seq_af_norm([self.site_dim*2 + self.interaction_dim, *attention_hidden_layers, self.glob_dim],af_dict[af],pairwise_norm_dict[set_norm])
        self.global_linear = set_seq_af_norm([self.site_dim, self.site_bottleneck],af_dict["none"],set_norm_dict["none"])
        self.global_linear_std = set_seq_af_norm([self.site_dim, self.site_bottleneck],af_dict["none"],set_norm_dict["none"])
        self.dim_upscale_layers = [self.site_bottleneck,*site_dot_hidden_layers]
        self.sample_upscale_layers = [self.full_elem_token_size + interaction_feature_size,*site_dot_hidden_layers]
        self.dim_upscale = set_seq_af_norm(self.dim_upscale_layers,af_dict[af],set_norm_dict["none"])
        self.sample_upscale = set_seq_af_norm(self.sample_upscale_layers,af_dict[af],set_norm_dict["none"])
        self.dim_upscale_final_layer = nn.Linear(self.dim_upscale_layers[-1],site_dot_space)
        self.sample_upscale_final_layer = nn.Linear(self.sample_upscale_layers[-1],site_dot_space)
    @staticmethod
    def head_reshape(x,attention_heads):
        return x.reshape(*x.shape[:-1],x.shape[-1]//attention_heads,attention_heads)
    
    @staticmethod
    #This requires the batch size to be at least twice as large as the largest sample
    def false_sample(x,dim):
        return torch.roll(x,x.shape[dim]//2,dim)

    def inference(self, x, Interaction_Features, Attention_Mask, Batch_Mask,KL = False):
        if self.distance_cutoff >= 0:
            cutoff_mask = torch.gt(Interaction_Features[:,:,0],self.distance_cutoff)
        else:
            cutoff_mask = None

        #Bring Interaction Features to dimensionality expected by the attention blocks
        Interaction_Features = self.interaction_featurization_norm(
            self.af(self.interaction_featurization(Interaction_Features))
        )
        #Bring Site Features to dimensionality expected by the attention blocks
        x = self.site_featurization_norm(
            self.af(self.site_featurization(x))
        )
        #Construct the Bond Features x_ije
        x_i = x[Batch_Mask["attention_i"],:]
        x_j = x[Batch_Mask["attention_j"],:]
        x_ije = torch.cat([x_i, x_j, Interaction_Features], axis=2)

        #Construct the Attention Weights
        multi_headed_attention_weights = self.pre_softmax_linear(self.ije_to_multihead(x_ije)) #g^W
        multi_headed_attention_weights[Attention_Mask] = float("-infinity") #Necessary to avoid interfering with the softmax
        
        #Apply cutoff mask
        if cutoff_mask is not None:
            multi_headed_attention_weights[cutoff_mask] = float("-infinity")

        #Perform softmax on j
        multi_headed_attention_weights = k_softmax(multi_headed_attention_weights, 1,self.k_softmax) #K_softmax is unused in the paper, ability to disable message passing beyond the highest N coefficients, dynamic graph
        #Compute the attention weights and perform attention
        x = torch.einsum(
            "ijk,ije->iek",
            multi_headed_attention_weights,
            self.ije_to_attention_features(x_ije) #g^F
        )
        #Combine the heads together
        x = torch.reshape(x,[x.shape[0],x.shape[1] * x.shape[2],],)
        x = self.global_linear(x) #g^S
        return x


    def forward(self, x, Interaction_Features, Attention_Mask, Batch_Mask,KL = False):
        if self.distance_cutoff >= 0:
            cutoff_mask = torch.gt(Interaction_Features[:,:,0],self.distance_cutoff)
        else:
            cutoff_mask = None
        #Detach the original input features so they can be used later for DIM
        detached_Interaction_Features = Interaction_Features.detach().clone()
        detached_x_j = x[Batch_Mask["attention_j"],:].detach().clone()
        #Bring Interaction Features to dimensionality expected by the attention blocks
        Interaction_Features = self.interaction_featurization_norm(
            self.af(self.interaction_featurization(Interaction_Features))
        )
        #Bring Site Features to dimensionality expected by the attention blocks
        x = self.site_featurization_norm(
            self.af(self.site_featurization(x))
        )
        #Construct the Bond Features x_ije
        x_i = x[Batch_Mask["attention_i"],:]
        x_j = x[Batch_Mask["attention_j"],:]
        x_ije = torch.cat([x_i, x_j, Interaction_Features], axis=2)
        #Construct the Attention Weights
        multi_headed_attention_weights = self.pre_softmax_linear(self.ije_to_multihead(x_ije)) #g^W
        multi_headed_attention_weights[Attention_Mask] = float("-infinity") #Necessary to avoid interfering with the softmax
        
        #Apply cutoff mask
        if cutoff_mask is not None:
            multi_headed_attention_weights[cutoff_mask] = float("-infinity")
        
        #Perform softmax on j
        multi_headed_attention_weights = k_softmax(multi_headed_attention_weights, 1,self.k_softmax) #K_softmax is unused in the paper, ability to disable message passing beyond the highest N coefficients, dynamic graph
        #Compute the attention weights and perform attention
        x = torch.einsum(
            "ijk,ije->iek",
            multi_headed_attention_weights,
            self.ije_to_attention_features(x_ije) #g^F
        )
        #Combine the heads together
        x = torch.reshape(x,[x.shape[0],x.shape[1] * x.shape[2],],)
        if KL:
            x_log_var = self.global_linear_std(x)
        x = self.global_linear(x) #g^S

        distance_weights = (detached_Interaction_Features[:,:,0]+1)**-2
        distance_weights_sum_reciprocal = (torch.sum((distance_weights*~Attention_Mask),1)**-1).unsqueeze(1)

        if KL:
            x_sample = x + torch.randn_like(x)*torch.exp(0.5*x_log_var)
        else:
            x_sample = x
            
        x_sample = self.dim_upscale_final_layer(self.dim_upscale(x_sample)[Batch_Mask["attention_i"],:])
        true_queries = self.sample_upscale_final_layer(self.sample_upscale(torch.cat([detached_x_j, detached_Interaction_Features], axis=2)))
        false_queries_1 = self.sample_upscale_final_layer(self.sample_upscale(torch.cat([self.false_sample(detached_x_j,0), self.false_sample(detached_Interaction_Features,0)], axis=2))) #Fully Fake
        false_queries_2 = self.sample_upscale_final_layer(self.sample_upscale(torch.cat([detached_x_j, self.false_sample(detached_Interaction_Features,0)], axis=2))) #Fake distances only
        false_queries_3 = self.sample_upscale_final_layer(self.sample_upscale(torch.cat([self.false_sample(detached_x_j,0), detached_Interaction_Features], axis=2))) #Fake composition only
        #Compute classification score and normalize by distance
        false_scores_1 = F.softplus(torch.einsum("ijk,ijk->ij",x_sample,false_queries_1)).squeeze()*self.false_sample(distance_weights,0) #Need to weight with false distances
        false_scores_2 = F.softplus(torch.einsum("ijk,ijk->ij",x_sample,false_queries_2)).squeeze()*self.false_sample(distance_weights,0) #Need to weight with false distances
        false_scores_3 = F.softplus(torch.einsum("ijk,ijk->ij",x_sample,false_queries_3)).squeeze()*distance_weights
        true_scores = F.softplus(-torch.einsum("ijk,ijk->ij",x_sample,true_queries)).squeeze()*distance_weights

        #Aggregate individual losses over the local environment, weighted by distance
        false_scores_1 =  torch.sum(false_scores_1*self.false_sample(distance_weights_sum_reciprocal,0)*~self.false_sample(Attention_Mask,0),1) #Need to weight with false distances and use false attention mask
        false_scores_2 =  torch.sum(false_scores_2*self.false_sample(distance_weights_sum_reciprocal,0)*~Attention_Mask,1) #Need to weight with false distances
        false_scores_3 =  torch.sum(false_scores_3*distance_weights_sum_reciprocal*~self.false_sample(Attention_Mask,0),1) #Need to use false attention mask
        true_scores =  torch.sum(true_scores*distance_weights_sum_reciprocal*~Attention_Mask,1)

        #Combine losses
        #DIM_loss = (true_scores+(false_scores_1+false_scores_2+false_scores_3)/3).squeeze()
        if self.extra_false_samples == True:
            DIM_loss = (true_scores + (false_scores_1 + false_scores_2 + false_scores_3)/3).squeeze()
        else:
            DIM_loss = (true_scores + false_scores_1).squeeze()

        #Calculate weighted loss per crystal
        DIM_loss = segment_csr(DIM_loss,Batch_Mask["CSR"],reduce="mean")
        #calculate weighted loss per batch
        DIM_loss = DIM_loss.flatten().mean()
        if KL:
            KL_loss = (0.5*x**2+torch.exp(x_log_var)-x_log_var).flatten().mean()
        else:
            KL_loss = torch.tensor(0,dtype=torch.float)

        #loss = DIM_loss + 0.1*KL_loss

        return x,DIM_loss,KL_loss

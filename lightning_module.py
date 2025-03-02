import torch
import pytorch_lightning as pl
from h5_handler import *
from torch.utils.data import DataLoader, random_split
from multiprocessing import cpu_count
from torch.nn.functional import pad
from modules import (
    SiteNetEncoder,SiteNetDIMAttentionBlock,SiteNetDIMGlobal
)
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import *
from torch import nn
from torch_scatter import segment_csr
from torch.utils.data import RandomSampler,Sampler,SequentialSampler
from pymatgen.transformations.standard_transformations import *
from itertools import cycle,islice
from random import shuffle
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.site.chemical import ChemicalSRO,EwaldSiteEnergy,LocalPropertyDifference
import sys
#Clamps negative predictions to zero without interfering with the gradients. "Transparent" ReLU
class TReLU(torch.autograd.Function):
    """
    A transparent version of relu that has a linear gradient but sets negative values to zero,
     used as the last step in band gap prediction to provide an alternative to relu which does not kill gradients
      but also prevents the model from being punished for negative band gap predictions as these can readily be interpreted as zero
    """

    @staticmethod
    def forward(ctx, input):
        """
        f(x) is equivalent to relu
        """
        
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        f'(x) is linear
        """
        return grad_output

#Dictionaries allow programatic access of torch modules according to the config file
optim_dict = torch.optim.__dict__
site_feauturizers_dict = matminer.featurizers.site.__dict__
af_dict = {"identity":lambda x:x,"relu":nn.functional.relu,"softplus":nn.functional.softplus,"TReLU":TReLU.apply,"relu6":nn.ReLU6}   


#Constructs a batch dictionary from the list of property dictionaries returned by the h5 loader
#Also performs necessary zero padding on the j axis and creates the batch masks
def collate_fn(batch, inference=False):
    batch_dict = {}
    #Necessary information to perform the batching
    primitive_lengths = [i["prim_size"] for i in batch]
    image_count = [i["images"] for i in batch]
    actual_length = [i["prim_size"]*i["images"] for i in batch]

    #Turns the numpy arrays into torch tensors ready for the model
    def initialize_tensors(batch_full, key, dtype, process_func=lambda _: _):
        batch = [i[key] for i in batch_full]
        batch = [process_func(i) for i in batch]
        batch = [torch.as_tensor(np.array(i), dtype=dtype) for i in batch]
        return batch

    def composition_tensor_from_structure(struct):
        composition = struct.composition
        frac_vector = ElementFraction().featurize(composition)
        return torch.as_tensor(np.array(frac_vector),dtype=torch.float)
 

    # 2 dimensional stacking for the adjaceny matricies
    def adjacency_stack(batch):
        max_matrix_dim = max(
            i.shape[1] for i in batch
        )  # Should be of dimensions (i,j,E)
        stacked = torch.cat(
            [
                pad(
                    input=i,
                    pad=(
                        0,
                        0,
                        0,
                        max_matrix_dim - i.shape[1],
                        0,
                        0,
                    ),
                )
                for i in batch
            ],
            0
        )
        return stacked
    Atomic_ID = initialize_tensors(batch, "Atomic_ID", torch.long)
    site_features = initialize_tensors(batch, "Site_Feature_Tensor", torch.float)
    site_labels = initialize_tensors(batch, "Site_Label_Tensor", torch.float)
    interaction_features = initialize_tensors(batch, "Interaction_Feature_Tensor", torch.float)
    Oxidation_State = initialize_tensors(batch, "Oxidation_State", torch.float)
    composition = [composition_tensor_from_structure(i["structure"]) for i in batch]
    # Pack Crystal Features
    batching_mask_COO = []
    batching_mask_CSR = []
    index_CSR = 0
    for idx,i in enumerate(site_features):
        batching_mask_CSR.append(index_CSR)
        index_CSR += i.shape[0]
        batching_mask_COO.extend([idx]*i.shape[0])
    batching_mask_CSR.append(index_CSR)
    array_i = []
    array_j = []
    for idx,batch_n in enumerate(np.unique(batching_mask_COO)):
        item_index = np.where(batching_mask_COO == batch_n)[0]
        prim_count = primitive_lengths[idx]
        images = image_count[idx]
        #Creates the site indicies for the i'th site in the bond feautres for each j
        array_i.append(np.array([[i for _ in range(max(actual_length))] for i in item_index]))
        #Creates the site indicies for the j'th component of the bond feautues for each i
        array_j.append(np.array([[item_index[(i//images)%prim_count] for i in range(max(actual_length))] for _ in item_index]))
    #These masks allows the self attention mechanism to index the correct site features when constructing the bond features, without the use of a batch dimension
    batching_mask_attention_i = torch.tensor(np.concatenate(array_i,0),dtype=torch.long)
    batching_mask_attention_j = torch.tensor(np.concatenate(array_j,0),dtype=torch.long)
    #Different "scattered" functions enabling batching along i require different expressions of the batch indicies
    #COO labels every position along i according to its batch
    #CSR provides the indicies where each new batch begins
    #iterate adds the length of the site features onto the end of the array to enable sliding window iteration
    batching_mask_COO = torch.tensor(batching_mask_COO,dtype=torch.long)
    batching_mask_CSR = torch.tensor(batching_mask_CSR,dtype=torch.long)

    #The attention mask excludes the zero padding along j during attention
    #Actual length is the number of real atomic sites each supercell actually has, the primitive point group multiplied by the number of images we consider
    #Anything past the actual length is junk data and needs to be marked as such
    base_attention_mask = torch.stack(
        [
            torch.tensor(
                [False if i < len else True for i in range(max(actual_length))],
                dtype=torch.bool,
            )
            for len in actual_length
        ]
    )
    batch_dict["Attention_Mask"] = base_attention_mask[batching_mask_COO,:]
    batch_dict["Site_Feature_Tensor"] = torch.cat(site_features,0)
    batch_dict["Site_Label_Tensor"] = torch.cat(site_labels,0)
    batch_dict["Interaction_Feature_Tensor"] = adjacency_stack(interaction_features)
    batch_dict["Atomic_ID"] = torch.cat(Atomic_ID,0)
    batch_dict["Oxidation_State"] = torch.cat(Oxidation_State,0)
    batch_dict["target"] = torch.as_tensor(np.array([(i["target"]) for i in batch]))
    batch_dict["Batch_Mask"] = {"COO":batching_mask_COO,"CSR":batching_mask_CSR,"attention_i":batching_mask_attention_i,"attention_j":batching_mask_attention_j}
    batch_dict["Composition"] = torch.stack(composition,0)
    if inference: #If the batch dictionary contains things that are not tensors while training it breaks pytorch lightning
        batch_dict["Structure"] = [i["structure"] for i in batch]
    return batch_dict

class SiteNet(pl.LightningModule):
    def __init__(
        self,
        config=None,
    ):
        super().__init__()
        if config != None: #Implies laoding from a checkpoint if None
            #Load in the hyper parameters as lightning model attributes
            self.config = config
            self.batch_size = self.config["Batch_Size"]
            self.decoder = nn.Linear(self.config["post_pool_layers"][-1], 1)
            self.encoder = SiteNetEncoder(**self.config)
            self.site_feature_scalers = nn.parameter.Parameter(torch.tensor(self.config["Site_Feature_scalers"]),requires_grad=False)

            self.config["pre_pool_layers_n"] = len(config["pre_pool_layers"])
            self.config["pre_pool_layers_size"] = sum(config["pre_pool_layers"]) / len(
                config["pre_pool_layers"]
            )
            self.config["post_pool_layers_n"] = len(config["post_pool_layers"])
            self.config["post_pool_layers_size"] = sum(config["post_pool_layers"]) / len(
                config["post_pool_layers"]
            )
            #Initialize the learnt elemental embeddings
            self.Elemental_Embeddings = nn.Embedding(
                200,
                self.config["embedding_size"],
                max_norm=1,
                scale_grad_by_freq=False,
            )
            self.save_hyperparameters(self.config)
    #Constructs the site features from the individual pieces, including the learnt atomic embeddings if enabled
    def input_handler(self, atomic_number, features, Learnt_Atomic_Embedding=True):
        #Removed learnt embeddings or one hot encodings for the deep infomax code
        return torch.cat(features, dim=1)

    #Inference mode, return the prediction
    def forward(self, b, batch_size=16,return_truth = False):
        self.eval()
        lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
        Encoding_list = []
        targets_list= []
        print("Inference in batches of %s" % batch_size)
        for inference_batch in tqdm(lob):
            batch_dictionary = collate_fn(inference_batch, inference=True)
            Attention_Mask = batch_dictionary["Attention_Mask"]
            Site_Feature = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
            Atomic_ID = batch_dictionary["Atomic_ID"]
            Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
            Oxidation_State = batch_dictionary["Oxidation_State"]
            Batch_Mask = batch_dictionary["Batch_Mask"]

            concat_embedding = self.input_handler(
                Atomic_ID, [Site_Feature, Oxidation_State]
            )
            with torch.no_grad():
                Encoding = self.encoder.forward(
                    concat_embedding,
                    Interaction_Features,
                    Attention_Mask,
                    Batch_Mask,
                    return_std=False,
                )
                Encoding = af_dict[self.config["last_af_func"]](self.decoder(Encoding))
                Encoding_list.append(Encoding)
                targets_list.append(batch_dictionary["target"])
        Encoding = torch.cat(Encoding_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        self.train()
        if return_truth:
            return [Encoding,targets]
        else:
            return Encoding

    def shared_step(
        self,
        batch_dictionary,
        log_list=None,
    ):
        #Unpack the data from the batch dictionary
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Feature = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        x = self.input_handler(Atomic_ID, [Site_Feature, Oxidation_State])
        # Pass through Encoder to get the global representation
        Global_Embedding = self.encoder.forward(
            x,
            Interaction_Features,
            Attention_Mask,
            Batch_Mask
        )
        #Perform the final layer and get the prediction
        prediction = af_dict[self.config["last_af_func"]](self.decoder(Global_Embedding))
        #Makes sure the prediction is a scalar just in case its a length 1 array
        while len(prediction.shape) > 1:
            prediction = prediction.squeeze()
        #Compute the MAE of the batch
        MAE = torch.abs(prediction - batch_dictionary["target"]).mean()
        MSE = torch.square(prediction - batch_dictionary["target"]).mean()
        #Log the average prediction
        prediction = prediction.mean()
        #Log for tensorboard
        if log_list is not None:
            for i in log_list:
                self.log(i[0], i[2](locals()[i[1]]), **i[3])
        return MAE,MSE
    #Makes sure the model is in training mode, passes a batch through the model, then back propogates
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        log_list = [
            ["MAE", "MAE", lambda _: _, {}, {"prog_bar": True}],
            ["MSE","MSE", lambda _: _, {}, {"prog_bar": True}],
            ["prediction", "prediction", lambda _: _, {}, {"prog_bar": True}],
        ]
        return self.shared_step(
            batch_dictionary,
            log_list=log_list,
        )[1]
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        log_list = None
        return self.shared_step(batch_dictionary, log_list=log_list)[0]

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        optimizer = optim_dict[Optimizer_Config["Name"]](
            self.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],
        )
        return {
            "optimizer": optimizer,
        }

    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", self.avg_loss)
        self.log("hp_metric", self.avg_loss)

class basic_callbacks(pl.Callback):
    def __init__(self,*pargs,filename = "current_model",**kwargs):
        super().__init__(*pargs,**kwargs)
        self.filename = filename + ".ckpt"

    def on_train_end(self, trainer, model):
        trainer.save_checkpoint("most_recent_complete_run.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint("current_model.ckpt")

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.filename)

class SiteNet_DIM(pl.LightningModule):
    def __init__(
        self,
        config=None,
    ):
        super().__init__()
        if config != None: #Implies laoding from a checkpoint if None
            #Load in the hyper parameters as lightning model attributes
            self.config = config
            self.batch_size = self.config["Batch_Size"]

            self.Site_DIM = SiteNetDIMAttentionBlock(**config)
            self.Global_DIM = SiteNetDIMGlobal(**config)
            self.Site_Prior = nn.Sequential(nn.Linear(config["site_bottleneck"],256),nn.Mish(),nn.Linear(256,1))
            self.Global_Prior = nn.Sequential(nn.Linear(config["post_pool_layers"][-1],256),nn.Mish(),nn.Linear(256,1))
            self.Composition_Decoder = nn.Sequential(nn.Linear(config["post_pool_layers"][-1],256),nn.Mish(),nn.Linear(256,103))
            self.decoder = nn.Sequential(nn.Linear(config["post_pool_layers"][-1],256),nn.Mish(),nn.Linear(256,1))
            self.local_decoder = nn.Sequential(nn.Linear(config["site_bottleneck"],256),nn.Mish(),nn.Linear(256,8))
            self.site_feature_scalers = nn.parameter.Parameter(torch.tensor(self.config["Site_Feature_scalers"]),requires_grad=False)
            self.site_label_scalers = nn.parameter.Parameter(torch.tensor(self.config["Site_Label_scalers"]),requires_grad=False)

            self.config["pre_pool_layers_n"] = len(config["pre_pool_layers"])
            self.config["pre_pool_layers_size"] = sum(config["pre_pool_layers"]) / len(
                config["pre_pool_layers"]
            )
            self.config["post_pool_layers_n"] = len(config["post_pool_layers"])
            self.config["post_pool_layers_size"] = sum(config["post_pool_layers"]) / len(
                config["post_pool_layers"]
            )
            self.save_hyperparameters(self.config)
            self.automatic_optimization=False
    #Constructs the site features from the individual pieces, including the learnt atomic embeddings if enabled
    def input_handler(self, atomic_number, features):
        #Removed learnt embeddings or one hot encodings for the deep infomax code
        return torch.cat(features, dim=1)

    #Inference mode, return the prediction
    def forward(self, b, batch_size=16):
        with torch.no_grad():  
            self.eval()
            lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
            Encoding_list = []
            print("Inference in batches of %s" % batch_size)
            for inference_batch in tqdm(lob):
                batch_dictionary = collate_fn(inference_batch,inference=True)

                Attention_Mask = batch_dictionary["Attention_Mask"].to(self.device)
                Site_Features = batch_dictionary["Site_Feature_Tensor"].to(self.device)*self.site_label_scalers
                Atomic_ID = batch_dictionary["Atomic_ID"].to(self.device)
                Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"].to(self.device)
                Oxidation_State = batch_dictionary["Oxidation_State"].to(self.device)
                Batch_Mask = {i:j.to(self.device) for i,j in batch_dictionary["Batch_Mask"].items()}
                
                Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
                Local_Environment_Features = self.Site_DIM.inference(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)
                Global_Embedding_Features = self.Global_DIM.inference(Local_Environment_Features.detach().clone(),Batch_Mask)
                Encoding_list.append(Global_Embedding_Features)
            Encoding = torch.cat(Encoding_list, dim=0)
            return Encoding

    @staticmethod
    #This requires the batch size to be at least twice as large as the largest sample
    def false_sample(x,dim):
        return torch.roll(x,x.shape[dim]//2,dim)

    #Makes sure the model is in training mode, passes a batch through the model, then back propogates
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        local_opt,global_opt,task_opt,local_task_opt,local_prior_opt,global_prior_opt = self.optimizers()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])

        #Perform a step on creating local environment representations while tricking the prior discriminator
        local_opt.zero_grad()
        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)
        Local_prior_samples = torch.rand_like(Local_Environment_Features)
        Local_prior_score = F.softplus(-self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(self.Site_Prior(Local_Environment_Features))
        #Get prior loss per site
        Local_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Local_prior_loss = segment_csr(Local_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Local_prior_loss = Local_prior_loss.flatten().mean()
        Local_Environment_Loss = self.config["DIM_loss_local"]*Local_Environment_DIM_loss + self.config["Prior_loss_local"]*Local_prior_loss + self.config["KL_loss_local"]*Local_Environment_KL_loss
        self.manual_backward(Local_Environment_Loss)
        local_opt.step()

        local_task_opt.zero_grad()
        Local_Prediction = self.local_decoder(Local_Environment_Features.detach().clone())
        Local_MSE = torch.square(Local_Prediction.flatten() - (batch_dictionary["Site_Label_Tensor"]*self.site_label_scalers).flatten()).mean()
        self.manual_backward(Local_MSE)
        local_task_opt.step()


        #Adversarially train the prior discriminator
        local_prior_opt.zero_grad()
        Local_prior_score = F.softplus(self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(-self.Site_Prior(Local_Environment_Features.detach().clone()))
        #Get prior loss per site
        Site_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Site_prior_loss = segment_csr(Site_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Site_prior_loss = Site_prior_loss.flatten().mean()
        self.manual_backward(Site_prior_loss)
        local_prior_opt.step()

        #Perform a step on creating global environment representations, loss depends on mutual information and being able to trick the prior discriminator
        global_opt.zero_grad()
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        with torch.no_grad():
            false_sites = self.false_sample(Site_Features,0)
            false_interactions = self.false_sample(Interaction_Features,0)
            false_attention_mask = self.false_sample(Attention_Mask,0)
            false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask).detach().clone()
            false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask).detach().clone()

            Perturbed_Batch_Mask = dict(Batch_Mask)
            Perturbed_Batch_Mask["attention_j"] = Perturbed_Batch_Mask["attention_j"].detach().clone()
            Perturbed_Batch_Mask["attention_j"] = Perturbed_Batch_Mask["attention_j"][:,torch.randperm(Perturbed_Batch_Mask["attention_j"].shape[1])]

            false_locals_permuted = self.Site_DIM.inference(Site_Features,Interaction_Features,Attention_Mask,Perturbed_Batch_Mask).detach().clone()
            if self.config["extra_false_samples"] == True:
                engineered_false_locals_list = [false_locals_composition,false_locals_structure,false_locals_permuted]
            else:
                engineered_false_locals_list = []

        #Perform global DIM
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),engineered_false_locals_list,Batch_Mask,KL=KL)
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),Batch_Mask,KL=KL)
        Global_prior_samples = torch.rand_like(Global_Embedding_Features)
        Global_prior_score = F.softplus(-self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(self.Global_Prior(Global_Embedding_Features))
        Global_prior_loss = (Global_prior_score+Global_posterior_score).flatten().mean()
        Recon_Composition = self.Composition_Decoder(Global_Embedding_Features).clamp(0)
        Recon_Composition = Recon_Composition/(torch.sum(Recon_Composition,dim=1).unsqueeze(1).repeat(1,103)+10e-6)
        Composition_Loss = (-torch.sum(torch.min(Recon_Composition,batch_dictionary["Composition"]),1)).flatten().mean() + 1 #Half taxi cab distance for ternaries
        Global_Loss = self.config["DIM_loss_global"]*Global_DIM_loss + self.config["Prior_loss_global"]*Global_prior_loss + self.config["KL_loss_global"]*Global_KL_loss + self.config["Composition_Loss"]*Composition_Loss
        self.manual_backward(Global_Loss)
        global_opt.step()

        #Train the prior discriminator
        global_prior_opt.zero_grad()
        Global_prior_score = F.softplus(self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(-self.Global_Prior(Global_Embedding_Features.detach().clone()))
        Global_prior_loss_discrim = (Global_prior_score+Global_posterior_score).flatten().mean()
        self.manual_backward(Global_prior_loss_discrim)
        global_prior_opt.step()


        #Perform a step on predicting the band gap with the learnt global embedding
        task_opt.zero_grad()
        Prediction = self.decoder(Global_Embedding_Features.detach().clone())
        MSE = torch.square(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        self.manual_backward(MSE)
        task_opt.step()

        #"Local_Environment_KL_loss":Local_Environment_KL_loss,
        #"Global_KL_loss":Global_KL_loss,

        self.log_dict({"task_loss":MSE,"local_task_loss":Local_MSE,"Local_Environment_DIM_Loss":Local_Environment_DIM_loss,
        "Global_DIM_loss":Global_DIM_loss,"Local_prior_loss":Local_prior_loss,"Global_prior_loss":Global_prior_loss,"Local_KL_loss":Local_Environment_KL_loss,"Global_KL_loss":Global_KL_loss,"Comp_Loss":Composition_Loss},prog_bar=True)
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
        #Perform site deep infomax to obtain loss and embedding
        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False
        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)
        Local_Prediction = self.local_decoder(Local_Environment_Features.detach().clone())
        Local_MAE = torch.abs(Local_Prediction.flatten() - (batch_dictionary["Site_Label_Tensor"]*self.site_label_scalers).flatten()).mean()

        #Detach the local nevironment features and do independant deep infomax to convert local environment features to global features
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        false_sites = self.false_sample(Site_Features,0)
        false_interactions = self.false_sample(Interaction_Features,0)
        false_attention_mask = self.false_sample(Attention_Mask,0)
        false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask).detach().clone()
        false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask).detach().clone()

        Perturbed_Batch_Mask = dict(Batch_Mask)
        Perturbed_Batch_Mask["attention_j"] = Perturbed_Batch_Mask["attention_j"].detach().clone()
        Perturbed_Batch_Mask["attention_j"] = Perturbed_Batch_Mask["attention_j"][:,torch.randperm(Perturbed_Batch_Mask["attention_j"].shape[1])]

        false_locals_permuted = self.Site_DIM.inference(Site_Features,Interaction_Features,Attention_Mask,Perturbed_Batch_Mask).detach().clone()

        if self.config["extra_false_samples"] == True:
            engineered_false_locals_list = [false_locals_composition,false_locals_structure,false_locals_permuted]
        else:
            engineered_false_locals_list = []

        #Perform global DIM
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),engineered_false_locals_list,Batch_Mask,KL=KL)
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),Batch_Mask,KL=KL)
        Recon_Composition = self.Composition_Decoder(Global_Embedding_Features).clamp(0)
        Recon_Composition = Recon_Composition/(torch.sum(Recon_Composition,dim=1).unsqueeze(1).repeat(1,103)+10e-6)
        Composition_Loss = (-torch.sum(torch.min(Recon_Composition,batch_dictionary["Composition"]),1)).flatten().mean() + 1#Half taxi cab distance for ternaries
        #Composition_Loss = torch.absolute(Recon_Composition-batch_dictionary["Composition"]).flatten().mean() #taxi cab distance for ternaries

        #Try and perform shallow property prediction using the global representation as a sanity check
        Prediction = self.decoder(Global_Embedding_Features)
        MAE = torch.abs(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        return [MAE,Local_MAE,Local_Environment_DIM_loss,Local_Environment_KL_loss,Global_DIM_loss,Global_KL_loss,Composition_Loss]

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        #Local DIM optimizer
        local_opt = optim_dict[Optimizer_Config["Name"]](
            self.Site_DIM.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global DIM optimizer
        global_opt = optim_dict[Optimizer_Config["Name"]](
            self.Global_DIM.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global Task optimizer
        global_task_opt = optim_dict[Optimizer_Config["Name"]](
            self.decoder.parameters(),
            lr=0.001,
            **Optimizer_Config["Kwargs"],)
        #Local Task optimizer
        local_task_opt = optim_dict[Optimizer_Config["Name"]](
            self.local_decoder.parameters(),
            lr=0.001,
            **Optimizer_Config["Kwargs"],)
        #Local prior optimizer
        local_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Site_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global prior optimizer
        global_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Global_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)

        return local_opt,global_opt,global_task_opt,local_task_opt,local_prior_opt,global_prior_opt

    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss_task = torch.stack([i[0] for i in outputs]).mean()
        self.avg_loss_task_local = torch.stack([i[1] for i in outputs]).mean()
        self.Local_Environment_DIM_loss = torch.stack([i[2] for i in outputs]).mean()
        self.Local_Environment_KL_loss = torch.stack([i[3] for i in outputs]).mean()
        self.Global_DIM_loss = torch.stack([i[4] for i in outputs]).mean()
        self.Global_KL_loss = torch.stack([i[5] for i in outputs]).mean()
        self.Composition_Loss = torch.stack([i[6] for i in outputs]).mean()
        self.log("avg_val_loss_task", self.avg_loss_task)
        self.log("avg_loss_task_local", self.avg_loss_task_local)
        self.log("avg_val_loss_local_DIM",self.Local_Environment_DIM_loss)
        self.log("avg_val_loss_global_DIM",self.Global_DIM_loss)
        self.log("avg_val_loss_local_KL",self.Local_Environment_KL_loss)
        self.log("avg_val_loss_global_KL",self.Global_KL_loss)
        self.log("avg_val_Composition_Loss",self.Composition_Loss)

class SiteNet_DIM_supervisedcontrol(SiteNet_DIM):
    def __init__(
        self,
        config=None,
        freeze="Neither"

    ):
        super().__init__(config)
        self.freeze = freeze
        self.last_layer = nn.Sequential(nn.Linear(config["post_pool_layers"][-1],64),nn.Mish(),nn.Linear(64,1))

    #Inference mode, return the prediction
    def forward(self, b, batch_size=16,return_truth = False):
        with torch.no_grad():  
            self.eval()
            lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
            Encoding_list = []
            targets_list= []
            print("Inference in batches of %s" % batch_size)
            for inference_batch in tqdm(lob):
                batch_dictionary = collate_fn(inference_batch,inference=True)

                Attention_Mask = batch_dictionary["Attention_Mask"].to(self.device)
                Site_Features = batch_dictionary["Site_Feature_Tensor"].to(self.device)*self.site_label_scalers
                Atomic_ID = batch_dictionary["Atomic_ID"].to(self.device)
                Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"].to(self.device)
                Oxidation_State = batch_dictionary["Oxidation_State"].to(self.device)
                Batch_Mask = {i:j.to(self.device) for i,j in batch_dictionary["Batch_Mask"].items()}
                
                Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
                Local_Environment_Features = self.Site_DIM.inference(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)
                Global_Embedding_Features = self.Global_DIM.inference(Local_Environment_Features.detach().clone(),Batch_Mask)
                Prediction = self.last_layer(Global_Embedding_Features)
                Encoding_list.append(Prediction)
                targets_list.append(batch_dictionary["target"])
            Encoding = torch.cat(Encoding_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            if return_truth:
                return [Encoding,targets]
            else:
                return Encoding

    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        opt = self.optimizers()
        opt.zero_grad()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]

        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])

        if self.freeze == "Local" or self.freeze == "Both":
            Local_Environment_Features = self.Site_DIM.inference(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask).detach().clone()
        else:
            Local_Environment_Features = self.Site_DIM.inference(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)

        if self.freeze == "Global" or self.freeze == "Both":
            Global_Embedding_Features = self.Global_DIM.inference(Local_Environment_Features,Batch_Mask).detach().clone()
        else:
            Global_Embedding_Features = self.Global_DIM.inference(Local_Environment_Features,Batch_Mask)
        

        Prediction = self.last_layer(Global_Embedding_Features)
        MSE = torch.square(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        self.manual_backward(MSE)
        opt.step()
        opt.zero_grad()
        self.log_dict({"task_loss":MSE},prog_bar=True)

    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
        Local_Environment_Features = self.Site_DIM.inference(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)
        Global_Embedding_Features = self.Global_DIM.inference(Local_Environment_Features,Batch_Mask)
        Prediction = self.last_layer(Global_Embedding_Features)
        MAE = torch.abs(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        return [MAE]
    
    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss_task = torch.stack([i[0] for i in outputs]).mean()
        self.log("avg_val_loss_task", self.avg_loss_task)

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        #Local DIM optimizer
        opt = optim_dict[Optimizer_Config["Name"]](
            self.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)

        return opt
class SiteNet_DIM_regularisation(SiteNet_DIM):
    

    #Makes sure the model is in training mode, passes a batch through the model, then back propogates
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        opt = self.optimizers()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])

        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)

        #Perform a step on creating global environment representations, loss depends on mutual information and being able to trick the prior discriminator
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        #false_sites = self.false_sample(Site_Features,0)
        #false_interactions = self.false_sample(Interaction_Features,0)
        #false_attention_mask = self.false_sample(Attention_Mask,0)
        #false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask)
        #false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask)
        
        #Perform global DIM
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),false_locals_composition.detach().clone(),false_locals_structure.detach().clone(),Batch_Mask,KL=KL)
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features,Batch_Mask,KL=KL)

        Prediction = self.decoder(Global_Embedding_Features)
        MSE = torch.square(Prediction.flatten()[0] - batch_dictionary["target"].flatten()[0]) #Only the first sample in the dictionary is a "labeled" sample
        loss = MSE + self.config["DIM_loss_local"]*Local_Environment_DIM_loss + self.config["KL_loss_local"]*Local_Environment_KL_loss + self.config["DIM_loss_global"]*Global_DIM_loss + self.config["KL_loss_global"]*Global_KL_loss
        self.manual_backward(loss)
        
        if (batch_dictionary_idx + 1) % self.config["grad_accumulate"] == 0:
            opt.step()
            opt.zero_grad()
        #"Local_Environment_KL_loss":Local_Environment_KL_loss,
        #"Global_KL_loss":Global_KL_loss,

        self.log_dict({"task_loss":MSE,"Local_Environment_DIM_Loss":Local_Environment_DIM_loss,
        "Global_DIM_loss":Global_DIM_loss,"Local_KL_loss":Local_Environment_KL_loss,"Global_KL_loss":Global_KL_loss},prog_bar=True)
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]*self.site_feature_scalers
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
        #Perform site deep infomax to obtain loss and embedding
        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False
        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)
        #Detach the local nevironment features and do independant deep infomax to convert local environment features to global features
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        # false_sites = self.false_sample(Site_Features,0)
        # false_interactions = self.false_sample(Interaction_Features,0)
        # false_attention_mask = self.false_sample(Attention_Mask,0)
        # false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask)
        # false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask)

        #Get global DIM scores
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),false_locals_composition.detach().clone(),false_locals_structure.detach().clone(),Batch_Mask,KL=KL)
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),Batch_Mask,KL=KL)
        #Try and perform shallow property prediction using the global representation as a sanity check
        Prediction = self.decoder(Global_Embedding_Features)
        MAE = torch.abs(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        return [MAE,Local_Environment_DIM_loss,Local_Environment_KL_loss,Global_DIM_loss,Global_KL_loss]
    
    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss_task = torch.stack([i[0] for i in outputs]).mean()
        self.Local_Environment_DIM_loss = torch.stack([i[1] for i in outputs]).mean()
        self.Local_Environment_KL_loss = torch.stack([i[2] for i in outputs]).mean()
        self.Global_DIM_loss = torch.stack([i[3] for i in outputs]).mean()
        self.Global_KL_loss = torch.stack([i[4] for i in outputs]).mean()
        self.log("avg_val_loss_task", self.avg_loss_task)
        self.log("avg_val_loss_local_DIM",self.Local_Environment_DIM_loss)
        self.log("avg_val_loss_global_DIM",self.Global_DIM_loss)
        self.log("avg_val_loss_local_KL",self.Local_Environment_KL_loss)
        self.log("avg_val_loss_global_KL",self.Global_KL_loss)

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        #Local DIM optimizer
        opt = optim_dict[Optimizer_Config["Name"]](
            self.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)

        return opt

#Combines the local and global optimizer into a single optimizer    
class SiteNet_DIM_monooptimizer(SiteNet_DIM):
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        global_opt,task_opt,local_prior_opt,global_prior_opt = self.optimizers()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])

        #Perform a step on creating local environment representations while tricking the prior discriminator
        global_opt.zero_grad()
        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)
        Local_prior_samples = torch.rand_like(Local_Environment_Features)
        Local_prior_score = F.softplus(-self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(self.Site_Prior(Local_Environment_Features))
        #Get prior loss per site
        Local_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Local_prior_loss = segment_csr(Local_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Local_prior_loss = Local_prior_loss.flatten().mean()
        Local_Environment_Loss = self.config["DIM_loss_local"]*Local_Environment_DIM_loss + self.config["Prior_loss_local"]*Local_prior_loss + self.config["KL_loss_local"]*Local_Environment_KL_loss

        local_prior_opt.zero_grad()
        Local_prior_score = F.softplus(self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(-self.Site_Prior(Local_Environment_Features.detach().clone()))
        #Get prior loss per site
        Site_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Site_prior_loss = segment_csr(Site_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Site_prior_loss = Site_prior_loss.flatten().mean()
        self.manual_backward(Site_prior_loss)
        local_prior_opt.step()

        #Perform a step on creating global environment representations, loss depends on mutual information and being able to trick the prior discriminator
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        false_sites = self.false_sample(Site_Features,0)
        false_interactions = self.false_sample(Interaction_Features,0)
        false_attention_mask = self.false_sample(Attention_Mask,0)
        false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask)
        false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask)
        
        #Perform global DIM
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features,false_locals_composition,false_locals_structure,Batch_Mask,KL=KL)
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features,Batch_Mask,KL=KL)
        Global_prior_samples = torch.rand_like(Global_Embedding_Features)
        Global_prior_score = F.softplus(-self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(self.Global_Prior(Global_Embedding_Features))
        Global_prior_loss = (Global_prior_score+Global_posterior_score).flatten().mean()
        Recon_Composition = self.Composition_Decoder(Global_Embedding_Features).clamp(0)
        Recon_Composition = Recon_Composition/(torch.sum(Recon_Composition,dim=1).unsqueeze(1).repeat(1,103)+10e-6)
        Composition_Loss = (-torch.sum(torch.min(Recon_Composition,batch_dictionary["Composition"]),1)).flatten().mean() + 1 #Half taxi cab distance for ternaries
        Global_Loss = self.config["DIM_loss_global"]*Global_DIM_loss + self.config["Prior_loss_global"]*Global_prior_loss + self.config["KL_loss_global"]*Global_KL_loss + 0*Composition_Loss + Local_Environment_Loss
        self.manual_backward(Global_Loss)
        global_opt.step()

        #Train the prior discriminator
        global_prior_opt.zero_grad()
        Global_prior_score = F.softplus(self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(-self.Global_Prior(Global_Embedding_Features.detach().clone()))
        Global_prior_loss_discrim = (Global_prior_score+Global_posterior_score).flatten().mean()
        self.manual_backward(Global_prior_loss_discrim)
        global_prior_opt.step()


        #Perform a step on predicting the band gap with the learnt global embedding
        task_opt.zero_grad()
        Prediction = self.decoder(Global_Embedding_Features.detach().clone())
        MAE = torch.abs(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        self.manual_backward(MAE)
        task_opt.step()

        #"Local_Environment_KL_loss":Local_Environment_KL_loss,
        #"Global_KL_loss":Global_KL_loss,

        self.log_dict({"task_loss":MAE,"Local_Environment_DIM_Loss":Local_Environment_DIM_loss,
        "Global_DIM_loss":Global_DIM_loss,"Local_prior_loss":Local_prior_loss,"Global_prior_loss":Global_prior_loss,"Local_KL_loss":Local_Environment_KL_loss,"Global_KL_loss":Global_KL_loss,"Comp_Loss":Composition_Loss},prog_bar=True)
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
        #Perform site deep infomax to obtain loss and embedding
        if self.config["KL_loss_local"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False
        Local_Environment_Features,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask,KL=KL)
        #Detach the local nevironment features and do independant deep infomax to convert local environment features to global features
        if self.config["KL_loss_global"] > 0: #If the KL Loss isn't being trained it will inevitably cause NAN values, so it gets turned off
            KL = True
        else:
            KL = False

        #We create some synthetic local environments, the composition matches the target crystal but the distances are incorrect, and vice versa
        #false_sites = self.false_sample(Site_Features,0)
        #false_interactions = self.false_sample(Interaction_Features,0)
        #false_attention_mask = self.false_sample(Attention_Mask,0)
        #false_locals_composition = self.Site_DIM.inference(false_sites,Interaction_Features,Attention_Mask,Batch_Mask)
        #false_locals_structure = self.Site_DIM.inference(Site_Features,false_interactions,false_attention_mask,Batch_Mask)

        #Get global DIM scores
        Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features,false_locals_composition,false_locals_structure,Batch_Mask,KL=KL)
        #Global_Embedding_Features,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),Batch_Mask,KL=KL)
        Recon_Composition = self.Composition_Decoder(Global_Embedding_Features).clamp(0)
        Recon_Composition = Recon_Composition/(torch.sum(Recon_Composition,dim=1).unsqueeze(1).repeat(1,103)+10e-6)
        Composition_Loss = (-torch.sum(torch.min(Recon_Composition,batch_dictionary["Composition"]),1)).flatten().mean() + 1#Half taxi cab distance for ternaries
        #Composition_Loss = torch.absolute(Recon_Composition-batch_dictionary["Composition"]).flatten().mean() #taxi cab distance for ternaries

        #Try and perform shallow property prediction using the global representation as a sanity check
        Prediction = self.decoder(Global_Embedding_Features)
        MAE = torch.abs(Prediction.flatten() - batch_dictionary["target"].flatten()).mean()
        return [MAE,Local_Environment_DIM_loss,Local_Environment_KL_loss,Global_DIM_loss,Global_KL_loss,Composition_Loss]

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        #Global DIM optimizer
        global_opt = optim_dict[Optimizer_Config["Name"]](
            [{"params":self.Global_DIM.parameters()},{"params":self.Site_DIM.parameters()}],
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Task optimizer
        task_opt = optim_dict[Optimizer_Config["Name"]](
            self.decoder.parameters(),
            lr=0.001,
            **Optimizer_Config["Kwargs"],)
        #Local prior optimizer
        local_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Site_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global prior optimizer
        global_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Global_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)

        return global_opt,task_opt,local_prior_opt,global_prior_opt


class basic_callbacks(pl.Callback):
    def __init__(self,*pargs,filename = "current_model",**kwargs):
        super().__init__(*pargs,**kwargs)
        self.filename = filename + ".ckpt"
    
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, model):
        trainer.save_checkpoint("most_recent_complete_run.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint("current_model.ckpt")

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.filename)

############ DATA MODULE ###############

class DIM_h5_Data_Module(pl.LightningDataModule):
    def __init__(
        self,
        config,
        overwrite=False,
        ignore_errors=False,
        max_len=100,
        Dataset=None,
        cpus = 1,
        chunk_size = 32,
        multitaskmode_labels=False,
        seed="FIXED_SEED",
        **kwargs
    ):

        super().__init__()
        self.seed = seed
        self.batch_size = config["Batch_Size"]
        #In dynamic batching, the number of unique sites is the limit on the batch, not the number of crystals, the number of crystals varies between batches
        self.dynamic_batch = config["dynamic_batch"]
        self.Site_Features = config["Site_Features"]
        self.Site_Labels = config["Site_Labels"]
        self.Interaction_Features = config["Interaction_Features"]
        self.h5_file = config["h5_file"]
        self.overwrite = overwrite
        self.ignore_errors = ignore_errors
        self.limit = config["Max_Samples"]
        self.max_len = max_len
        self.cpus=cpus
        self.multitaskmode_labels = multitaskmode_labels
        if Dataset is None:
            self.Dataset = torch_h5_cached_loader(
                self.Site_Features,
                self.Site_Labels,
                self.Interaction_Features,
                self.h5_file,
                max_len=self.max_len,
                ignore_errors=self.ignore_errors,
                overwrite=self.overwrite,
                limit=self.limit,
                cpus=cpus,
                chunk_size=chunk_size,
                seed=self.seed
            )
        else:
            self.Dataset = Dataset

    def prepare_data(self):
        self.Dataset_Train, self.Dataset_Val = random_split(
            self.Dataset,
            [len(self.Dataset) - len(self.Dataset) // 20, len(self.Dataset) // 20],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        torch.manual_seed(hash(self.seed))#Makes sampling reproducable
        if self.dynamic_batch:
            if self.multitaskmode_labels:
                return DataLoader(
                    self.Dataset_Train,
                    collate_fn=collate_fn,
                    batch_sampler=Multitask_batch_sampler(SequentialSampler(self.Dataset_Train),self.batch_size,N_labels=self.multitaskmode_labels),
                    pin_memory=False,
                    num_workers=self.cpus,
                    prefetch_factor=8,
                    persistent_workers=True
                )
            else:
                return DataLoader(
                    self.Dataset_Train,
                    collate_fn=collate_fn,
                    batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Train),self.batch_size),
                    pin_memory=False,
                    num_workers=self.cpus,
                    prefetch_factor=8,
                    persistent_workers=True
                )
        else:
            return DataLoader(
                self.Dataset_Train,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )

    def val_dataloader(self):
        torch.manual_seed(hash(self.seed))#Makes sampling reproducable
        if self.dynamic_batch:
            return DataLoader(
                self.Dataset_Val,
                collate_fn=collate_fn,
                batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Val),self.batch_size),
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Val,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )
class SiteNet_batch_sampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
    def __iter__(self):
            #The VRAM required by the model in each batch is proportional to the number of unique atomic sites, or the length of the i axis
            #Keep extending the batch with more crystals until doing so again brings us over the batch size
            #If extending the batch would bring it over the max batch size, yield the current batch and seed a new one
            #Keep extending the batch with more crystals until doing so again brings us over the batch size
            #If extending the batch would bring it over the max batch size, yield the current batch and seed a new one
            sampler_iter = iter(self.sampler)
            batch = []
            size = 0
            idx = next(sampler_iter)
            while True:
                try:
                    #Check if this crystal brings us above the batch limit
                    if self.sampler.data_source[idx]["prim_size"] + size > self.batch_size:
                        yield batch
                        size = self.sampler.data_source[idx]["prim_size"]
                        batch = [idx]
                        idx = next(sampler_iter)
                    else:
                        size+= self.sampler.data_source[idx]["prim_size"]
                        batch.append(idx)
                        idx = next(sampler_iter)
                except StopIteration:
                    #Don't throw away the last batch if it isnt empty
                    if batch != []:
                        yield batch
                    #break to let lightning know the epoch is over
                    break

class Multitask_batch_sampler(Sampler):
    def __init__(self, sampler, batch_size,N_labels = 1000):
        self.sampler = sampler
        self.batch_size = batch_size
        self.chosen_labels = list(islice(iter(self.sampler),N_labels))
    def __iter__(self):
            #The VRAM required by the model in each batch is proportional to the number of unique atomic sites, or the length of the i axis
            #Keep extending the batch with more crystals until doing so again brings us over the batch size
            #If extending the batch would bring it over the max batch size, yield the current batch and seed a new one
            sampler_iter = cycle(filter(lambda x: x not in self.chosen_labels,iter(self.sampler)))
            shuffle(self.chosen_labels)
            chosen_labels = iter(self.chosen_labels)
            labels_idx = next(chosen_labels)
            idx = next(sampler_iter)
            batch = [labels_idx]
            size = self.sampler.data_source[labels_idx]["prim_size"]
            while True:
                try:
                    if self.sampler.data_source[idx]["prim_size"] + size > self.batch_size:
                        yield batch
                        labels_idx = next(chosen_labels)
                        size = self.sampler.data_source[labels_idx]["prim_size"]
                        batch = [labels_idx]
                    else:
                        size+= self.sampler.data_source[idx]["prim_size"]
                        batch.append(idx)
                        idx = next(sampler_iter)
                except StopIteration:
                    #Don't throw away the last batch if it isnt empty
                    if batch != []:
                        yield batch
                    #break to let lightning know the epoch is over
                    break
    def __len__(self):
        return len(self.chosen_labels)

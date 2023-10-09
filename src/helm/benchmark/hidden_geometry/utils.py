from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from einops import reduce



def get_instances_id(hidden_states,run_meta, algorithm = "2nn") -> np.ndarray:
    """
    Collect hidden states of all instances and compute ID
    we employ two different approaches: the one of the last token, the sum of all tokens
    Parameters
    ----------
    hidden_states: np.array(num_instances, num_layers, model_dim)
    algorithm: 2nn or gride --> what algorithm to use to compute ID

    Output
    ---------- 
    Dict np.array(num_layers)
    """
    assert algorithm in ["2nn", "gride"], "method must be 2nn or gride"
    # Compute ID
    id_per_layer = []
    layers = run_meta.num_layers
    for i in range(layers): #iterate over layers
        # (num_instances, model_dim)
        layer = Data(hidden_states[:,i,:])
        layer.remove_identical_points()
        if algorithm == "2nn":
            #id_per_layer.append(layer.compute_id_2NN()[0])
            raise NotImplementedError
        elif algorithm == "gride":
            id_per_layer.append(layer.return_id_scaling_gride(range_max = 1000)[0])
    return  np.stack(id_per_layer[1:]) #skip first layer - for some reason it outputs a wrong number of IDs

def hidden_states_collapse(instances_hiddenstates,method)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Output
    ----------
    (num_instances, num_layers, model_dim)
    """ 
    assert method in ["last", "sum"], "method must be last or sum"
    hidden_states = []
    for i in  instances_hiddenstates:
        #collect only test question tokens
        instance_hidden_states = i.hidden_states[:,-i.len_tokens_question:,:]
        if method == "last":
            hidden_states.append(instance_hidden_states[:,-1,:])
        elif method == "sum":
            hidden_states.append(reduce(instance_hidden_states, "l s d -> l d", "mean"))
            
    # (num_instances, num_layers, model_dim)
    hidden_states = torch.stack(hidden_states)
    return hidden_states.detach().cpu().numpy()


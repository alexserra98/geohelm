from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from einops import reduce
from collections import Counter



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
    for i in range(1,layers): #iterate over layers
        # (num_instances, model_dim)
        layer = Data(hidden_states[:,i,:])
        layer.remove_identical_points()
        if algorithm == "2nn":
            #id_per_layer.append(layer.compute_id_2NN()[0])
            raise NotImplementedError
        elif algorithm == "gride":
            id_per_layer.append(layer.return_id_scaling_gride(range_max = 1000)[0])
    return  np.stack(id_per_layer[1:]) #skip first layer - for some reason it outputs a wrong number of IDs

# def hidden_states_collapse(instances_hiddenstates,method)-> np.ndarray:
#     """
#     Collect hidden states of all instances and collapse them in one tensor
#     using the provided method

#     Parameters
#     ----------
#     instances_hiddenstates: List[HiddenGeometry]
#     method: last or sum --> what method to use to collapse hidden states

#     Output
#     ----------
#     (num_instances, num_layers, model_dim)
#     """ 
#     assert method in ["last", "sum"], "method must be last or sum"
#     hidden_states = []
#     for i in  instances_hiddenstates:
#         #collect only test question tokens
#         instance_hidden_states = i.hidden_states[:,-i.len_tokens_question:,:]
#         if method == "last":
#             hidden_states.append(instance_hidden_states[:,-1,:])
#         elif method == "sum":
#             hidden_states.append(reduce(instance_hidden_states, "l s d -> l d", "mean"))
            
#     # (num_instances, num_layers, model_dim)
#     hidden_states = torch.stack(hidden_states)
#     return hidden_states.detach().cpu().numpy()

def hidden_states_collapse(instances_hiddenstates,method)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    instances_hiddenstates: List[HiddenGeometry]
    method: last or sum --> what method to use to collapse hidden states

    Output
    ----------
    (num_instances, num_layers, model_dim)
    """ 
    assert method in ["last", "sum"], "method must be last or sum"
    hidden_states = []
    for i in  instances_hiddenstates:
        # if method == "sum":
        #     hidden_states.append(i.hidden_states)
        # elif method == "last":
        #     hidden_states.append(i.hidden_states[:,-1,:])
        hidden_states.append(i.hidden_states[method])
        
    # for some reason huggingface does not the full list of activations
    counter = Counter([i.shape[0] for i in hidden_states])
    # Find the most common element
    most_common_element = counter.most_common(1)[0][0]
    hidden_states = list(filter(lambda x: x.shape[0] == most_common_element, hidden_states))
    # (num_instances, num_layers, model_dim)
    #hidden_states = torch.stack(hidden_states)
    hidden_states = np.stack(hidden_states)
    #return hidden_states.detach().cpu().numpy()
    return hidden_states
def hidden_states_process(instance_hiddenstates: dict)-> dict:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    instances_hiddenstates: Dict.Keys: ["len_tokens_question", "hidden_states"] 
    Output
    ----------
    Dict.Keys: ["last", "sum"] --> (num_layers, model_dim)
    """ 
    len_tokens_question, hidden_states = instance_hiddenstates.values() 
    hidden_states = hidden_states.detach().cpu().numpy()
    out = {}
    out["last"] = hidden_states[:,-1,:]
    out["sum"] = reduce(hidden_states[:,-len_tokens_question:,:], "l s d -> l d", "mean")
    return out





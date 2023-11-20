from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from dataclasses import dataclass
from einops import reduce
from collections import Counter
from enum import Enum

@dataclass
class RunMeta():
  num_layers: int
  num_instances: int
  model_dim: int

class Match(Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    ALL = "all"
    
@dataclass
class InstanceHiddenSates():
  id: str 
  match: Match
  hidden_states: dict[str, np.ndarray]


def get_instances_id(hidden_states: np.ndarray ,run_meta: RunMeta, algorithm = "2nn") -> np.ndarray:
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
    assert algorithm == "gride", "gride is the only algorithm supported"
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

def hidden_states_collapse(instances_hiddenstates: list[InstanceHiddenSates], method: str, match: str)-> np.ndarray:
    """
    Collect hidden states of all instances and collapse them in one tensor
    using the provided method

    Parameters
    ----------
    instances_hiddenstates: List[InstanceHiddenSates]
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
        if i.match == match or match == "all":
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

def exact_match(instance, request_state):
    """
    Check if the generated answer is correct
    """
    gold = list(filter(lambda a: a.tags and a.tags[0] == "correct", instance.references))[0].output.text
    pred_index = request_state.result.completions[0].text.strip()
    pred = request_state.output_mapping.get(pred_index)
    if not pred:
        return "wrong"
    return "correct" if gold == pred else "wrong"

    # prompt = request_state.request.prompt
    # index_in_prompt = prompt.rfind("Question")
    # tokens_question = tokenizer(prompt[index_in_prompt:], return_tensors="pt", return_token_type_ids=False)
    # len_tokens_question = tokens_question["input_ids"].shape[1]
    # # generation
    # output = request_state.result.completions[0]
    # answer = tokenizer.decode(output.text).strip()
    # reference_index = request_state.output_mapping.get(answer, "incorrect")
    # if reference_index != "incorrect":
    #     result = list(filter(lambda a: a.output.text == reference_index, request_state.instance.references))
    #     if result and result[0].tags and result[0].tags[0] == "correct":
    #         return True
    # return False



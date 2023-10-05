from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch



def get_instances_id(hidden_states, algorithm = "2nn") -> np.ndarray:
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
    layers = hidden_states.shape[1]
    for i in range(layers): #iterate over layers
        # (num_instances, model_dim)
        layer = Data(hidden_states[:,i,:])
        layer.remove_identical_points()
        if algorithm == "2nn":
            id_per_layer.append(layer.compute_id_2NN()[0])
        elif algorithm == "gride":
            id_per_layer.append(layer.return_id_scaling_gride(range_max = 1000)[0])
    return  np.asarray(id_per_layer)


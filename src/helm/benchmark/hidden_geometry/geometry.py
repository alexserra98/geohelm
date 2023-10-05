from dadapy.data import Data
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import numpy as np
import torch
from helm.benchmark.adaptation.scenario_state import ScenarioState
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Set, Type
from einops import reduce
import einsum
import torch.nn.functional as F
import functools
from sklearn.neighbors import NearestNeighbors
from .utils import get_instances_id

K = 50

# Define a type hint 
Array = Type[np.ndarray]
Tensor = Type[torch.Tensor]

@dataclass
class InstanceHiddenSates():
  id: str 
  len_tokens_question : int
  hidden_states: Tensor


class RunGeometry():
  """
  RunGeometry stores the hidden states of all instances with some metadata
  and collect methods to compute geometric information of the representations
  
  Methods
  ----------
  _get_instances_hidden_states: collect hidden states of all instances
  _get_instances_id: compute the ID of all instances
  hidden_states_collapse: collect hidden states of all instances and collapse them in one tensor
  nearest_neighbour: compute the nearest neighbours of each instance in the run per layer
  """

  def __init__(self, scenario_state: ScenarioState):
    self.scenario_state = scenario_state
    self._instances_hiddenstates = self._get_instances_hidden_states()
    self.max_train_instances = self.scenario_state.adapter_spec.max_train_instances
    self._instances_id = {}
    
  
  def _get_instances_hidden_states(self) -> List[InstanceHiddenSates]:  
    """
    Instantiate a list of InstanceHiddenSates
    """
    instances_hiddenstates = []
    for instance, request_state in zip(self.scenario_state.instances, self.scenario_state.request_states):
      # hd --> (num_layers, num_tokens, model_dim)
      hd = request_state.result.completions[0].hidden_states["hidden_states"].detach().cpu()
      id = instance.id
      len_tokens_question = request_state.result.completions[0].hidden_states["len_tokens_question"]
      instances_hiddenstates.append(InstanceHiddenSates(id,len_tokens_question, hd))
    return instances_hiddenstates
  
  @property
  def instances_id(self) -> Dict:
    for emb_processing in ["last", "sum"]:
      for algorithm in ["2nn", "gride"]:
        key = emb_processing+" "+algorithm
        self._instances_id[key] = get_instances_id(self.hidden_states_collapse(emb_processing), algorithm)
    return self._instances_id
  
  
  def hidden_states_collapse(self, method)-> Array:
      """
      Collect hidden states of all instances and collapse them in one tensor
      using the provided method

      Output
      ----------
      (num_instances, num_layers, model_dim)
      """ 
      assert method in ["last", "sum"], "method must be last or sum"
      hidden_states = []
      for i in self._instances_hiddenstates:
        #collect only test question tokens
        instance_hidden_states = i.hidden_states[:,-i.len_tokens_question:,:]
        if method == "last":
          hidden_states.append(instance_hidden_states[:,-1,:])
        elif method == "sum":
          hidden_states.append(reduce(instance_hidden_states, "l s d -> l d", "mean"))
      # (num_instances, num_layers, model_dim)
      hidden_states = torch.stack(hidden_states)
      return hidden_states.detach().cpu().numpy()

  def nearest_neighbour(self, method: str, k: int) -> List[Array]:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided method
    Output
    ----------
    Array(num_layers, num_instances, k_neighbours)
    """
    hidden_states = self.hidden_states_collapse(method)
    assert k <= hidden_states.shape[0], "K must be smaller than the number of instances"
    layers = hidden_states.shape[1]
    neigh_matrix_list = []
    for i in range(layers):
      neigh = NearestNeighbors(n_neighbors=k)
      neigh.fit(hidden_states[:,i,:])
      dist, indices = neigh.kneighbors(hidden_states[:,i,:])
      indices = np.delete(indices, 0, 1) # removing the first column which is the instance itself
      neigh_matrix_list.append(indices)
    
    return np.stack(neigh_matrix_list)
     
  def get_instances_id(self) -> Dict:
    return self._instances_id
  
  def get_instances_hiddenstates(self) -> List[InstanceHiddenSates]:
    return self._instances_hiddenstates


class Geometry():
  """
  Geometry stores all the runs in a version and collect methods to compute geometric information of the representations
  across runs
  Methods
  ----------
  _instances_overlap: compute the overlap between two representations
  neig_overlap: compute the overlap between two representations
  """
  def __init__(self, runs: List[RunGeometry]) -> None:
    self.runs = runs
    
  def _instances_overlap(self,run1, run2, method, k) -> Array:
    nn1 = run1.nearest_neighbour(method, k=k)
    nn2 = run2.nearest_neighbour(method, k=k)
    assert nn1.shape == nn2.shape, "The two nearest neighbour matrix must have the same shape" 
    layers_len = nn1.shape[0]
    overlaps = np.empty([layers_len, layers_len])
    for i in range(layers_len):
      for j in range(layers_len):
        overlaps[i][j] = self.neig_overlap(nn1[i], nn2[j], K=k)
    return overlaps
  
  def neig_overlap(self, X, Y, K = 10):
    """
    Computes the neighborhood overlap between two representations.
    Parameters
    ----------
    X : 2D array of ints
        nearest neighbor index matrix of the first representation
    Y : 2D array of ints
        nearest neighbor index matrix of the second representation
    k : int
        number of nearest neighbors used to compute the overlap

    Returns
    -------
    overlap : float
        neighborhood overlap between the two representations
    """
    assert X.shape[0] == Y.shape[0]
    ndata = X.shape[0]
    iter = map(lambda x,y : np.intersect1d(x,y).shape[0]/K, X,Y)
    out = functools.reduce(lambda x,y: x+y, iter)
    return out/ndata
  

        



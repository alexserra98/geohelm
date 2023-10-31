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
from .utils import get_instances_id, hidden_states_collapse
import tqdm

# Define a type hint 
Array = Type[np.ndarray]
Tensor = Type[torch.Tensor]

@dataclass
class InstanceHiddenSates():
  id: str 
  hidden_states: dict

@dataclass
class RunMeta():
  num_layers: int
  num_instances: int
  model_dim: int


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

  def __init__(self, scenario_state: ScenarioState | None = None, instaces_hiddenstates: List[InstanceHiddenSates] | None = None):
    self.scenario_state = scenario_state
    self._instances_hiddenstates = instaces_hiddenstates
    self._hidden_states = self.set_hidden_states(self._instances_hiddenstates)
    self.run_meta = self._set_run_meta()
    #self.max_train_instances = self.scenario_state.adapter_spec.max_train_instances
    self._instances_id = self.instances_id_set()
    self._dict_nn = self.set_dict_nn(k=int(self.run_meta.num_instances*0.8))
    

  def _set_run_meta(self):
    num_layers = self.hidden_states["last"].shape[1]
    num_instances = self.hidden_states["last"].shape[0]
    model_dim = self.hidden_states["last"].shape[2]
    return RunMeta(num_layers, num_instances, model_dim)
  
  def _get_instances_hidden_states(self, scenario_state) -> List[InstanceHiddenSates]:  
    """
    Instantiate a list of InstanceHiddenSates
    """
    instances_hiddenstates = []
    for instance, request_state in zip(scenario_state.instances, scenario_state.request_states):
      # hd --> (num_layers, num_tokens, model_dim)
      hd = request_state.result.completions[0].hidden_states
      id = instance.id
      instances_hiddenstates.append(InstanceHiddenSates(id, hd))
    return instances_hiddenstates
  
  @property
  def hidden_states(self):
    """
    Output
    ----------
    (num_instances, num_layers, model_dim)
    """ 
    return self._hidden_states
  
  def set_hidden_states(self, instances_hiddenstates = None):
    if instances_hiddenstates is None:
      instances_hiddenstates = self._get_instances_hidden_states(self.scenario_state)
    hidden_states = {}
    for method in ["last", "sum"]:
      hidden_states[method] = hidden_states_collapse(instances_hiddenstates, method)
    return hidden_states

  @property
  def instances_id(self):
    return self._instances_id

 
  def instances_id_set(self):
    """
    Compute the ID of all instances

    Output
    ----------
    Dict[str, np.array(num_layers)]
    """
    id = {}
    for method in ["last", "sum"]:
      for algorithm in ["gride"]: #2nn no longer supported
        id[method] = get_instances_id(self.hidden_states[method], self.run_meta, algorithm)
    
    return id
  
  def nearest_neighbour(self, method: str, k: int) -> np.ndarray:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided method
    Output
    ----------
    Array(num_layers, num_instances, k_neighbours)
    """
    hidden_states = self.hidden_states[method]
    assert k <= hidden_states.shape[0], "K must be smaller than the number of instances"
    layers = self.run_meta.num_layers
    neigh_matrix_list = []
    for i in range(layers):
      neigh = NearestNeighbors(n_neighbors=k)
      neigh.fit(hidden_states[:,i,:])
      dist, indices = neigh.kneighbors(hidden_states[:,i,:])
      indices = np.delete(indices, 0, 1) # removing the first column which is the instance itself
      neigh_matrix_list.append(indices)
    
    return np.stack(neigh_matrix_list)

  @property
  def dict_nn(self):
    return self._dict_nn
  
 
  def set_dict_nn(self, k):
    """
    Compute the nearest neighbours of each layer
    with k --> [1, K, num_instances]
    
    Output
    ----------
    Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    dict_nn = {}
    for method in ["last", "sum"]:
      key = method
      dict_nn[key] = self.nearest_neighbour(method, k)
    return dict_nn
  
      
class Geometry():
  """
  Geometry stores all the runs in a version and collect methods to compute geometric information of the representations
  across runs
  Methods
  ----------
  _instances_overlap: compute the overlap between two representations
  neig_overlap: compute the overlap between two representations
  """

  def __init__(self, nearest_neig: List[Dict]) -> None:
    """
    Parameters
    ----------
    nearest_neig : List[Dict]
        List of dictionaries containing the nearest neighbours of each layer of the two runs
        Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    self.nearest_neig = nearest_neig


  def get_all_overlaps(self) -> Dict:
    """
    Compute the overlap between all the runs
    Output
    ----------
    Dict[k-method, np.ndarray[i,j, Array(num_layers, num_layers)]]
    """
    overlaps = {}
    num_runs = len(self.nearest_neig)
    num_layers = self.nearest_neig[0]["last"].shape[0]
    for method in ["last", "sum"]:
      overlaps[method] = {}
      desc = "Computing overlap for method " + method
      for i in tqdm.tqdm(range(num_runs-1), desc = desc):
        for j in range(i+1,num_runs):
          # indexes are not 0-based because it's easier to mange with plolty sub_trace
          overlaps[method][(i+1,j+1)] = self._instances_overlap(i,j,method)
    return overlaps
    
  def _instances_overlap(self, i,j,method) -> np.ndarray:
    nn1 = self.nearest_neig[i][method]
    nn2 = self.nearest_neig[j][method]
    assert nn1.shape == nn2.shape, "The two nearest neighbour matrix must have the same shape" 
    layers_len = nn1.shape[0]
    overlaps = np.empty([layers_len, layers_len])
    for i in range(layers_len):
      for j in range(layers_len):
        # WARNING : the overlap is computed with K=K THE OTHER OCC IS IN RUNGEOMETRY
        overlaps[i][j] = self.neig_overlap(nn1[i], nn2[j])
    return overlaps
  
  def neig_overlap(self, X, Y):
    """
    Computes the neighborhood overlap between two representations.
    Parameters
    ----------
    X : 2D array of ints
        nearest neighbor index matrix of the first representation
    Y : 2D array of ints
        nearest neighbor index matrix of the second representation
    
    Returns
    -------
    overlap : float
        neighborhood overlap between the two representations
    """
    assert X.shape[0] == Y.shape[0]
    ndata = X.shape[0]
    # Is this correct?
    k = X.shape[1]
    iter = map(lambda x,y : np.intersect1d(x,y).shape[0]/k, X,Y)
    out = functools.reduce(lambda x,y: x+y, iter)
    return out/ndata
  

        



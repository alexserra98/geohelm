import pickle
import pickle
from helm.benchmark.hidden_geometry.geometry import Geometry, RunGeometry
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Instance
import os
import numpy as np
import plotly.express as px  # (version 4.7.0 or higher)
import pandas as pd

path_version = "/u/dssc/zenocosini/helm_tests/benchmark_output/runs/v1"
path = "/mmlu:subject=anatomy,method=multiple_choice_joint,model=huggingface_gpt2,max_train_instances=3/"
path_intdim = os.path.join(path_version+path, "int_dim.pkl")

files = os.listdir(path_version)
files = list(filter(lambda x: x!= "eval_cache", files))
hidden_geometry = {}
hidden_states = {}
for k,i in enumerate(files):
    path_run = os.path.join(path_version, i)
    with open(path_run+"/hidden_geometry.pkl", "rb") as f:
        hidden_geometry[k] = pickle.load(f)

hidden_geometry[0].instances_id
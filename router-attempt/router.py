import os
import yaml
import numpy as np
import torch.nn as nn
import copy
from sklearn.svm import SVC
from PIL import Image


#import model loader here - also make split between 

class Router(nn.Module):

    def _init_(self, model: nn.Module, yaml_path: str | None = None, resources=None):
     super().__init__() 
     self.model = model
     self.resources = resources
     self.cfg = {}
     self.metric_weights = []
     if yaml_path is not None:
                 if not os.path.exists(yaml_path):
                     raise FileNotFoundError(f"YAML file not found: {yaml_path}")
     
                 with open(yaml_path, "r", encoding="utf-8") as f:
                     self.cfg = yaml.safe_load(f)
     
                 # Compute project root (two levels up from models/) ->potentially fix this
                 project_root = os.path.abspath((os.path.dirname(__file__)))
     
                 # Load data via DataLoader (side-effect: attach datasets to `self`) - > remove/replace this
                 loader = DataLoader(project_root)
                 loader.load_data(self.cfg, self)
     
                 # Load metric weights if provided
                 weights_dict = self.cfg.get("metric", {}).get("weights", {})
                 self.metric_weights = list(weights_dict.values())
     dummy_model = nn.Identity()
     self.model = dummy_model
     svm_params = self.cfg["hparam"]
     self.svm_model = SVC(**svm_params)
     
     # Select best-performing model for each query  - fix selves here
    routing_best = self.routing_data_train.loc[
    self.routing_data_train.groupby("query")["performance"].idxmax()
    ].reset_index(drop=True)
     
             # Prepare training data
    query_embedding_id = routing_best["embedding_id"].tolist()
    self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
    self.model_name_list = routing_best["model_name"].tolist()


     
    def determine_final_model(self, image: Image.Image, prompt: str):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        #fix this section here
        self.svm_model = load_model(load_model_path)
        
        # Compute embedding and predict - fix longformer section
        query_embedding = [get_longformer_embedding(query["query"]).numpy()]
        model_name = self.svm_model.predict(query_embedding)[0]
        return model_name

    def determine_final_model_mult(self, image: str, prompt:str):
         return model_name



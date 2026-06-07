import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Dict
from games.encoder.GraphEncoder import HeteroGNNEncoderPong
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from games.encoder.GraphEncoder import HeteroGNNEncoderPong, GraphEncoderFreeway, GraphEncoderPacman, GraphEncoderBreakout
from games.model.hetero_gnn import HeteroGNN
import torch_geometric as pyg
from games.model.cnn_model import CNNgame

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomMLPExtractor(nn.Module):
    def __init__(self, features_dim):
        super(CustomMLPExtractor, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Assuming 3 actions in the environment
        )

    def forward(self, gnn_embeddings):
        # Apply the shared MLP to the embeddings
        shared_embeddings = self.shared_mlp(gnn_embeddings)

        # Compute value
        value = self.value_mlp(shared_embeddings).mean(dim=0, keepdim=True)  # Aggregate object embeddings to a single value

        # Compute policy logits
        policy_logits = self.policy_mlp(shared_embeddings.mean(dim=0))  # Aggregate and compute logits for actions

        return value, policy_logits

    def forward_actor(self, gnn_embeddings):
        # Apply the shared MLP to the embeddings
        shared_embeddings = self.shared_mlp(gnn_embeddings)

        # Compute policy logits
        policy_logits = self.policy_mlp(shared_embeddings.mean(dim=0))  # Aggregate and compute logits for actions

        return policy_logits

    def forward_critic(self, gnn_embeddings):
        # Apply the shared MLP to the embeddings
        shared_embeddings = self.shared_mlp(gnn_embeddings)

        # Compute value
        value = self.value_mlp(shared_embeddings).mean(dim=0, keepdim=True)  # Aggregate object embeddings to a single value

        return value


class CustomHeteroGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, hidden_size=64, num_layer=2, obj_type_id='obj', arity_dict={'atom': 2}, game = 'pong'):
        super().__init__(observation_space, features_dim=features_dim)
        if game == 'pong':
            self.encoder = HeteroGNNEncoderPong()
        elif game == 'freeway':
            self.encoder = GraphEncoderFreeway() 
        elif game == 'pacman':
            self.encoder = GraphEncoderPacman()
            self.model = HeteroGNN(hidden_size, num_layer, obj_type_id, arity_dict, input_size=11)
        elif game == 'breakout':
            self.encoder = GraphEncoderBreakout()
        
        # set device to mps if available
        #self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = HeteroGNN(hidden_size, num_layer, obj_type_id, arity_dict, input_size=12).to(self.device)
        self.model = HeteroGNN(hidden_size, num_layer, obj_type_id, arity_dict, input_size=7).to(self.device)

    def forward(self, observations):
        # Encode observations to a graph using the encoder
        pyg_data = self.encoder.encode(observations)
        # if observations.shape[0] >1:
        #     print(f"Time to encode: {time.time() - start}")
        pyg_data = pyg_data.to(self.device) 
        obj_emb = self.model(pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.batch_dict)
        # Flatten or pool the embeddings if necessary to match the expected features_dim
        return obj_emb



import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

# Checking the output dimension right before feeding into the Linear layer
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = 4  # should be 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.adjust_to_features_dim = nn.Linear(2560, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #check first dimension of the input tensor
        # if observations.shape[0] >=1:
        #     # take the last observation
        #     observations = observations[-1] 
        
        cnn_output = self.cnn(observations)
        final_output = self.adjust_to_features_dim(cnn_output)
        return final_output
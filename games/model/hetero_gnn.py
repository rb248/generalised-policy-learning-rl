# from collections import defaultdict
# from typing import Dict, List, Optional

# import torch
# import torch_geometric as pyg
# import wandb
# from lightning import LightningModule
# from torch import Tensor
# from torch.nn.modules.loss import L1Loss, MSELoss, _Loss
# from torch_geometric.typing import Adj

# from games.model.hetero_message_passing import FanInMP, FanOutMP
# from torch_geometric.nn import GCNConv, GATConv

# class HeteroGNN(torch.nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         num_layer: int,
#         obj_type_id: str,
#         arity_dict: Dict[str, int],
#         aggr: Optional[str | pyg.nn.aggr.Aggregation] = "sum",
#         input_size: int = 7,
#     ):
#         super().__init__()

#         self.hidden_size: int = hidden_size
#         self.num_layer: int = num_layer
#         self.obj_type_id: str = obj_type_id
#         self.arity_dict: Dict[str, int] = arity_dict
#         self.encoding_mlps = torch.nn.ModuleDict()
#         self.encoding_mlps[obj_type_id] = self.mlp(input_size, hidden_size, hidden_size)
#         for pred, arity in arity_dict.items():
#             if arity > 0:
#                 self.encoding_mlps[pred] = self.mlp(input_size*arity, hidden_size * arity, hidden_size * arity)

#         mlp_dict = {
#             pred: HeteroGNN.mlp(
#                 input_size * arity, hidden_size * arity, input_size * arity
#             )
#             for pred, arity in arity_dict.items()
#             if arity > 0
#         } 
#         self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id, aggr=aggr)

#         self.obj_update = HeteroGNN.mlp(
#             in_size=2 * hidden_size, hidden_size=2 * hidden_size, out_size=hidden_size
#         )

#         self.atom_to_obj = FanInMP(
#             hidden_size=input_size,
#             dst_name=obj_type_id,
#             aggr=aggr,
#         )


#     def encoding_layers(self, x_dict: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
#         # Resize everything by the hidden_size
#         for k, v in x_dict.items():
#             assert v.dim() == 2
#             x_dict[k] = self.encoding_mlps[k](v.to(device))
#         return x_dict

#     def layer(self, x_dict, edge_index_dict):
#         # Groups object embeddings that are part of an atom and
#         # applies predicate-specific MLP based on the edge type.
#         out = self.obj_to_atom(x_dict, edge_index_dict, self.arity_dict)
#         x_dict.update(out)  # update atom embeddings
#         # Distribute the atom embeddings back to the corresponding objects.
#         x_dict = self.atom_to_obj(x_dict, edge_index_dict, self.arity_dict)
#         # Update the object embeddings using a shared update-MLP.
#         # obj_emb = torch.cat([x_dict[self.obj_type_id], out[self.obj_type_id]], dim=1)
#         # obj_emb = self.obj_update(obj_emb)
#         # x_dict[self.obj_type_id] = obj_emb

#     def forward(
#         self,
#         x_dict: Dict[str, Tensor],
#         edge_index_dict: Dict[str, Adj],
#         batch_dict: Optional[Dict[str, Tensor]] = None,
#     ):  
#         #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
#         # Filter out dummies and move to device
#         x_dict = {k: v.to(device) for k, v in x_dict.items() if v.numel() != 0}
#         edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items() if v.numel() != 0}
        
#         # Resize everything by the hidden_size
#         #x_dict = self.encoding_layers(x_dict, device)
#         # print("After encoding layers:", x_dict)

#         for _ in range(self.num_layer):
#             self.layer(x_dict, edge_index_dict)

#         obj_emb = x_dict[self.obj_type_id].to(device)
#         batch = (
#             batch_dict[self.obj_type_id].to(device)
#             if batch_dict is not None
#             else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=device)
#         )
        
#         # Aggregate all object embeddings into one aggregated embedding
#         aggr = pyg.nn.global_add_pool(obj_emb, batch)  # shape [hidden, 1]
#         # print("After aggregation:", aggr)
        
#         # Produce final single scalar of shape [1]
#         return aggr

#     @staticmethod
#     def mlp(in_size: int, hidden_size: int, out_size: int):
#         return pyg.nn.MLP([in_size, hidden_size, out_size], norm=None, dropout=0.0)


# class HeteroGCN(torch.nn.Module):
#     def __init__(self, in_channels_dict, out_channels):
#         super(HeteroGCN, self).__init__()
#         self.convs = torch.nn.ModuleDict()
#         for obj_type, in_channels in in_channels_dict.items():
#             self.convs[obj_type] = GCNConv(in_channels, out_channels)

#     def forward(self, data):
#         for obj_type in data.node_types:
#             x = data[obj_type].x
#             edge_index = data[obj_type, 'to', obj_type].edge_index
#             x = self.convs[obj_type](x, edge_index)
#             x = F.relu(x)
#             data[obj_type].x = x

#         return data

# class HeteroGAT(torch.nn.Module):
#     def __init__(self, in_channels_dict, out_channels):
#         super(HeteroGAT, self).__init__()
#         self.convs = torch.nn.ModuleDict()
#         for obj_type, in_channels in in_channels_dict.items():
#             self.convs[obj_type] = GATConv(in_channels, out_channels)

#     def forward(self, data):
#         for obj_type in data.node_types:
#             x = data[obj_type].x
#             edge_index = data[obj_type, 'to', obj_type].edge_index
#             x = self.convs[obj_type](x, edge_index)
#             x = F.relu(x)
#             data[obj_type].x = x

#         return data 

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch_geometric as pyg
import wandb
from lightning import LightningModule
from torch import Tensor
from torch.nn.modules.loss import L1Loss, MSELoss, _Loss
from torch_geometric.typing import Adj

from games.model.hetero_message_passing import FanInMP, FanOutMP
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import AttentionalAggregation

class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layer: int,
        obj_type_id: str,
        arity_dict: Dict[str, int],
        aggr: Optional[str | pyg.nn.aggr.Aggregation] = "sum",
        input_size: int = 7,
    ):
        """
        :param hidden_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param obj_type_id: The type identifier of objects in the x_dict.
        :param arity_dict: A dictionary mapping predicate names to their arity.
        Creates one MLP for each predicate.
        Note that predicates as well as goal-predicates are meant.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_layer: int = num_layer
        self.obj_type_id: str = obj_type_id

        # Initialize encoding MLPs
        self.encoding_mlps = torch.nn.ModuleDict()
        self.encoding_mlps[obj_type_id] = self.mlp(input_size, hidden_size, hidden_size)  # Assuming initial input size is 7
        for pred, arity in arity_dict.items():
            if arity > 0:
                self.encoding_mlps[pred] = self.mlp(input_size*arity, hidden_size * arity, hidden_size * arity)  # Adjust initial input size as needed

        mlp_dict = {
            # One MLP per predicate (goal-predicates included)
            # For a predicate p(o1,...,ok) the corresponding MLP gets k object
            # embeddings as input and generates k outputs, one for each object.
            pred: HeteroGNN.mlp(
                hidden_size * arity, hidden_size * arity, hidden_size * arity
            )
            for pred, arity in arity_dict.items()
            if arity > 0
        }
        self.obj_to_atom = FanOutMP(mlp_dict, src_name=obj_type_id)

        self.obj_update = HeteroGNN.mlp(
            in_size=2 * hidden_size, hidden_size=2 * hidden_size, out_size=hidden_size
        )

        # Messages from atoms flow to objects
        self.atom_to_obj = FanInMP(
            hidden_size=hidden_size,
            dst_name=obj_type_id,
            aggr=aggr,
        )
        self.attn_aggregation = AttentionalAggregation(gate_nn=torch.nn.Linear(hidden_size, 1))



    def encoding_layers(self, x_dict: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
        # Resize everything by the hidden_size
        for k, v in x_dict.items():
            assert v.dim() == 2
            x_dict[k] = self.encoding_mlps[k](v.to(device))
        return x_dict

    def layer(self, x_dict, edge_index_dict):
        # Groups object embeddings that are part of an atom and
        # applies predicate-specific MLP based on the edge type.
        out = self.obj_to_atom(x_dict, edge_index_dict)
        x_dict.update(out)  # update atom embeddings
        # Distribute the atom embeddings back to the corresponding objects.
        out = self.atom_to_obj(x_dict, edge_index_dict)
        # Update the object embeddings using a shared update-MLP.
        obj_emb = torch.cat([x_dict[self.obj_type_id], out[self.obj_type_id]], dim=1)
        obj_emb = self.obj_update(obj_emb)
        x_dict[self.obj_type_id] = obj_emb


    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ):  
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        # Filter out dummies and move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items() if v.numel() != 0}
        
        # Resize everything by the hidden_size
        x_dict = self.encoding_layers(x_dict, device)

        for _ in range(self.num_layer):
            self.layer(x_dict, edge_index_dict)

        obj_emb = x_dict[self.obj_type_id].to(device)
        batch = (
            batch_dict[self.obj_type_id].to(device)
            if batch_dict is not None
            else torch.zeros(obj_emb.shape[0], dtype=torch.long, device=device)
        )
        
        # Aggregate all object embeddings into one aggregated embedding
        #aggr = pyg.nn.global_add_pool(obj_emb, batch)  # shape [hidden, 1]
        #
        # Produce final single scalar of shape [1]
        aggr = self.attn_aggregation(obj_emb, batch)
        return aggr

    @staticmethod
    def mlp(in_size: int, hidden_size: int, out_size: int):
        return pyg.nn.MLP([in_size, hidden_size, out_size], norm=None, dropout=0.0)


class HeteroGCN(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(HeteroGCN, self).__init__()
        self.convs = torch.nn.ModuleDict()
        for obj_type, in_channels in in_channels_dict.items():
            self.convs[obj_type] = GCNConv(in_channels, out_channels)

    def forward(self, data):
        for obj_type in data.node_types:
            x = data[obj_type].x
            edge_index = data[obj_type, 'to', obj_type].edge_index
            x = self.convs[obj_type](x, edge_index)
            x = F.relu(x)
            data[obj_type].x = x

        return data

class HeteroGAT(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(HeteroGAT, self).__init__()
        self.convs = torch.nn.ModuleDict()
        for obj_type, in_channels in in_channels_dict.items():
            self.convs[obj_type] = GATConv(in_channels, out_channels)

    def forward(self, data):
        for obj_type in data.node_types:
            x = data[obj_type].x
            edge_index = data[obj_type, 'to', obj_type].edge_index
            x = self.convs[obj_type](x, edge_index)
            x = F.relu(x)
            data[obj_type].x = x

        return data
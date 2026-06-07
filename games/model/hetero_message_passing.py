# import abc
# from typing import Any, Dict, List, Optional, Union

# import torch
# import torch_geometric as pyg
# from torch import Tensor
# from torch_geometric.nn import Aggregation, SimpleConv, GATConv
# from torch_geometric.nn.conv.hetero_conv import group
# from torch_geometric.nn.module_dict import ModuleDict
# from torch_geometric.typing import Adj, EdgeType, OptPairTensor
# from torch import nn


# class HeteroRouting(torch.nn.Module):
#     """
#     Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
#     Instead of specifying a convolution for each EdgeType more generic rules can be used.
#     """

#     def __init__(self, aggr: Optional[str | Aggregation] = None, arity_dict: Optional[Dict[str, int]] = None) -> None:
#         super().__init__()
#         self.aggr = aggr
#         self.arity_dict = arity_dict or {}

#     @abc.abstractmethod
#     def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
#         pass

#     @abc.abstractmethod
#     def _internal_forward(self, x, edges_index, edge_type: EdgeType):
#         pass

#     def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
#         aggregated: Dict[str, Tensor] = {}
#         for key, value in out_dict.items():
#             if isinstance(self.aggr, Aggregation):
#                 out = torch.stack(value, dim=0)
#                 out = self.aggr(out, dim=0)
#                 if out.dim() == 3 and out.shape[0] == 1:
#                     out = out[0]  # TODO Why does Softmax return one dim to much
#             else:
#                 out = group(value, self.aggr)
#             aggregated[key] = out
#         return aggregated

# class FanOutMP(HeteroRouting):
#     def __init__(
#         self,
#         update_mlp_by_dst: Dict[str, torch.nn.Module],
#         src_name: str,
#         aggr: Optional[str | Aggregation] = None
#     ) -> None:
#         super().__init__(aggr=aggr)
#         self.update_mlp_by_dst = ModuleDict(update_mlp_by_dst)
#         self.src_name = src_name

#     def _accept_edge(self, src, rel, dst) -> bool:
#         return src == self.src_name

#     def _internal_forward(self, x, edge_index, edge_type: EdgeType):
#         src, rel, dst = edge_type
#         mlp = self.update_mlp_by_dst[dst]
#         src_features = x[0][edge_index[0]]
#         dst_features = x[1][edge_index[1]]
#         concatenated_features = torch.cat([src_features, dst_features], dim=-1)
#         out = mlp(concatenated_features)
#         return out

#     def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
#         for dst, value in out_dict.items():
#             stacked = torch.cat(value, dim=1)
#             out_dict[dst] = stacked
#         return out_dict
    
#     def forward(self, x_dict, edge_index_dict, arity_dict):
#         out_dict: Dict[str, Any] = {}
#         for pred, arity in arity_dict.items():
#             if arity > 0:  # Processing only predicates with defined arities
#                 rel_0_features = torch.tensor([], device=next(iter(x_dict.values())).device, dtype=torch.float32)
#                 rel_1_features = torch.tensor([], device=next(iter(x_dict.values())).device, dtype=torch.float32)
#                 edge_type_0 = (self.src_name, '0', pred)
#                 edge_type_1 = (self.src_name, '1', pred)
#                 if edge_index_dict.get(edge_type_0) is not None:
#                     edge_indices_0 = edge_index_dict[edge_type_0]
#                     src_indices_0 = edge_indices_0[0]
#                     src_features_0 = x_dict[self.src_name][src_indices_0]
#                     rel_0_features = torch.cat([rel_0_features, src_features_0], dim=0) 
#                     edge_indices_1 = edge_index_dict[edge_type_1]
#                     src_indices_1 = edge_indices_1[0]
#                     src_features_1 = x_dict[self.src_name][src_indices_1]
#                     rel_1_features = torch.cat([rel_1_features, src_features_1], dim=0)
#                     concat_features = torch.cat([rel_0_features, rel_1_features], dim=1)
#                     out = self.update_mlp_by_dst[pred](concat_features)
#                     out_dict[pred] = out
#         return out_dict
#         #         
# class FanInMP(HeteroRouting):
#     def __init__(
#         self,
#         hidden_size: int,
#         dst_name: str,
#         aggr: Optional[Union[str, Aggregation]] = "mean",  # Use "mean" for averaging
#     ) -> None:
#         super().__init__(aggr=aggr)
#         self.hidden_size = hidden_size
#         self.dst_name = dst_name
#         self.comb_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, hidden_size)
#         )

#     def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
#         return dst == self.dst_name

#     def _internal_forward(self, x, edges_index, edge_type):
#         src, rel, dst = edge_type
#         src_features = x[edges_index[0]]
#         if rel == '0':
#             selected_features = src_features[:, :self.hidden_size]
#         elif rel == '1':
#             selected_features = src_features[:, self.hidden_size:]
#         return selected_features

#     def forward(self, x_dict, edge_index_dict, arity_dict):
#         device = next(iter(x_dict.values())).device  # Get the device from the first tensor
#         obj_features = x_dict[self.dst_name]

#         # Initialize aggregated features and counts
#         aggregated_features = torch.zeros_like(obj_features, device=device)
#         count = torch.zeros(obj_features.size(0), device=device)  # Counter for averaging

#         # Process edges and aggregate features
#         for edge_type, edge_indices in edge_index_dict.items():
#             src, rel, dst = edge_type
#             if dst == self.dst_name:  # Only process edges where the object is the destination
#                 selected_features = self._internal_forward(x_dict[src], edge_indices, edge_type)
#                 # Use scatter_add for efficient aggregation
#                 aggregated_features.index_add_(0, edge_indices[1], selected_features)
#                 count.index_add_(0, edge_indices[1], torch.ones_like(edge_indices[1], dtype=torch.float, device=device))

#         # Normalize by the count to get the average
#         count = count.clamp(min=1).unsqueeze(1)  # Avoid division by zero
#         aggregated_features /= count

#         # Combine the original features with the aggregated features
#         combined_features = torch.cat([obj_features, aggregated_features], dim=-1)
#         out_features = self.comb_mlp(combined_features)

#         return {self.dst_name: out_features}

# class SelectMP(pyg.nn.MessagePassing):

#     def __init__(
#         self,
#         hidden_size: int,
#         aggr: Optional[str | List[str]] = "mean",
#         aggr_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         super().__init__(
#             aggr,
#             aggr_kwargs=aggr_kwargs,
#         )
#         self.hidden = hidden_size

#     def forward(
#         self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
#     ) -> Tensor:
#         if isinstance(x, Tensor):
#             x = (x, x)
#         return self.propagate(edge_index, x=x, position=position)

#     def message(self, x_j: Tensor, position: int) -> Tensor:
#         # Take the i-th hidden-number of elements from the last dimension
#         # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
#         # alternatively split = torch.split(x_j, self.hidden, dim=-1)
#         #               split[self.position]
#         sliced = x_j[:, position * self.hidden : (position + 1) * self.hidden]
#         return sliced

#     def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
#         # Ensure dim_size is correct
#         dim_size = inputs.size(0) if dim_size is None else dim_size
#         return super().aggregate(inputs, index, ptr, dim_size)

import abc
from typing import Any, Dict, List, Optional, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import Aggregation, SimpleConv
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, OptPairTensor


class HeteroRouting(torch.nn.Module):
    """
    Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
    Instead of specifying a convolution for each EdgeType more generic rules can be used.
    """

    def __init__(self, aggr: Optional[str | Aggregation] = None) -> None:
        super().__init__()
        self.aggr = aggr

    @abc.abstractmethod
    def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
        pass

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType):
        pass

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            # hetero_conv.group does not yet support Aggregation modules
            if isinstance(self.aggr, Aggregation):
                out = torch.stack(value, dim=0)
                out = self.aggr(out, dim=0)
                if out.dim() == 3 and out.shape[0] == 1:
                    out = out[0]  # TODO Why does Softmax return one dim to much
            else:
                out = group(value, self.aggr)
            aggregated[key] = out
        return aggregated

    def forward(self, x_dict, edge_index_dict):
        """
        Apply message passing to each edge_index key if the edge-type is accepted.
        Calls the internal forward with a normal homogenous signature of x, edge_index
        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
       
        out_dict: Dict[str, Any] = {}
        for edge_type in edge_index_dict.keys():
            src, rel, dst = edge_type

            if not self._accept_edge(src, rel, dst):
                continue

            if src == dst and src in x_dict:
                x = x_dict[src]
            elif src in x_dict or dst in x_dict:
                
                x = (
                    x_dict.get(src, None),
                    x_dict.get(dst, None),
                )
                
            else:
                raise ValueError(
                    f"Neither src ({src}) nor destination ({dst})"
                    + f" found in x_dict ({x_dict})"
                )

            out = self._internal_forward(x, edge_index_dict[edge_type], edge_type)
            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        return self._group_out(out_dict)


class FanOutMP(HeteroRouting):
    """
     Accepts EdgeTypes with the defined src_name.
    1. For each destination concatenate all embeddings of the source.
    2. Run the destination specific MLP on the concatenated embeddings.
    3. Save the new embedding under the destination key.
    FanOut should be aggregation free in theory.
    Every atom gets only as many messages as the arity of its predicate.
    :param update_mlp_by_dst: An MLP for each possible destination.
        Needs the degree of incoming edges as input and output dimension
    :param src_name: The node-type for which outgoing edges should be accepted.
    """

    def __init__(
        self,
        update_mlp_by_dst: Dict[str, torch.nn.Module],
        src_name: str,
    ) -> None:
        """ """
        super().__init__()
        self.update_mlp_by_dst = ModuleDict(update_mlp_by_dst)
        self.simple = SimpleConv()
        self.src_name = src_name

    def _accept_edge(self, src, rel, dst) -> bool:
        return src == self.src_name

    def _internal_forward(self, x, edge_index, edge_type: EdgeType):
        position = int(edge_type[1])
        
        # Debug: Print edge index and node features

        # Ensure edge indices are within bounds
        max_index = x.shape[0] if isinstance(x, torch.Tensor) else max(x[0].shape[0], x[1].shape[0])
        if edge_index.max() >= max_index:
            raise ValueError(f"Invalid edge index: {edge_index.max()} is out of bounds for tensor with size {max_index}")
        

        out = self.simple(x, edge_index)
        return position, out

    def _group_out(self, out_dict: Dict[str, List]) -> Dict[str, Tensor]:
        for dst, value in out_dict.items():
            sorted_out = sorted(value, key=lambda tpl: tpl[0])
            stacked = torch.cat([out for _, out in sorted_out], dim=1)
            out_dict[dst] = self.update_mlp_by_dst[dst](stacked)

        return out_dict


class FanInMP(HeteroRouting):

    def __init__(
        self,
        hidden_size: int,
        dst_name: str,
        aggr: Optional[Union[str, Aggregation]] = "sum",
    ) -> None:
        super().__init__(aggr)
        self.select = SelectMP(hidden_size)
        self.dst_name = dst_name

    def _accept_edge(self, src: str, rel: str, dst: str) -> bool:
        return dst == self.dst_name

    def _internal_forward(self, x, edges_index, edge_type):
        return self.select(x, edges_index, int(edge_type[1]))


class SelectMP(pyg.nn.MessagePassing):

    def __init__(
        self,
        hidden_size: int,
        aggr: Optional[str | List[str]] = "sum",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.hidden = hidden_size

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively split = torch.split(x_j, self.hidden, dim=-1)
        #               split[self.position]
        sliced = x_j[:, position * self.hidden : (position + 1) * self.hidden]
        return sliced


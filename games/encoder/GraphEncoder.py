import torch
import networkx as nx
from torch_geometric.data import HeteroData, Batch
from collections import defaultdict
from itertools import combinations, product
import os
import matplotlib.pyplot as plt
from typing import List

class HeteroGNNEncoderPong:
    def __init__(self, obj_type_id: str = "obj", atom_type_id: str = "atom"):
        self.obj_type_id = obj_type_id
        self.atom_type_id = atom_type_id

    def encode(self, batch_node_features: torch.Tensor, proximity_threshold: float = 1000) -> Batch:
        batch_data = []
        batch_size = batch_node_features.size(0)

        for b in range(batch_size):
            node_features = batch_node_features[b]
            num_nodes = node_features.size(0)
            graph = nx.Graph()

            # Adding object nodes
            for i in range(num_nodes):
                graph.add_node(i, type=self.obj_type_id, features=node_features[i].tolist())

            # Adding atom nodes
            atom_index = num_nodes
            object_feature_length = node_features.size(1)
            for i, j in combinations(range(num_nodes), 2):
                dist = torch.norm(node_features[i, :2] - node_features[j, :2]).item()
                if dist < proximity_threshold:
                    # Create atom node with a 2D zero vector of the shape (2, object_feature_length)
                    atom_features = torch.zeros((2, object_feature_length)).tolist()
                    graph.add_node(atom_index, type=self.atom_type_id, features=atom_features)
                    graph.add_edge(i, atom_index, position=0)
                    graph.add_edge(j, atom_index, position=1)
                    atom_index += 1
            
            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def to_pyg_data(self, batch_graphs):
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = {self.obj_type_id: {}, self.atom_type_id: {}}
            obj_features = []
            atom_features = []
            edge_dict = defaultdict(list)

            current_obj_features = []
            current_atom_features = []

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)
                elif node_type == self.atom_type_id:
                    node_index_mapping[node_type][node] = len(current_atom_features)
                    current_atom_features.append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))
            if current_atom_features:
                flattened_atom_features = [f.view(-1) for f in current_atom_features]
                atom_features.append(torch.stack(flattened_atom_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)
            if atom_features:
                data[self.atom_type_id].x = torch.cat(atom_features)

            for src, dst, attr in graph.edges(data=True):
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                pos = str(attr['position'])
                edge_type = (src_type, pos, dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, pos, src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list


class HeteroGNNEncoderPongProximity:
    def __init__(self, obj_type_ids: dict):
        self.obj_type_ids = obj_type_ids  # e.g., {'ball': 'ball', 'paddle': 'paddle'}

    def encode(self, batch_node_features: dict, proximity_threshold: float = 1000) -> Batch:
        batch_data = []
        batch_size = len(next(iter(batch_node_features.values())))  # Assume all have the same batch size

        for b in range(batch_size):
            graph = nx.Graph()

            for obj_type, features in batch_node_features.items():
                node_features = features[b]
                num_nodes = node_features.size(0)

                # Adding nodes of a specific type
                for i in range(num_nodes):
                    graph.add_node((obj_type, i), type=self.obj_type_ids[obj_type], features=node_features[i].tolist())

            # Adding edges based on proximity
            for obj_type1, features1 in batch_node_features.items():
                for obj_type2, features2 in batch_node_features.items():
                    if obj_type1 != obj_type2:
                        for i, j in product(range(features1[b].size(0)), range(features2[b].size(0))):
                            dist = torch.norm(features1[b][i, :2] - features2[b][j, :2]).item()
                            if dist < proximity_threshold:
                                graph.add_edge((obj_type1, i), (obj_type2, j))
            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def to_pyg_data(self, batch_graphs):
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = {obj_type: {} for obj_type in self.obj_type_ids.values()}
            node_features_dict = defaultdict(list)
            edge_dict = defaultdict(list)

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                node_index_mapping[node_type][node] = len(node_features_dict[node_type])
                node_features_dict[node_type].append(features)

            for node_type, features in node_features_dict.items():
                data[node_type].x = torch.stack(features)

            for src, dst in graph.edges:
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                edge_type = (src_type, 'to', dst_type)
                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))

                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, 'to', src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list
    
def visualize_graph(graph: nx.Graph):
        pos = nx.spring_layout(graph)
        plt.figure(figsize=(12, 8))
        node_labels = {node: f"{node}\n{attrs['type']}" for node, attrs in graph.nodes(data=True)}
        edge_labels = {(src, dst): f"{attrs['position']}" for src, dst, attrs in graph.edges(data=True)}

        nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=500, node_color='lightblue', font_size=10)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

        plt.show()

def print_graph_details(graph: nx.Graph):
    print("Nodes:")
    for node, attrs in graph.nodes(data=True):
        print(f"  Node {node}: Type: {attrs['type']}, Features: {attrs['features']}")
    
    print("Edges:")
    for src, dst, attrs in graph.edges(data=True):
        print(f"  Edge from {src} to {dst}: Position: {attrs['position']}")


class GraphEncoderFreeway:
    def __init__(self, obj_type_id: str = "obj", proximity_threshold: float = 1):
        self.obj_type_id = obj_type_id
        self.proximity_threshold = proximity_threshold

    def encode(self, batch_node_features: torch.Tensor) -> Batch:
        batch_data = []
        batch_size = batch_node_features.size(0)

        for b in range(batch_size):
            node_features = batch_node_features[b]
            num_nodes = node_features.size(0)
            graph = nx.Graph()

            object_feature_length = node_features.size(1)

            # Add object nodes
            for i in range(num_nodes):
                graph.add_node(i, type=self.obj_type_id, features=node_features[i].tolist())

            atom_index = num_nodes

            # Identify node types
            chicken_indices = [i for i in range(num_nodes) if node_features[i, -3] == 1]
            lane_indices = [i for i in range(num_nodes) if node_features[i, -2] == 1]
            car_indices = [i for i in range(num_nodes) if node_features[i, -1] == 1]
            atom_features = torch.zeros((2, object_feature_length)).tolist()

            #Add atom which connects all objects to chicken
            if chicken_indices:
                for i in chicken_indices:
                    for j in range(num_nodes):
                        if i != j:
                            graph.add_node(atom_index, type="atom", features=atom_features)
                            graph.add_edge(i, atom_index, position=0) 
                            graph.add_edge(j, atom_index, position=1)
                            atom_index += 1 
            # #connect all objects to each other
            # for i in range(num_nodes):
            #     for j in range(num_nodes):
            #         if i != j:
            #             graph.add_node(atom_index, type="atom", features=atom_features)
            #             graph.add_edge(i, atom_index, position=0)
            #             graph.add_edge(j, atom_index, position=1)
            #             atom_index += 1

            # # Add ChickenOnLane and CarOnLane atoms and edges
            # for i in chicken_indices + car_indices:
            #     for j in lane_indices:
            #         if node_features[i, 1] == node_features[j, 1]:
            #             atom_type = "ChickenOnLane" if i in chicken_indices else "CarOnLane"
            #             graph.add_node(atom_index, type=atom_type, features=atom_features)
            #             graph.add_edge(i, atom_index, position=0)
            #             graph.add_edge(j, atom_index, position=1)
            #             atom_index += 1

            # # Add LaneNextToLane atoms and edges
            # for i in range(len(lane_indices) - 1):
            #     atom_features = torch.zeros((2, object_feature_length)).tolist()
            #     graph.add_node(atom_index, type="LaneNextToLane", features=atom_features)
            #     graph.add_edge(lane_indices[i], atom_index, position=0)
            #     graph.add_edge(lane_indices[i + 1], atom_index, position=1)
            #     atom_index += 1

            # # Add PlayerNearCar edges
            # for i in chicken_indices:
            #     for j in car_indices:
            #         distance = abs(node_features[i, 0] - node_features[j, 0]) + abs(node_features[i, 1] - node_features[j, 1])
            #         # if distance < self.proximity_threshold:
            #         #     graph.add_node(atom_index, type="PlayerNearCar", features=atom_features)
            #         #     graph.add_edge(i, atom_index, position=0)
            #         #     graph.add_edge(j, atom_index, position=1)
            #         #     atom_index += 1
            #         graph.add_node(atom_index, type="PlayerNearCar", features=atom_features)
            #         graph.add_edge(i, atom_index, position=0)
            #         graph.add_edge(j, atom_index, position=1)
            #         atom_index += 1

            # Add CarApproachingPlayer edges
            # for i in chicken_indices:
            #     for j in car_indices:
            #         rel_velocity = node_features[j, 2] - node_features[i, 2]  # Assuming index 2 is x-velocity
            #         if rel_velocity > 0 and node_features[j, 0] < node_features[i, 0]:

            # print the graph and show the node types and edges as well
            # visualize_graph(graph)

            # print_graph_details(graph)


            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))
    def to_pyg_data(self, batch_graphs: List[nx.Graph]) -> List[HeteroData]:
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = defaultdict(dict)
            obj_features = []
            atom_features_dict = defaultdict(list)
            edge_dict = defaultdict(list)

            current_obj_features = []
            current_atom_features_dict = defaultdict(list)

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)
                else:
                    node_index_mapping[node_type][node] = len(current_atom_features_dict[node_type])
                    current_atom_features_dict[node_type].append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))
            for node_type, features_list in current_atom_features_dict.items():
                if features_list:
                    flattened_features = [f.view(-1) for f in features_list]
                    atom_features_dict[node_type].append(torch.stack(flattened_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)
            for node_type, features_list in atom_features_dict.items():
                if features_list:
                    data[node_type].x = torch.cat(features_list)

            for src, dst, attr in graph.edges(data=True):
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                pos = str(attr['position'])
                edge_type = (src_type, pos, dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, pos, src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list
    
class GraphEncoderFreewayProximity:
    def __init__(self, obj_type_id: str = "obj"):
        self.obj_type_id = obj_type_id

    def encode(self, batch_node_features: torch.Tensor, proximity_threshold: float = 50) -> Batch:
        batch_data = []
        batch_size = batch_node_features.size(0)

        for b in range(batch_size):
            node_features = batch_node_features[b]
            num_nodes = node_features.size(0)
            graph = nx.Graph()

            # Add object nodes
            for i in range(num_nodes):
                graph.add_node(i, type=self.obj_type_id, features=node_features[i].tolist())

            # Find the player node
            player_indices = [i for i in range(num_nodes) if node_features[i, -3] == 1]

            if player_indices:
                player_index = player_indices[0]
                # Add edges between the player node and other cars and lanes
                for i in range(num_nodes):
                    if i != player_index:
                        dist = torch.norm(node_features[player_index, :2] - node_features[i, :2]).item()
                        if dist <= proximity_threshold:
                            graph.add_edge(player_index, i)
                            graph.add_edge(i, player_index)  # Add reverse edge for bidirectionality

            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def to_pyg_data(self, batch_graphs: List[nx.Graph]) -> List[HeteroData]:
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = defaultdict(dict)
            obj_features = []
            edge_dict = defaultdict(list)

            current_obj_features = []

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)

            for src, dst in graph.edges:
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                edge_type = (src_type, 'to', dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, 'to', src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list

# class GraphEncoderFreeway:
#     def __init__(self, obj_type_id: str = "obj"):
#         self.obj_type_id = obj_type_id

#     def encode(self, batch_node_features: torch.Tensor, proximity_threshold: float = 50) -> Batch:
#         batch_data = []
#         batch_size = batch_node_features.size(0)

#         for b in range(batch_size):
#             node_features = batch_node_features[b]
#             num_nodes = node_features.size(0)
#             graph = nx.Graph()

#             # Adding object nodes
#             for i in range(num_nodes):
#                 graph.add_node(i, type=self.obj_type_id, features=node_features[i].tolist())

#             # Adding atom nodes based on proximity and specific predicates
#             atom_index = num_nodes
#             object_feature_length = node_features.size(1)

#             # Add ChickenOnLane atoms and edges
#             for i in range(num_nodes):
#                 if node_features[i, -3] == 1:  # Assuming the 5th feature is a flag for the chicken
#                     for j in range(num_nodes):
#                         if node_features[j, -2] == 1:  # Assuming the 6th feature is a flag for lanes
#                             if abs(node_features[i, 1] - node_features[j, 1]) <= 50:
#                                 atom_features = torch.zeros((2, object_feature_length)).tolist()
#                                 graph.add_node(atom_index, type="ChickenOnLane", features=atom_features)
#                                 graph.add_edge(i, atom_index, position=0)
#                                 graph.add_edge(j, atom_index, position=1)
#                                 atom_index += 1

#             # Add CarOnLane atoms and edges
#             for i in range(num_nodes):
#                 if node_features[i, -1] == 1:  # Assuming the last feature is a flag for cars
#                     for j in range(num_nodes):
#                         if node_features[j, -2] == 1:  # Assuming the 6th feature is a flag for lanes
#                             if abs(node_features[i, 1] - node_features[j, 1]) <= 50:
#                                 atom_features = torch.zeros((2, object_feature_length)).tolist()
#                                 graph.add_node(atom_index, type="CarOnLane", features=atom_features)
#                                 graph.add_edge(i, atom_index, position=0)
#                                 graph.add_edge(j, atom_index, position=1)
#                                 atom_index += 1

#             # Add LaneNextToLane atoms and edges
#             lanes = [i for i in range(num_nodes) if node_features[i, -2] == 1]  # Collect lane nodes
#             for i in range(len(lanes) - 1):
#                 atom_features = torch.zeros((2, object_feature_length)).tolist()
#                 graph.add_node(atom_index, type="LaneNextToLane", features=atom_features)
#                 graph.add_edge(lanes[i], atom_index, position=0)
#                 graph.add_edge(lanes[i + 1], atom_index, position=1)
#                 atom_index += 1

#             batch_data.append(graph)

#         return Batch.from_data_list(self.to_pyg_data(batch_data))

#     def to_pyg_data(self, batch_graphs):
#         data_list = []

#         for graph in batch_graphs:
#             data = HeteroData()
#             node_index_mapping = defaultdict(dict)
#             obj_features = []
#             atom_features_dict = defaultdict(list)
#             edge_dict = defaultdict(list)

#             current_obj_features = []
#             current_atom_features_dict = defaultdict(list)

#             for node, attrs in graph.nodes(data=True):
#                 node_type = attrs['type']
#                 features = torch.tensor(attrs['features'])
#                 if node_type == self.obj_type_id:
#                     node_index_mapping[node_type][node] = len(current_obj_features)
#                     current_obj_features.append(features)
#                 else:
#                     node_index_mapping[node_type][node] = len(current_atom_features_dict[node_type])
#                     current_atom_features_dict[node_type].append(features)

#             if current_obj_features:
#                 obj_features.append(torch.stack(current_obj_features))
#             for node_type, features_list in current_atom_features_dict.items():
#                 if features_list:
#                     flattened_features = [f.view(-1) for f in features_list]
#                     atom_features_dict[node_type].append(torch.stack(flattened_features))

#             if obj_features:
#                 data[self.obj_type_id].x = torch.cat(obj_features)
#             for node_type, features_list in atom_features_dict.items():
#                 if features_list:
#                     data[node_type].x = torch.cat(features_list)

#             for src, dst, attr in graph.edges(data=True):
#                 src_type = graph.nodes[src]['type']
#                 dst_type = graph.nodes[dst]['type']
#                 pos = str(attr['position'])
#                 edge_type = (src_type, pos, dst_type)

#                 src_idx = node_index_mapping[src_type][src]
#                 dst_idx = node_index_mapping[dst_type][dst]
#                 edge_dict[edge_type].append((src_idx, dst_idx))
#                 # Add reverse edges for bidirectionality
#                 reverse_edge_type = (dst_type, pos, src_type)
#                 edge_dict[reverse_edge_type].append((dst_idx, src_idx))

#             for edge_type, edges in edge_dict.items():
#                 edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
#                 data[edge_type].edge_index = edge_tensor

#             data_list.append(data)

#         return data_list 

class GraphEncoderPacman:
    def __init__(self, obj_type_id: str = "obj", atom_type_id: str = "atom"):
        self.obj_type_id = obj_type_id
        self.atom_type_id = atom_type_id
        self.grid_width = 21
        self.grid_height = 25
        self.map = {}

    def load_map(self, level_num: int, script_path: str):
        self.map = {}
        file_path = os.path.join(script_path, "res", "levels", f"{level_num}.txt")
        with open(file_path, 'r') as f:
            line_num = -1
            row_num = 0
            is_reading_level_data = False

            for line in f:
                line_num += 1
                line = line.strip()
                if not line or line.startswith("'") or line.startswith("#"):
                    continue

                if line.startswith("#"):
                    parts = line.split(' ')
                    key = parts[1]

                    if key == "lvlwidth":
                        self.grid_width = int(parts[2])
                    elif key == "lvlheight":
                        self.grid_height = int(parts[2])
                    elif key == "startleveldata":
                        is_reading_level_data = True
                        row_num = 0
                    elif key == "endleveldata":
                        is_reading_level_data = False
                elif is_reading_level_data:
                    values = list(map(int, line.split(' ')))
                    for col, val in enumerate(values):
                        self.map[(row_num, col)] = val
                    row_num += 1

    def GetMapTile(self, row, col):
        return self.map.get((row, col), 0)

    def IsWall(self, row, col):
        if row > self.grid_height - 1 or row < 0:
            return True
        if col > self.grid_width - 1 or col < 0:
            return True

        result = self.GetMapTile(row, col)
        if result >= 100 and result <= 199:
            return True
        else:
            return False

    def find_non_wall_cells(self):
        non_wall_cells = []
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if not self.IsWall(row, col):
                    non_wall_cells.append((row, col))
        return non_wall_cells

    
    def encode(self, batch_object_features: torch.Tensor) -> Batch:
        batch_data = []
        batch_size = batch_object_features.size(0)
        non_wall_cells = self.find_non_wall_cells()
        cell_feature_dim = batch_object_features.size(2)

        for b in range(batch_size):
            object_features = batch_object_features[b]
            # remove the values from batch_object_features that have all zeros in the vector
            object_features = object_features[~torch.all(object_features == 0, dim=1)]
            graph = nx.Graph()

            # Add nodes for non-wall cells
            for (i, j) in non_wall_cells:
                cell_features = torch.zeros(cell_feature_dim).tolist()
                cell_features[0] = i
                cell_features[1] = j
                graph.add_node((i, j), type=self.obj_type_id, features=cell_features)

            atom_index = 0  # Atom index counter

            # Add "to right of" relation and atom nodes
            for (i, j) in non_wall_cells:
                if (i, j + 1) in non_wall_cells:
                    atom_features = torch.zeros((2, cell_feature_dim)).tolist()
                    graph.add_node(atom_index, type="RightOf", features=atom_features)
                    graph.add_edge((i, j), atom_index, position=0)
                    graph.add_edge((i, j + 1), atom_index, position=1)
                    atom_index += 1

            # Add "above" relation and atom nodes
            for (i, j) in non_wall_cells:
                if (i + 1, j) in non_wall_cells:
                    atom_features = torch.zeros((2, cell_feature_dim)).tolist()
                    graph.add_node(atom_index, type="Above", features=atom_features)
                    graph.add_edge((i, j), atom_index, position=0)
                    graph.add_edge((i + 1, j), atom_index, position=1)
                    atom_index += 1

            # Add nodes and edges for objects
            for obj_idx, (obj_x, obj_y) in enumerate(object_features[:, :2]):
                obj_node = (obj_x.item(), obj_y.item(), "obj")
                cell_node = (obj_x.item(), obj_y.item())

                graph.add_node(obj_node, type=self.obj_type_id, features=object_features[obj_idx].tolist())
                
                # Create an atom node for the "at" relation
                atom_features = torch.zeros((2, cell_feature_dim)).tolist()
                graph.add_node(atom_index, type="At", features=atom_features)
                graph.add_edge(obj_node, atom_index, position=0)
                graph.add_edge(cell_node, atom_index, position=1)
                atom_index += 1

            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def to_pyg_data(self, batch_graphs):
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = defaultdict(dict)
            obj_features = []
            atom_features_dict = defaultdict(list)
            edge_dict = defaultdict(list)

            current_obj_features = []
            current_atom_features_dict = defaultdict(list)

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)
                else:
                    node_index_mapping[node_type][node] = len(current_atom_features_dict[node_type])
                    current_atom_features_dict[node_type].append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))
            for node_type, features_list in current_atom_features_dict.items():
                if features_list:
                    flattened_features = [f.view(-1) for f in features_list]
                    atom_features_dict[node_type].append(torch.stack(flattened_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)
            for node_type, features_list in atom_features_dict.items():
                if features_list:
                    data[node_type].x = torch.cat(features_list)

            for src, dst, attr in graph.edges(data=True):
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                pos = str(attr['position'])
                edge_type = (src_type, pos, dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, pos, src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list 
    


class GraphEncoderBreakout:
    def __init__(self, proximity_threshold=100, adjacency_threshold=5):
        self.proximity_threshold = proximity_threshold
        self.adjacency_threshold = adjacency_threshold
        self.obj_type_id = "obj"
        self.atom_type_id = "atom"

    def encode(self, batch_object_features: torch.Tensor) -> Batch:
        batch_data = []
        batch_size = batch_object_features.size(0)
        cell_feature_dim = batch_object_features.size(2)

        for b in range(batch_size):
            object_features = batch_object_features[b]
            # remove the values from batch_object_features that have all zeros in the vector
            object_features = object_features[~torch.all(object_features == 0, dim=1)]
            graph = nx.Graph()
            num_nodes = object_features.size(0)
            graph = nx.Graph()

            # Adding object nodes
            for i in range(num_nodes):
                graph.add_node(i, type=self.obj_type_id, features=object_features[i].tolist())

            atom_index = 0  # Atom index counter

            # Create an atom node for the "proximity" relation between all objects
            atom_index = num_nodes
            object_feature_length = object_features.size(1)  # Corrected to get the length of the feature vector

            for i, j in combinations(range(num_nodes), 2):
                dist = torch.norm(object_features[i, :2] - object_features[j, :2]).item()
                if dist < self.proximity_threshold:
                    # Create atom node with a 2D zero vector of the shape (2, object_feature_length)
                    atom_features = torch.zeros((2, object_feature_length)).tolist()
                    graph.add_node(atom_index, type=self.atom_type_id, features=atom_features)
                    graph.add_edge(i, atom_index, position=0)
                    graph.add_edge(j, atom_index, position=1)
                    atom_index += 1

            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def check_proximity(self, obj1, obj2, threshold):
        center1 = obj1[:2]
        center2 = obj2[:2]
        distance = torch.norm(torch.tensor(center1) - torch.tensor(center2)).item()
        return distance < threshold

    def check_adjacent(self, obj1, obj2, threshold):
        return self.check_proximity(obj1, obj2, threshold)

    def to_pyg_data(self, batch_graphs):
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = defaultdict(dict)
            obj_features = []
            atom_features_dict = defaultdict(list)
            edge_dict = defaultdict(list)

            current_obj_features = []
            current_atom_features_dict = defaultdict(list)

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)
                else:
                    node_index_mapping[node_type][node] = len(current_atom_features_dict[node_type])
                    current_atom_features_dict[node_type].append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))
            for node_type, features_list in current_atom_features_dict.items():
                if features_list:
                    flattened_features = [f.view(-1) for f in features_list]
                    atom_features_dict[node_type].append(torch.stack(flattened_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)
            for node_type, features_list in atom_features_dict.items():
                if features_list:
                    data[node_type].x = torch.cat(features_list)

            for src, dst, attr in graph.edges(data=True):
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                pos = str(attr['position'])
                edge_type = (src_type, pos, dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, pos, src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list

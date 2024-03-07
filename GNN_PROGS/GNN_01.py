# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx

# grafo com 4 nós - edges list
#  aqui, as origens estao acima e os destinos abaixo, onde, p.ex.
#  nó 0 liga-se com os nós 1, 2 e 3. Nó 1, liga-se ao zero e assim segue...
edge_list = torch.tensor([
                        [0,0,0,1,2,2,3,3], # source nodes
                        [1,2,3,0,0,3,0,2]  # target nodes
                        ], dtype=torch.long)
#features do grafo
node_features = torch.tensor([
                            [-8, 1, 5, 8, 2, -3], #feature do Nó 0
                            [-1, 0, 2, -3, 0, 1], #feature do Nó 1
                            [1, -1, 2, -3, 0, 1], #feature do Nó 2
                            [0, 1, 4, -2, 3, 4]   #feature do Nó 3
                            ],dtype=torch.long)
# pesos dos arcos - 
weight_list = torch.tensor([
                        [35,], # peso para nós (0,1)
                        [48,], # peso para nós (0,2)
                        [12,], # peso para nós (0,3)
                        [10,], # peso para nós (1,0)
                        [70,], # peso para nós (2,0)
                        [5,], # peso para nós (2,3)
                        [8,], # peso para nós (3,0)
                        [15,], # peso para nós (3,2)
                        ], dtype=torch.long)

data = Data(x=node_features, edge_index=edge_list, edge_attr=weight_list)

" Print the graph info "
print("Num. de nós:     ", data.num_nodes, "\n")
print("Num. de ligações:", data.num_edges, "\n")
print("Num. de features por nós:", data.num_node_features, "\n")
print("Num. de pesos por nó:", data.num_edge_features, "\n")

" Plot the graph "
G = to_networkx(data)
nx.draw_networkx(G)
import osmnx as ox

G = ox.graph_from_place(
    "Vilnius, Lithuania",
    network_type="drive"
)

print(len(G.nodes), len(G.edges))
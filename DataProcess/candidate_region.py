
import numpy as np

def get_candidate_residues(G, ligand_coords, cutoff=10.0):
    
    """
    G: networkx Graph (residue graph)
    ligand_coords: np.ndarray [N, 3]
    return: list of node_id
    """
    
    candidates = []

    for node_id, data in G.nodes(data=True):
        ca = data["coord"]  # (3,)
        dists = np.linalg.norm(ligand_coords - ca, axis=1)
        if dists.min() <= cutoff:
            candidates.append(node_id)

    return candidates

import numpy as np



def energy(G, state, alpha=1.0, beta=1.0):
    active = [r for r, v in state.items() if v == 1]
    if len(active) <= 1:
        return 1e6  # 惩罚空口袋

    # 1. 紧凑性（平均距离）
    coords = np.array([G.nodes[r]["coord"] for r in active])
    center = coords.mean(axis=0)
    dist_term = np.mean(np.linalg.norm(coords - center, axis=1))

    # 2. 图连通性
    edge_cnt = 0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if G.has_edge(active[i], active[j]):
                edge_cnt += 1

    graph_term = -edge_cnt

    return alpha * dist_term + beta * graph_term

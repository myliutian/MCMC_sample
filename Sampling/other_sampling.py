import numpy as np
from sklearn.cluster import KMeans
import random
from networkx import Graph
import json
import random
import numpy as np

def run_greedy_probs(G, candidates, steps=1, max_size=30, seed_perturb=0.01):
    """
    贪心算法接口，返回 List[Dict[residue_id, int]] 
    每个元素是一次采样的状态，格式与 run_mcmc 一致
    steps: 采样次数
    max_size: 口袋最大残基数
    seed_perturb: 对评分函数加入随机扰动，增加不同采样结果的多样性
    """
    trajectory = []

    for _ in range(steps):
        # 节点评分函数，加入随机扰动
        def score(r):
            deg = G.degree(r)
            coord = G.nodes[r]["coord"]
            neighbors = list(G.neighbors(r))
            if not neighbors:
                return -1e6
            center = np.mean([G.nodes[n]["coord"] for n in neighbors], axis=0)
            return deg - np.linalg.norm(coord - center) + random.uniform(0, seed_perturb)

        # 随机选择种子节点，而不是总是最大值
        seed = random.choice(candidates)
        pocket = {seed}
        frontier = {seed}

        while len(pocket) < max_size and frontier:
            candidate_nodes = set()
            for r in frontier:
                candidate_nodes.update(set(G.neighbors(r)) & set(candidates))
            candidate_nodes -= pocket

            if not candidate_nodes:
                break

            # 从 top-k 节点随机选择一个扩展
            k = min(3, len(candidate_nodes))  # top 3
            top_k = sorted(candidate_nodes, key=score, reverse=True)[:k]
            best = random.choice(top_k)
            pocket.add(best)
            frontier = {best}

        # 转成 {res: 1/0} 字典
        state = {r: (1 if r in pocket else 0) for r in candidates}
        trajectory.append(state)

    return trajectory

# ==========================
# Graph-cut 算法改接口
# ==========================
def run_graph_cut_probs(G, candidates, steps=300, lam=1.0):
    trajectory = []

    for _ in range(steps):
        state = {r: 0 for r in candidates}
        avg_deg = np.mean([G.degree(n) for n in candidates])
        for r in candidates:
            if G.degree(r) > avg_deg:
                state[r] = 1

        def energy(s):
            E = 0.0
            for r, v in s.items():
                deg = G.degree(r)
                E += -deg if v == 1 else 0.2 * deg
            for u, v in G.edges:
                if u in candidates and v in candidates:
                    if s[u] != s[v]:
                        E += lam
            return E

        for _ in range(steps):
            r = random.choice(candidates)
            new_state = state.copy()
            new_state[r] = 1 - state[r]
            if energy(new_state) < energy(state):
                state = new_state

        trajectory.append(state.copy())

    return trajectory


# ==========================
# Spectral 算法改接口
# ==========================
def run_spectral_probs(G, candidates, k=3):
    nodes = list(candidates)
    idx = {r: i for i, r in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))

    for u, v, attr in G.edges(data=True):
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            d = attr.get("distance", 1.0)
            A[i, j] = A[j, i] = np.exp(-d)

    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)
    X = eigvecs[:, :k]

    labels = KMeans(n_clusters=k, n_init=10).fit_predict(X)

    # 选择最紧凑簇
    cluster_scores = {}
    for c in range(k):
        cluster_nodes = [i for i in range(n) if labels[i] == c]
        if len(cluster_nodes) < 2:
            cluster_scores[c] = 1e6
            continue
        coords = np.array([G.nodes[nodes[i]]["coord"] for i in cluster_nodes])
        center = coords.mean(axis=0)
        cluster_scores[c] = np.mean(np.linalg.norm(coords - center, axis=1))

    best_cluster = min(cluster_scores, key=cluster_scores.get)

    # 输出轨迹 List[Dict[str,int]]
    trajectory = []
    state = {r: (1 if labels[idx[r]] == best_cluster else 0) for r in candidates}
    trajectory.append(state)

    return trajectory
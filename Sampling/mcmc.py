import random
import math
from Sampling.energy import energy
import random

def init_state(candidates, p=0.3):
    state = {}
    for r in candidates:
        state[r] = 1 if random.random() < p else 0
    return state

def run_mcmc(G, candidates, steps=1000, T=1.0, burn_in=0):
    """
    返回: List[Dict[residue_id, int]] 
    每个元素是一个字典，表示该轮采样的状态（0/1）
    """
    state = init_state(candidates)
    trajectory = []  # 存储每一步的状态
    E = energy(G, state)

    for step in range(steps):
        # 1. proposal
        r = random.choice(candidates)
        new_state = state.copy()
        new_state[r] = 1 - state[r]

        # 2. energy & accept/reject
        E_new = energy(G, new_state)
        dE = E_new - E
        if dE < 0 or random.random() < math.exp(-dE / T):
            state = new_state
            E = E_new

        # 3. 记录状态
        trajectory.append(state.copy())  # 必须 copy，否则引用会变

    # 可选：去掉 burn-in
    if burn_in > 0:
        trajectory = trajectory[burn_in:]
    
    return trajectory  



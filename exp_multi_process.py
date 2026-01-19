import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import pickle
import zipfile
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from DataProcess.candidate_region import get_candidate_residues
from Sampling.mcmc import run_mcmc
from Sampling.other_sampling import run_graph_cut_probs, run_greedy_probs, run_spectral_probs


# ======================
# MCMC 参数（默认）
# ======================
MCMC_STEPS = 1000
BURN_IN = 200
TEMPERATURE = 1.0
CONFIDENCE_THRESHOLD = 0.7
FLUCTUATION_THRESHOLD = 0.2


# ======================
# 辅助函数
# ======================

def pdb_id_to_year_range(pdb_id: str, raw_data_dir: Path) -> str:
    year_ranges = ["1981-2000", "2001-2010", "2011-2019"]
    for yr in year_ranges:
        if (raw_data_dir / yr / pdb_id).exists():
            return yr
    raise FileNotFoundError(f"Cannot find year range for {pdb_id}")


def load_ligand_coords(ligand_dir: Path):
    mol2_path = ligand_dir / f"{ligand_dir.name}_ligand.mol2"
    mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load {mol2_path}")

    conf = mol.GetConformer()
    coords = [[conf.GetAtomPosition(i).x,
               conf.GetAtomPosition(i).y,
               conf.GetAtomPosition(i).z]
              for i in range(mol.GetNumAtoms())]

    return np.array(coords)


def load_pocket_residues(pocket_pdb: Path):
    residues = set()
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21].strip() or "A"
                resseq = int(line[22:26])
                residues.add((chain, resseq))
    return residues


def compute_frequencies_block(samples, all_candidates, num_blocks=10):
    total_steps = len(samples)
    sampled_list = [r for s in samples for r, v in s.items() if v == 1]
    freq_counter = Counter(sampled_list)

    probs = {}
    for r in all_candidates:
        prob = freq_counter.get(r, 0) / total_steps
        probs[r] = prob

    block_size = max(1, total_steps // num_blocks)
    fluctuations = {}

    for r in all_candidates:
        block_freqs = []
        for b in range(num_blocks):
            block = samples[b * block_size:(b + 1) * block_size]
            if block:
                block_freqs.append(sum(1 for s in block if r in s) / len(block))
        fluctuations[r] = np.std(block_freqs) if len(block_freqs) > 1 else 0.0

    return probs, fluctuations


def calculate_metrics(pred, true):
    pred, true = set(pred), set(true)
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    iou = tp / len(pred | true) if pred | true else 0

    return dict(f1=f1, recall=recall, precision=precision, iou=iou)


def extract_castp_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def load_castp_pocket_residues(poc_dir: Path):
    residues = set()
    for poc_file in poc_dir.glob("*.poc"):
        with open(poc_file) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain = line[21].strip() or "A"
                    try:
                        residues.add((chain, int(line[22:26])))
                    except ValueError:
                        pass
    return residues


def compute_centroid_from_residues(G, residues):
    coords = [
        data["coord"]
        for node_id, data in G.nodes(data=True)
        if node_id in residues
    ]
    return np.mean(coords, axis=0) if coords else None


def compute_dcc(pred, true, G):
    c1 = compute_centroid_from_residues(G, pred)
    c2 = compute_centroid_from_residues(G, true)
    return float(np.linalg.norm(c1 - c2)) if c1 is not None and c2 is not None else None


# ======================
# sampling 算法
# ======================
def run_sampling(G, candidates, steps=MCMC_STEPS, T=TEMPERATURE, sampling='mcmc'):
    
    if sampling == 'mcmc':
        return run_mcmc(G, candidates, MCMC_STEPS, T)[BURN_IN:]
    elif sampling == 'graph_cut':
        return run_graph_cut_probs(G, candidates, 10)
    elif sampling == 'greedy':
        return run_greedy_probs(G, candidates, 100)
    elif sampling == 'spectral':
        return run_spectral_probs(G, candidates)


# ======================
# 单 PDB 处理
# ======================

def process_one_pdb(pdb_id, mode, graph_dir, raw_data_dir, castp_zip_dir, sampling):
    try:
        year = pdb_id_to_year_range(pdb_id, raw_data_dir)

        with open(graph_dir / year / pdb_id / f"{pdb_id}_graph.pkl", "rb") as f:
            G = pickle.load(f)

        ligand_coords = load_ligand_coords(raw_data_dir / year / pdb_id)
        candidates = get_candidate_residues(G, ligand_coords)

        if not candidates:
            return pdb_id, None, "no_candidates"

        samples = run_sampling(G, candidates, steps=MCMC_STEPS, T=TEMPERATURE, sampling=sampling)

        # save_trajectory_to_json(samples, f"./results/{sampling}_trajectory.json")

        probs, fluctuations = compute_frequencies_block(samples, candidates)

        # print(type(probs), probs)
        # print(type(fluctuations), fluctuations)

        pred = [
            r for r in candidates
            if probs[r] >= CONFIDENCE_THRESHOLD
            and fluctuations[r] < FLUCTUATION_THRESHOLD
        ]

        if mode == "test":
            zip_path = castp_zip_dir / f"{pdb_id}.zip"
            if not zip_path.exists():
                return pdb_id, None, "no_castp"
            out_dir = castp_zip_dir / f"{pdb_id}_extracted"
            extract_castp_zip(zip_path, out_dir)
            true = load_castp_pocket_residues(out_dir)
        else:
            true = load_pocket_residues(
                raw_data_dir / year / pdb_id / f"{pdb_id}_pocket.pdb"
            )

        # print(type(pred))
        # print(pred)

        dcc = compute_dcc(pred, true, G)
        metrics = calculate_metrics(pred, true)
        metrics["dcc"] = dcc
        metrics["success_6A"] = dcc is not None and dcc < 6.0

        # print('finished', pdb_id)

        return pdb_id, {
            "metrics": metrics,
            "num_candidates": len(candidates),
            "num_predicted": len(pred),
            "num_true": len(true),
        }, "ok"

    except Exception as e:
        # print(e)
        return pdb_id, str(e), "error"


# ======================
# ThreadPool + tqdm
# ======================

def exp_with_threads(split_file, split_name, graph_dir, raw_data_dir, castp_zip_dir, sampling):
    pdb_ids = [l.strip() for l in open(split_file) if l.strip()]
    # pdb_ids = pdb_ids[:128]
    results = {}

    # num_workers = min(8, os.cpu_count())
    if sampling == 'mcmc':
        num_workers = 32
    elif sampling == 'graph_cut':
        num_workers = 32
    elif sampling == 'greedy':
        num_workers = 8
    elif sampling == 'spectral':
        num_workers = 2

    # num_workers = 32

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [
            executor.submit(
                process_one_pdb,
                pid,
                split_name,
                graph_dir,
                raw_data_dir,
                castp_zip_dir,
                sampling
            )
            for pid in pdb_ids
        ]

        for fut in tqdm(tasks, total=len(tasks), desc="Processing PDBs"):
            pid, res, status = fut.result()
            if status == "ok":
                results[pid] = res

    with open(f"./results/{sampling}_{split_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ======================
# main
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=Path, required=True)
    parser.add_argument("--raw_data_dir", type=Path, required=True)
    parser.add_argument("--castp_zip_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument(
        "--sampling",
        default="mcmc",
        choices=["mcmc", "graph_cut", "greedy", "spectral"],
        help="Sampling method for pocket prediction"
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    exp_with_threads(
        args.split_file,
        args.split,
        args.graph_dir,
        args.raw_data_dir,
        args.castp_zip_dir,
        args.sampling,
    )


if __name__ == "__main__":
    main()

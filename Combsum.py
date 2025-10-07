#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CombMNZ and Weighted CombMNZ with QPP-based weights
---------------------------------------------------
Supports:
    - Standard CombMNZ
    - Weighted CombMNZ (W-CombMNZ) using per-query QPP estimates

QPP inputs:
    Each .qpp file contains per-query QPP scores for all rankers.
    You can select one QPP model using --qpp_model (e.g., NQC, SCNQC, etc.)
"""

import os
import argparse
import pandas as pd
from collections import defaultdict

# -------------------------------
#  QPP model mapping
# -------------------------------
model_name_dict = {
    0:"SMV", 1:"Sigma_max", 2:"Sigma(%)", 3:"NQC", 4:"UEF", 5:"RSD",
    6:"QPP-PRP", 7:"WIG", 8:"SCNQC", 9:"QV-NQC", 10:"DM",
    11:"NQA-QPP", 12:"BERTQPP"
}


# -------------------------------
#  Loaders
# -------------------------------
def load_runs(res_path):
    """Load normalized run files (.norm.res) into a dict of DataFrames."""
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")]
    for f in files:
        ranker = f.replace(".norm.res", "")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+",
                         names=["qid", "iter", "docno", "rank", "score", "runid"])
        runs[ranker] = df
    return runs


def load_qpp_estimates(qpp_path):
    """Load QPP files: {qid: {ranker: [qpp_scores...]}}"""
    qpp_data = defaultdict(dict)
    files = [os.path.join(qpp_path, f) for f in os.listdir(qpp_path) if f.endswith(".mmnorm.qpp")]
    for f in files:
        ranker = os.path.basename(f).replace(".res.mmnorm.qpp", "")
        with open(f, "r") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]  # all QPP methods
                qpp_data[qid][ranker] = scores
    return qpp_data


# -------------------------------
#  Fusion functions
# -------------------------------
def combmnz_fusion(runs):
    """Standard CombMNZ"""
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df.qid.unique()) for df in runs.values()]))

    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)

        for ranker, df in runs.items():
            sub = df[df.qid == qid]
            for _, row in sub.iterrows():
                doc_scores[row.docno] += row.score
                doc_counts[row.docno] += 1

        for docid, score_sum in doc_scores.items():
            fused[qid].append((docid, score_sum * doc_counts[docid]))

    return fused


def weighted_combmnz_fusion(runs, qpp_data, qpp_model_name):
    """Weighted CombMNZ using selected QPP model"""

    if qpp_model_name=="fusion":
        qpp_index=-1
    else:
        # find the index of desired QPP model
        qpp_index = None
        for i, name in model_name_dict.items():
            if name.lower() == qpp_model_name.lower():
                qpp_index = i
                break
        if qpp_index is None:
            raise ValueError(f"Invalid QPP model '{qpp_model_name}'. Must be one of: {list(model_name_dict.values())}")

    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df.qid.unique()) for df in runs.values()]))

    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)
        w = 0

        for ranker, df in runs.items():
            sub = df[df.qid == qid]
            if qid not in qpp_data or ranker not in qpp_data[qid]:
                w = 1.0
            elif not qpp_index==-1:
                w = qpp_data[qid][ranker][qpp_index]  # QPP weight for (qid, ranker)a
            else:
                for j, name in model_name_dict.items():
                    w += qpp_data[qid][ranker][j]  # Average QPP weight for (qid, ranker)

        if qpp_index==-1:
            w = w / len(model_name_dict)

        w = w/len(runs)
 
        for _, row in sub.iterrows():
            doc_scores[row.docno] += w * row.score
            doc_counts[row.docno] += 1

        for docid, score_sum in doc_scores.items():
            fused[qid].append((docid, score_sum * doc_counts[docid]))

    return fused

# -------------------------------
#  Writer
# -------------------------------
def write_runfile(fused, output_path, tag="combmnz"):
    with open(output_path, "w") as fout:
        for qid in sorted(fused.keys()):
            ranked = sorted(fused[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
    print(f"✅ Fused run written to {output_path}")


# -------------------------------
#  Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="CombMNZ / Weighted CombMNZ with QPP")
    parser.add_argument("--res_path", required=True, help="Directory with normalized run files (.norm.res)")
    parser.add_argument("--qpp_path", help="Directory containing .qpp files")
    parser.add_argument("--output", required=True, help="Output fused run file path")
    parser.add_argument("--qpp_model", default=None, help="QPP model name (e.g., NQC, SCNQC, WIG)")
    args = parser.parse_args()

    runs = load_runs(args.res_path)

    if args.qpp_model:
        if not args.qpp_path:
            raise ValueError("QPP path must be provided when using weighted fusion.")
        qpp_data = load_qpp_estimates(args.qpp_path)
        fused = weighted_combmnz_fusion(runs, qpp_data, args.qpp_model)
        tag = f"wcombmnz-{args.qpp_model.lower()}"
    else:
        fused = combmnz_fusion(runs)
        tag = "combmnz"

    write_runfile(fused, args.output, tag=tag)


if __name__ == "__main__":
    main()

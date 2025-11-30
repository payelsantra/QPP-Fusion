import pyterrier as pt
import pandas as pd
import os
import argparse
from collections import defaultdict

model_name_dict={0:"SMV",1:"Sigma_max",2:"Sigma(%)",3:"NQC",4:"UEF",5:"RSD",
                 6:"QPP-PRP",7:"WIG",8:"SCNQC",9:"QV-NQC",10:"DM",
                 11:"NQA-QPP",12:"BERTQPP"}

def load_qpp_estimates(qpp_path):
    qpp_data = defaultdict(dict)
    files = [os.path.join(qpp_path, f) for f in os.listdir(qpp_path) if f.endswith(".mmnorm.qpp")] #zscore 
    for f in files:
        ranker = os.path.basename(f).replace(".res.mmnorm.qpp","")   #.res.mmnorm.qpp minmax  .res.znorm.qpp
        with open(f, "r") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0]
                scores = [float(x) for x in parts[1:]]  # all QPP methods in columns
                qpp_data[qid][ranker] = scores
    return qpp_data

def load_runs(res_path):
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".100.norm.res")] #<-- z-score. minmax--> #100.minmax.res
    for f in files:
        ranker = f.replace(".norm.res","")
        df = pd.read_csv(os.path.join(res_path, f), sep=r"\s+", 
                         names=["qid","iter","docno","rank","score","runid"])
        runs[ranker] = df
    return runs

def build_combsum_run(runs, qpp_data=None, qpp_method_index=None):
    records = []
    rankers = list(runs.keys())
    for qid in runs[rankers[0]]["qid"].unique():
        doc_scores = defaultdict(float)
        for ranker in rankers:
            df_q = runs[ranker][runs[ranker]["qid"] == int(qid)]
            if df_q.empty:
                continue
            weight = 1.0
            if qpp_data and qpp_method_index is not None:
                try:
                    weight = qpp_data[str(qid)][ranker][qpp_method_index]
                except KeyError:
                    weight = 0.0
            # add score (weighted or not)
            for _, row in df_q.iterrows():
                doc_scores[row["docno"]] += weight * row["score"]
        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (docno, score) in enumerate(sorted_docs, start=1):
                runid = "W-CombSUM" if qpp_data else "CombSUM"
                records.append([qid,"Q0",docno,rank,score,runid])
    return pd.DataFrame(records, columns=["qid","iter","docno","rank","score","runid"])


def main():
    parser = argparse.ArgumentParser(description="Flexible QPP evaluation")
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--qpp_path", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs='+', default=["map","ndcg@10","recip_rank"])
    parser.add_argument("--strategy", type=str, choices=["combsum","wcombsum"], required=True,
                        help="Choose fusion strategy: combsum | wcombsum")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Optional: path to save fused run(s) in TREC format")
    args = parser.parse_args()

    runs = load_runs(args.res_path)
    qpp_data = load_qpp_estimates(args.qpp_path)

    qrels = pt.io.read_qrels(args.qrels)
    topics = pd.read_csv(args.topics, sep="\t", names=["qid","query"]).astype({'qid':'str'})

    sample_scores = next(iter(next(iter(qpp_data.values())).values()))
    num_qpp_methods = len(sample_scores)
    print(f"Detected {num_qpp_methods} QPP methods (columns) per ranker")

    measures = []
    for m in args.metrics:
        if m.lower() == "map":
            measures.append(pt.measures.AP(rel=2))
        elif "ndcg@10" in m.lower():
            k = int(m.split("@")[1]) if "@" in m else None
            measures.append(pt.measures.nDCG(cutoff=k) if k else pt.measures.nDCG())
        elif m.lower() == "recip_rank":
            measures.append(pt.measures.RR())

    all_runs, all_names = [], []

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    if args.strategy == "combsum":
        df = build_combsum_run(runs)
        all_runs.append(df)
        all_names.append("CombSUM")
        if args.save_path:
            save_file = os.path.join(args.save_path, "combsum_trec20.res")
            df.to_csv(save_file, sep=" ", header=False, index=False)
            print(f"[Saved] Fused run written to: {save_file}")

    elif args.strategy == "wcombsum":
        for i in range(num_qpp_methods):
            df = build_combsum_run(runs, qpp_data=qpp_data, qpp_method_index=i)
            all_runs.append(df)
            all_names.append(f"W-CombSUM-QPP-{model_name_dict[i]}")
            if args.save_path:
                save_file = os.path.join(args.save_path, f"wcombsum_qpp_{model_name_dict[i]}_trec2020.res")
                df.to_csv(save_file, sep=" ", header=False, index=False)
                print(f"[Saved] Weighted fused run written to: {save_file}")

    # Always add baselines
    all_runs.extend(runs.values())
    all_names.extend(runs.keys())

    results = pt.Experiment(
        all_runs,
        qrels=qrels,
        topics=topics,
        eval_metrics=measures,
        names=all_names
    )
    print(results)

if __name__ == "__main__":
    main()

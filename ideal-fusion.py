import os
import pyterrier as pt
from pyterrier.measures import *
import pandas as pd

import os
import pandas as pd

def load_runs(res_path):
    """Load all .norm.res runs from a directory into a dict of DataFrames."""
    runs = {}
    files = [f for f in os.listdir(res_path) if f.endswith(".norm.res")]
    for f in files:
        ranker = f.replace(".norm.res", "")
        df = pd.read_csv(
            os.path.join(res_path, f),
            sep=r"\s+",
            names=["qid", "iter", "docno", "rank", "score", "runid"],
            dtype={"qid": str, "docno": str}
        )
        runs[ranker] = df
    return runs


def ideal_fusion(res_path, qrels, top_k=None):
    """
    Generate an oracle fusion run by pooling documents from all runs and sorting by relevance.

    Args:
        res_path: path containing .norm.res files
        qrels: DataFrame with columns ['qid', 'docno', 'label']
        top_k: optional cutoff for output ranking (e.g., 1000)

    Returns:
        DataFrame with columns [qid, Q0, docid, rank, score, runname="ideal"]
    """
    # Load runs
    runs = load_runs(res_path)

    # Build a dict of qrels for quick lookup
    qrels_dict = (
        qrels.groupby("qid")[["docno", "label"]]
        .apply(lambda x: dict(zip(x.docno, x.label)))
        .to_dict()
    )

    ideal_rows = []

    # Collect all query IDs from runs
    all_qids = sorted(set().union(*[set(df["qid"].unique()) for df in runs.values()]))

    for qid in all_qids:
        # Union of all documents retrieved by any run for this query
        docs = set().union(*[df.loc[df.qid == qid, "docno"] for df in runs.values()])

        # Lookup relevance labels (default 0 if unjudged)
        qrel_lookup = qrels_dict.get(qid, {})
        scored_docs = [(doc, qrel_lookup.get(doc, 0)) for doc in docs]

        # Sort descending by label, tie-break by docid
        ranked_docs = sorted(scored_docs, key=lambda x: (-x[1], x[0]))

        if top_k:
            ranked_docs = ranked_docs[:top_k]

        for rank, (docno, label) in enumerate(ranked_docs, start=1):
            ideal_rows.append([qid, "Q0", docno, rank, label, "ideal"])

    ideal_df = pd.DataFrame(ideal_rows, columns=["qid", "iter", "docno", "rank", "score", "runid"])
    return ideal_df

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019')
qrels = dataset.get_qrels()
ideal_df = ideal_fusion("data/rl/2019/", qrels, top_k=100)
results = pt.Experiment([ideal_df], dataset.get_topics(), qrels, eval_metrics=[AP(rel=2), nDCG@10], names=["Oracle"])   

print(results)

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020')
qrels = dataset.get_qrels()
ideal_df = ideal_fusion("data/rl/2020/", qrels, top_k=100)
results = pt.Experiment([ideal_df], dataset.get_topics(), qrels, eval_metrics=[AP(rel=2), nDCG@10], names=["Oracle"])   
print(results)

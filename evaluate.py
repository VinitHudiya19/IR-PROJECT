"""Measure search quality: Precision@k, Recall@k, MRR."""
import pandas as pd


def parse_judgments(text, known_files):
    """Lines look like:  query | paper1.pdf, paper2.pdf   ->  ({query: {files}}, [problems])"""
    judgments, problems = {}, []
    for line in text.splitlines():
        if "|" not in line:
            continue
        query, files = line.split("|", 1)
        query = query.strip()
        relevant = {f.strip() for f in files.split(",") if f.strip()}
        unknown = relevant - set(known_files)
        if unknown:
            problems.append(f"'{query}': unknown file(s) {', '.join(sorted(unknown))}")
        relevant -= unknown
        if query and relevant:
            judgments[query] = relevant
    return judgments, problems


def evaluate(judgments, rankers, k=3):
    """rankers: {name: function(query) -> list of file names, best first}."""
    rows = []
    for name, rank in rankers.items():
        precision = recall = mrr = 0.0
        for query, relevant in judgments.items():
            ranked = rank(query)
            hits = len(set(ranked[:k]) & relevant)
            precision += hits / k
            recall += hits / len(relevant)
            mrr += next((1 / i for i, f in enumerate(ranked, 1) if f in relevant), 0.0)
        n = len(judgments)
        rows.append({"Method": name, f"Precision@{k}": precision / n, f"Recall@{k}": recall / n, "MRR": mrr / n})
    return pd.DataFrame(rows)

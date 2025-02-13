import os
import pandas as pd
import numpy as np
import json
import scipy
from tqdm import tqdm
from math_verify import parse, verify

cache_dir = "./cache"
sample_file = "global_step_100/samples.json"
sample_dir = "../../samples/sft/"
subject_tags = {
    "Algebra": "AL",
    "Intermediate Algebra": "IA",
    "Prealgebra": "PA",
    "Number Theory": "NO",
    "Geometry": "GT",
    "Precalculus": "PC",
    "Counting & Probability": "PR"
}

def score(f, desc=None, cache_dir=None):
    if cache_dir is not None:
        cache_f = "__".join(f.split(os.path.sep)[-3:-1])
        cache_path = os.path.join(cache_dir, cache_f + ".npz")
        try:
            return np.load(cache_path)["results"]
        except Exception:
            pass

    with open(f, "r") as fi:
        samples = json.load(fi)
    results = []
    for q, out in tqdm(samples.items(), desc=desc, disable=desc is None):
        try:
            correct = verify(parse(out["true"]), parse(out["sampled"]))
        except:
            correct = False
        results.append(correct)
    results = np.array(results)

    if cache_dir is not None:
        np.savez_compressed(cache_path, results=results)
    return results


def frozendict(d):
    return tuple(sorted(d.items()))


def model_slice(excluded=None, synthetic=None):
    return frozendict({
        "excluded": tuple(excluded or []),
        "synthetic": tuple(synthetic or [])
    })


def parse_filename(s, include_version=False):
    parts = s.split("__")
    excluded_subject = parts[1]
    synthetic_subject = parts[3]
    version = parts[5]
    r_subject_tags = {v: k for k, v in subject_tags.items()}

    def map_tags(tag):
        if tag == "NULL":
            return ()
        if "_" in tag:
            return tuple([r_subject_tags.get(t, t) for t in tag.split("_")])
        return r_subject_tags.get(tag, tag),

    data = {
        "excluded": map_tags(excluded_subject),
        "synthetic": map_tags(synthetic_subject),
    }
    if include_version:
        data["version"] = version if version != "NULL" else None
    return frozendict(data)


def ci(x, confidence=0.90):
    m, se = np.mean(x), scipy.stats.sem(x)
    return se * scipy.stats.t.ppf((1 + confidence) / 2., len(x) - 1)


print('computing scores...')
for i, model in enumerate(sorted(os.listdir(sample_dir))):
    path = os.path.join(sample_dir, model, sample_file)
    info = parse_filename(model)
    desc = f"({i + 1} / {len(os.listdir(sample_dir))})"
    scores = score(path, desc=desc, cache_dir=cache_dir)

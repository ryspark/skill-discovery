# TODO - 1. load dataset from huggingface
#        2. mix in from the generated dataset
#        3. coerce into correct sft format
# for the tests, pull out increasingly larger subjects
# should have (n subjects) * 3 total runs

import os
import datasets
import argparse
from collections import defaultdict

subject_tags = {
    "Algebra": "AL",
    "Intermediate Algebra": "IA",
    "Prealgebra": "PA",
    "Number Theory": "NO",
    "Geometry": "GT",
    "Precalculus": "PC",
    "Counting & Probability": "PR"
}

def tag(subjects):
    if not subjects:
        return "NULL"
    result = []
    for subject in subjects:
        result.append(subject_tags[subject])
    return "_".join(result)


def make_map_fn(split, prompt, synthetic_subjects=None, synthetic=None):
    def process_fn(example, idx):
        subject = example.pop("subject")
        data_source = "RLAIF/math"
        if subject in (synthetic_subjects or []):
            example = next(synthetic[subject])
            data_source = "<Synthetic>"
        question = example["problem"]
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt + "\n" + question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["answer"]
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'question': question,
                'answer': example["solution"]
            }
        }
        return data
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", default="../../scripts/questions/data/")
    parser.add_argument("--write_dir", default="../../data")
    parser.add_argument("--synthetic_ds", default="NULL")
    parser.add_argument("--excluded", nargs="*")
    parser.add_argument("--synthetic", nargs="*")
    args = parser.parse_args()

    dataset = datasets.load_dataset("RLAIF/math")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"loaded train ({len(train_dataset)}) and test data ({(len(test_dataset))}) from RLAIF/math")

    synthetic = None
    if args.synthetic_dir != "NULL" and args.synthetic_ds != "NULL":
        synthetic_data = datasets.load_from_disk(
            os.path.join(args.synthetic_dir, args.synthetic_ds)
        )
        synthetic = defaultdict(list)
        for entry in synthetic_data["train"]:
            synthetic[entry["subject"]].append(entry)
        synthetic = {key: iter(value) for key, value in synthetic.items()}
        print(f"loaded {len(synthetic)} synthetic train subjects")

    # https://arxiv.org/html/2409.12122v1#A2.F11
    prompt = "Please reason step by step, and put your final answer within \\boxed{}"

    train_dataset = train_dataset.filter(
        lambda row: row["subject"] not in (args.excluded or [])
    )
    train_dataset = train_dataset.map(
        function=make_map_fn("train", prompt, args.synthetic, synthetic),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", prompt),
        with_indices=True
    )

    n_synthetic = sum(int(row["data_source"] == "<Synthetic>") for row in train_dataset)
    print(f"train ({len(train_dataset)}), test ({len(test_dataset)}) done")
    print(f"train has {n_synthetic} synthetic examples")
   
    excl = ""
    if args.excluded:
        excl = map(lambda x: x.replace(" ", "_"), args.excluded)
    synth = ""
    if args.synthetic:
        synth = map(lambda x: x.replace(" ", "_"), args.synthetic)

    version_tag = os.path.splitext(str(args.synthetic_ds))[0]
    tag = f"excluded__{tag(args.excluded)}__synthetic__{tag(args.synthetic)}__version__{version_tag}"
    os.makedirs(os.path.join(args.write_dir, tag))
    train_dataset.to_parquet(
        os.path.join(args.write_dir, tag, "train.parquet")
    )
    test_dataset.to_parquet(
        os.path.join(args.write_dir, tag, "test.parquet")
    )
    print(f"Wrote data to {os.path.join(args.write_dir, tag)}")

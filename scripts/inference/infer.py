import json
import sys
import os
import requests
import concurrent
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

model = sys.argv[1]
save = sys.argv[2]
port = sys.argv[3]

def generate_requests(data):
    reqs = []
    for row in data:
        content = row["problem"]
        reqs.append({
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "max_tokens": 4096,
            "answer": row["solution"]
        })
    return reqs

url = f"http://localhost:{port}/v1/chat/completions"
dataset = load_dataset("RLAIF/math")
reqs = generate_requests(dataset["test"])
print(f'[{os.path.sep.join(model.split(os.path.sep)[-3:-1])}] {len(reqs)=}')
responses = {}
f = lambda p: requests.post(url, json=p).json()

with concurrent.futures.ThreadPoolExecutor() as executor:
    desc = f"Answering questions"
    golds = [r.pop('answer') for r in reqs]
    answers = list(tqdm(executor.map(f, reqs), total=len(reqs), desc=desc))
    for req, result, gold in zip(reqs, answers, golds):
        answer = result['choices'][0]['message']['content']
        responses[req["messages"][0]["content"]] = {
            "sampled": answer, "true": gold
        }

with open(save, "w+") as f:
    json.dump(responses, f, indent=4)

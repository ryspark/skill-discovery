import re
import requests
import random
import concurrent
import math_verify
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

# v1: seed from -10000 to 10000, answers are too large, max_tokens = 8000
# v2: seed from -100 to 100, but 5 of them, max_tokens = 2048

save = "data/v2_t1.0_seeded_qwen2.5_7b_instruct.hf"
model = "Qwen/Qwen2.5-7B-Instruct"
#model = "meta-llama/Llama-3.1-8B-Instruct"
#model = "Qwen/Qwen2.5-Math-7B-Instruct"
#model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
temperature = 1.0
url = "http://localhost:30000/v1/chat/completions"
question_prompt = """
Generate a 1-3 sentence medium to difficult question for {subject}.
Do NOT include the answer. All questions should have a single, unambiguous
correct answer (no open ended questions).
Change the STYLE, SUBCATEGORY, and VALUES in your question according
to these random seeds: {seed1} {seed2} {seed3} {seed4} {seed5}.
Begin the text of your question with "QUESTION: <your question>".
DO NOT INCLUDE THE ANSWER.
"""
answer_prompt = """You MUST put your final answer in \\boxed{}"""

def parse_question(text):
    return text[text.rfind("QUESTION") + 10:]

def parse_answer(text):
    try:
        _, result = math_verify.parse(text)
        return result
    except ValueError:
        return None

def generate_requests(ds, split, n=-1, request_answers=False):
    reqs = []
    for q in ds[split]:
        if n > 0 and len(reqs) >= n:
            break
        seeds = {f"seed{i}": random.randint(-100, 100) for i in range(1, 6)}
        if not request_answers:
            content = question_prompt.format(subject=q['subject'], **seeds)
            reqs.append({
                "model": model,
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": temperature,
                "metadata": dict(subject=q["subject"])
            })
        else:
            content = q["problem"] + "\n\n "+ answer_prompt
            reqs.append({
                "model": model,
                "messages": [
                    {"role": "user", "content": content}
                ],
                "max_tokens": 2048
            })
    return reqs

ds = load_dataset("RLAIF/math")
print("loaded dataset")

reqs = {
    "train": generate_requests(ds, "train")
}
for k, v in reqs.items():
    print(f"loaded {k} with {len(v)} items")

responses = {k: [] for k in reqs}
f = lambda p: requests.post(url, json=p).json()
failed = 0
for k, v in reqs.items():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        desc = f"Generating questions for {k}"
        metadatas = [r.pop("metadata") for r in v]
        questions = list(tqdm(executor.map(f, v), total=len(v), desc=desc))
        for result, metadata in zip(questions, metadatas):
            q = parse_question(result['choices'][0]['message']['content'])
            if q is not None:
                responses[k].append({"problem": q, **metadata})

    ans_reqs = generate_requests(responses, k, request_answers=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        desc = f"Generating responses for {k}"
        answers = list(tqdm(executor.map(f, ans_reqs), total=len(ans_reqs), desc=desc))
        for i, result in enumerate(answers):
            ans = result['choices'][0]['message']['content']
            r = parse_answer(ans)
            if r is not None:
                responses[k][i]["answer"] = r
                responses[k][i]["solution"] = ans
            else:
                failed += 1

    filtered = {k: [] for k in responses}
    for split in responses:
        for response in responses[split]:
            if "answer" in response:
                filtered[split].append(response)

print("no. failed to parse:", failed)
dataset = DatasetDict({k: Dataset.from_list(v) for k, v in filtered.items()})
dataset.save_to_disk(save)
print(dataset)
print("saved to disk " + save)

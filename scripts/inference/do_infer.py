import os

model_dir = "../../models/sft"
sample_dir = "../../samples/sft"
step = "global_step_100"
skip_existing = True
cmd = "sbatch --export MODEL={model},SAVE={save},PORT={port} infer.sh"

for i, model in enumerate(os.listdir(model_dir)):
    load_from = os.path.join(model_dir, model, step)
    save = os.path.join(sample_dir, model, step, "samples.json")
    if skip_existing and os.path.exists(save):
        print("skipping existing samples for " + model)
        continue
    
    os.makedirs(os.path.split(save)[0], exist_ok=True)
    cmd_ = cmd.format(
        model=os.path.realpath(load_from),
        save=os.path.realpath(save),
        port=30000 + i
    )
    os.system(cmd_)

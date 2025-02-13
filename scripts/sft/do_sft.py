import sys
import os
import glob
import wandb

base = "../../data"
base_save = "../../models/sft"
fmt = "excluded__*__synthetic__*__version__*"
cmd = "sbatch --export DATA_DIR=\"{ds}\",EXP_NAME=\"{exp}\",SAVE_DIR=\"{save}\" template.sh"

ignore_prev = True
wandb_username = "orangese"
wandb_project = "qwen0.5b-sft"
api = wandb.Api()
if ignore_prev:
    runs = api.runs(f"{wandb_username}/{wandb_project}")
    runs = [run.name for run in runs]

try:
    dry = (sys.argv[1] == "1")
except IndexError:
    dry = False

cmds = []
for ds in glob.glob(os.path.join(base, fmt)):
    _, exp = os.path.split(ds)
    if ignore_prev and exp in runs:
        print(f"skipping existing run {exp}")
        continue
    save = os.path.join(base_save, exp)
    cmd_ = cmd.format(ds=os.path.abspath(ds), exp=exp, save=save)
    cmds.append(cmd_)

print("=" * 80)
print(f"found {len(cmds)} runs to do ({dry=})")
print("=" * 80)

for cmd_ in cmds:
    print(cmd_)

print()
if input("Continue? (q to exit) ").lower() not in ['q', 'n']:
    for cmd_ in cmds:
        if not dry:
            os.system(cmd_)
        else:
            print("<dry run>")

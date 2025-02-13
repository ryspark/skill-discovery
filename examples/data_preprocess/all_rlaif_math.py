import os

mode = 'parallel'
subjects = [
    # Ordered by decreasing size
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Number Theory",
    "Geometry",
    "Precalculus",
    "Counting & Probability"
]
subjects = list(reversed(subjects))
#synthetic = "v1_t1.0_seeded_qwen2.5_7b_instruct.hf"
synthetic = "v2_t1.0_seeded_qwen2.5_7b_instruct.hf"
cmd = "python3 rlaif_math.py --synthetic_ds {ds} --excluded {excluded} --synthetic {synthetic}"

if mode == 'sequential':
    # This is for sequential (removing i subjects as a time, i=0...n)
    for i in range(len(subjects)):
        joined = " ".join(map(lambda x: f"'{x}'", subjects[:i]))

        # First generate excluded subjects for baseline
        cmd = "python3 rlaif_math.py --synthetic_ds NULL"
        if len(joined) > 0:
            cmd += " --excluded " + joined
        os.system(cmd)

        # Then generate synthetic subjects for testing
        if i != 0:
            cmd = f"python3 rlaif_math.py --synthetic_ds {synthetic}"
            if len(joined) > 0:
                cmd += " --synthetic " + joined
            os.system(cmd)

elif mode == 'parallel':
    # This is for parallel (removing 1 subject at a time)
    for i in range(len(subjects)):
        joined = subjects[i]

        # First generate excluded subjects for baseline
        cmd = f"python3 rlaif_math.py --synthetic_ds NULL --excluded '{joined}'"
        os.system(cmd)

        # Then generate synthetic subjects for testing
        cmd = f"python3 rlaif_math.py --synthetic_ds {synthetic} --synthetic '{joined}'"
        os.system(cmd)


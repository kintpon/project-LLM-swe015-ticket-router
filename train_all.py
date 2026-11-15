"""One command for the whole pipeline: data -> M1 -> M2 -> M3 -> M3s -> M4 -> report -> docs.
Each step is a separate script; this file only runs them in order and stops at the first failure.
Usage: python train_all.py            (full run; M3/M3s need a GPU to finish in reasonable time)
       python train_all.py --smoke    (300 rows, 1 epoch per model; results are not written)"""
import subprocess
import sys

STEPS = ["src.data", "scripts.data_checks", "src.m1_tfidf", "src.m2_ffn", "src.m3_bert",
         "src.m4_distill", "src.m4_quantize", "src.make_report", "scripts.fill_docs"]
TRAINING = {"src.m1_tfidf", "src.m2_ffn", "src.m3_bert", "src.m4_distill", "src.m4_quantize"}

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    for step in STEPS:
        if smoke and step not in TRAINING and step != "src.data":
            continue  # report and docs read the real results, so they are skipped in a smoke run
        args = [sys.executable, "-m", step] + (["--smoke"] if smoke and step in TRAINING else [])
        print(f"== {' '.join(args[1:])}", flush=True)
        subprocess.run(args, check=True)
    print("all steps done")

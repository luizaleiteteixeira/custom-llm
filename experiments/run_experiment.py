import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def make_variant(label, steps, n_embd, n_head, n_layer, block_size, lr=0.001):
    src = (REPO / "custom_llm.py").read_text()
    src = re.sub(r"^TRAINING_STEPS = \d+.*$", f"TRAINING_STEPS = {steps}      # experiment: {label}",
                 src, count=1, flags=re.M)
    src = re.sub(r"^LEARNING_RATE = [\d.]+.*$", f"LEARNING_RATE = {lr}", src, count=1, flags=re.M)
    src = re.sub(
        r"^SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = .*$",
        f"SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = 42, {n_embd}, {n_head}, {n_layer}, {block_size}, 32",
        src, count=1, flags=re.M,
    )
    out = REPO / f"_exp_{label}.py"
    out.write_text(src)
    return out

if __name__ == "__main__":
    label, steps, n_embd, n_head, n_layer, block_size = sys.argv[1:7]
    lr = float(sys.argv[7]) if len(sys.argv) > 7 else 0.001
    path = make_variant(label, int(steps), int(n_embd), int(n_head), int(n_layer), int(block_size), lr)
    print(path)

# Extended experiments: more steps, a bigger model — does it get "smarter"?

My professor asked me to go further than the required 3,000-step baseline: train for
longer, try a bigger model, and see how much "smarter" it actually gets. This is
extra work I did under an **earlier version of this assignment, before the 48-case
language test and the chat script existed** — see [README.md](README.md) for my
actual, required submission (two experiments, the full test comparison, and chat
proof). This page doesn't replace that. Every run below just changes one setting at
a time on the same notebook (`custom_llm.py`, which is the script version of
`custom_llm.ipynb`).

**Runs**, all on the same made-up classroom corpus (4,632 unique passages, seed 42,
CPU, learning rate 0.001, with warmup and cosine decay):

| Run | Steps | Params | Time | Final train loss | Final val loss |
|---|---|---|---|---|---|
| Baseline (required submission) | 3,000 | 111,872 | 9.7s | 0.6956 | **0.7057** |
| More steps | 5,000 | 111,872 | 13.7s | 0.6976 | **0.7068** |
| More steps | 20,000 | 111,872 | 55.4s | 0.6859 | **0.6969** |
| More steps | 100,000 | 111,872 | 317.0s (5m17s) | 0.6860 | **0.8523** |
| Bigger model (4 blocks, 8 heads, 128-dim, 64-token context) | 20,000 | 818,944 (7.3x bigger) | 197.2s | 0.6851 | **0.6976** |

All the files for each run are here: [experiments/steps5k/](experiments/steps5k/),
[experiments/steps20k/](experiments/steps20k/), [experiments/steps100k/](experiments/steps100k/),
[experiments/bigarch20k/](experiments/bigarch20k/) — same set of files as the
baseline's [archive_pre_eval_run/](archive_pre_eval_run/) folder (`history.json`,
`config.json`, `checkpoint.json`, `model.pt`, `samples/`, `training_curves.svg`, and
so on — that folder used to be called `evidence/`, before the language test existed;
I renamed it to make space for my two current required experiments). The scripts I
used to make these (`experiments/run_experiment.py`, `experiments/run_all.sh`) are
in the repo too, so anyone can redo this.

![validation loss vs training steps across all five runs, log x-axis](experiments/comparison_curves.svg)

## Does training longer make it "smarter"?

**No, not after a certain point — and it can actually get worse.** The validation
loss drops from 0.706 (3k steps) to 0.697 (20k steps), which is a small real
improvement, but then it **goes back up to 0.852 at 100k steps**, even though the
training loss stays almost flat (0.686) the whole time. That's a classic sign of
overfitting: after about 20,000 steps, the model just keeps getting better at
matching the exact training examples, without actually getting better at anything
new — and the validation score gets worse because of it. On this small, repetitive
corpus, training longer does not simply mean a better model.

What surprised me even more: **the generated text barely changes at all after step
1,500**, in every single run. The 3k run, the 5k run, the 20k run, even the 100k
run — they all end up producing almost the exact same handful of sentences (like
*"our school has a question about the new educator and lesson."* or *"the consumer
compared the merchandise after checking the price."*, with only tiny word swaps like
"tutor" instead of "educator"). If I only looked at the generated text, I couldn't
tell the 3,000-step run apart from the 100,000-step run — even though their
validation losses are actually quite different (0.706 vs 0.852). This taught me that
just reading the generated samples is not a reliable way to judge how "smart" a
model is — the real numbers can be very different while the small set of sentences
the model likes to produce looks the same.

## Does a bigger model make it "smarter"?

**Not on this corpus.** The bigger version of the model (4 blocks instead of 2, 8
heads instead of 4, 128 numbers per embedding instead of 64, 7.3 times more
parameters) ends up with basically the same validation loss at 20,000 steps as the
small model at 20,000 steps (0.6976 vs 0.6969 — even smaller than the normal
difference I already saw between the 5k and 20k runs of the small model). Extra size
didn't help, because the real limit here isn't the model — it's the **data**. My
corpus only has around 7 sentence patterns and 133 different words, and even the
small model already has more than enough room to fully learn that. So giving it more
parameters just gives it more room it doesn't need. A bigger model would probably
only help if I also gave it a bigger, more varied corpus that actually needs that
extra room.

## What about the words the model learned?

I checked which words are closest to `customer` (using all 64 or 128 numbers of its
embedding, not just a couple) across all five runs:

| Run | Closest 5 words to "customer" | Top similarity |
|---|---|---|
| Baseline (3k) | client, buyer, subscriber, consumer, shopper | 0.985 |
| 5k steps | client, buyer, consumer, subscriber, shopper | 0.990 |
| 20k steps | consumer, client, buyer, shopper, subscriber | 0.696 |
| 100k steps | client, consumer, buyer, shopper, subscriber | 0.902 |
| Bigger model, 20k | buyer, shopper, consumer, client, subscriber | 0.786 |

The **same 5 words** show up every single time, no matter how long I trained or how
big the model was — that grouping is learned almost right away and stays stable. But
the actual **similarity numbers** jump around a lot (0.70 to 0.99) without any clear
pattern tied to more training or a bigger model. To me, this says the model learns
*which* words belong together very quickly and reliably, but the exact distances
between them stay a bit noisy, and more training or more parameters doesn't really
clean that up on a corpus this small and repetitive.

## Honest takeaway

On a small, repetitive, made-up corpus like this one, training for longer or using a
bigger model hits a wall almost immediately (well before 20,000 steps here), and
pushing past that wall risks making things worse instead of better. If I actually
want a "smarter" model on this corpus, the thing that would really help is a bigger
and more varied **corpus** — not more steps and not more parameters. That's exactly
what led me to try the corpus-extension experiment described in the main
[README.md](README.md).

## How to reproduce this

```
source .venv/bin/activate
python experiments/run_experiment.py steps20k 20000 64 4 2 48   # writes _exp_steps20k.py
python _exp_steps20k.py                                          # runs it, saves to llm_runs/
```
`run_experiment.py <label> <steps> <n_embd> <n_head> <n_layer> <block_size> [learning_rate]`
makes a copy of `custom_llm.py` with those settings swapped in, so it never touches
the original notebook or script. `experiments/run_all.sh` is what I used to run all
four experiments above, one after another.

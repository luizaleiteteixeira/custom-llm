# My Custom LLM Experiment

Class 4 assignment for *From Zero to AI Agents* — training Karpathy's actual nanoGPT
transformer from scratch on a small word-token corpus, testing it against a fixed
48-case language eval suite before and after training, extending the corpus for
specific eval skills, and chatting with the trained model. This README is the
grading entry point: it links every piece of evidence so it can be reviewed without
rerunning the notebook.

**Two required experiments, each executed end-to-end:**

| | Notebook | Evidence folder | Results ZIP |
|---|---|---|---|
| **Starter** (classroom corpus only) | [evidence/custom_llm_starter.ipynb](evidence/custom_llm_starter.ipynb) | [evidence/starter/](evidence/starter/) | [evidence/starter_results.zip](evidence/starter_results.zip) |
| **Expanded** (classroom + 4 extension categories) | [evidence/custom_llm_expanded.ipynb](evidence/custom_llm_expanded.ipynb) / [custom_llm.ipynb](custom_llm.ipynb) (same file, repo root) | [evidence/expanded/](evidence/expanded/) | [evidence/expanded_results.zip](evidence/expanded_results.zip) |

Also: [evals/language_evals.json](evals/language_evals.json) (the fixed 48-case suite,
unchanged), [run_evals.py](run_evals.py) (the scorer, also usable standalone),
[chat.py](chat.py) (terminal chat interface) and [evidence/chat/](evidence/chat/)
(real chat evidence). [ASSIGNMENT.md](ASSIGNMENT.md) is the assignment spec;
[EXPERIMENTS.md](EXPERIMENTS.md) is optional supplementary work (more training
steps / a bigger model) done under the *previous* version of this assignment,
before the eval suite existed — see [archive_pre_eval_run/](archive_pre_eval_run/)
for that original single-experiment submission, kept for continuity.

## My three choices (same for both experiments)

| Choice | Value | Reason |
|---|---|---|
| Training steps | `TRAINING_STEPS = 3000` | The assignment's suggested starting budget; a 10-step dry run confirmed the environment works in seconds on CPU. |
| Learning rate | `LEARNING_RATE = 0.001` | The suggested default, with warmup + cosine decay handled inside the training cell. Too large risks the loss diverging past the minimum; too small barely moves 111K–132K parameters in 3,000 steps. |
| Corpus | `CORPUS = "classroom"` in both experiments — **starter**: no files in `corpus/`; **expanded**: 5 files added (below) | Keeping steps/LR fixed between the two runs isolates the corpus as the only variable, so the eval-score comparison is fair. |

## Corpus sources and permissions

**Starter:** only the notebook's built-in synthetic classroom sentences. No external
files, no permission concerns.

**Expanded:** 5 files I wrote myself, entirely synthetic, in `corpus/` (Git-ignored
per the assignment's convention, so not literally in this repo, but every source
line is reproducible from the generator script below and fully visible in
`evidence/expanded/corpus.txt` and `corpus_manifest.json`):

| File | Lines | Unique passages added |
|---|---|---|
| `extension_opposites.txt` | 160 | 160 |
| `extension_negation.txt` | 43 | 87 (duplicates collapsed) |
| `extension_everyday_knowledge.txt` | 17 | 17 |
| `extension_categories_and_analogies.txt` | 30 | 30 |
| `extension_vocab_filler.txt` | 12 | 12 |
| **Total** | **262 lines** | **306 new unique passages** |

No PDFs were used (all plain `.txt`), so there is nothing to check for extraction
warnings — `corpus_manifest.json` confirms 0 ignored files and 0 warnings for both
runs.

## Why these four extension categories

The eval suite's 24 `extend_corpus` cases cover 8 skills (3 cases each): grammar,
opposites, negation, reference, sequence, spatial_relations, everyday_knowledge,
categories_and_analogies — none of which exist anywhere in the starter corpus. I
chose **opposites, negation, everyday_knowledge, and categories_and_analogies**
(4 of the required minimum of 2) because they're learnable from short, self-contained
sentences without needing multi-turn narrative context, unlike reference/sequence.
I did not attempt grammar, reference, sequence, or spatial_relations.

**Getting real signal out of this took three corrected iterations, which I think is
the most important thing I learned from this whole exercise:**

1. **First attempt:** taught the four concepts using words that deliberately avoided
   every word the evals use (e.g. "sparrow" instead of "robin", "umbrellas" instead of
   singular "umbrella"). Result: all 24 `extend_corpus` cases stayed
   `out_of_vocabulary` in both stages — 0% coverage, no signal whatsoever. A
   word-token model with no subword structure does not generalize across synonyms;
   teaching the *concept* with *different vocabulary* taught the model nothing it
   could use on these specific tests.
2. **Second attempt:** reused the evals' own prompt and answer words in new
   sentences. Still stuck at 0% coverage — because `run_evals.py` requires **all
   four** multiple-choice words to be in-vocabulary before a case is scored at all,
   and I had only covered the *correct* answers, not the three wrong distractor
   choices per case (e.g. "heavy", "loud", "yellow", "pillow").
3. **Final version** (what's actually in `corpus/` and trained below): added
   `extension_vocab_filler.txt` covering every remaining distractor word, plus a
   couple of extra sentences for words that had appeared only once and happened to
   land entirely in the held-out validation split by chance (`tree`, `carrot`,
   `asleep`, `metal`) — a reminder that a single occurrence isn't reliably enough to
   survive a random 90/10 split. Coverage finally reached 33/48 (68.75%), with the
   12 cases in my 4 target categories actually testable.

Every version was checked with a leakage script before training (see below), and the
notebook's own automatic checks (`eval_separation.json`, `reject_eval_leakage`)
independently confirm none of the 48 reserved test prompts ever appear in any corpus
file, in either experiment.

## Corpus-separation evidence

- [evidence/starter/eval_separation.json](evidence/starter/eval_separation.json) and
  [evidence/expanded/eval_separation.json](evidence/expanded/eval_separation.json):
  both report **160 excluded passages** (classroom-generated sentences that happened
  to contain a reserved test prefix, withheld automatically before the train/val
  split), covering all 16 `starter_patterns` case IDs, in both runs.
- The notebook's own `validate_corpus_location` and `reject_eval_leakage` functions
  (from `run_evals.py`) run automatically during `Run All` and would raise an error
  and stop the run if any corpus file, or the final assembled corpus, contained a
  reserved test prompt. Both experiments completed without that error.
- I additionally ran a standalone leakage check (the same `normalized(prompt) in
  content` logic from `run_evals.py`) against every `extension_*.txt` file before
  each training run, confirming zero matches — see the prediction cell in
  [custom_llm.ipynb](custom_llm.ipynb) for the exact iteration history.

## My prediction vs. what actually happened

Both experiments' predictions are written in the "My prediction" cell of their
respective executed notebooks (linked above), written *before* running that
experiment. Summary:

- **Starter:** I predicted `starter_patterns` would jump well above the 25% random
  baseline and `extend_corpus` would stay near 0% and mostly `out_of_vocabulary`.
  **Actual:** `starter_patterns` 37.5% → **100%**, `starter_transfer` 37.5% → 50%,
  `extend_corpus` stayed at **0.0%** in both stages (all 24 cases
  `out_of_vocabulary`) — matching the prediction closely.
- **Expanded (final iteration):** I predicted coverage on `extend_corpus` would
  reach close to 12/24 with scores "meaningfully above 25% random" on the categories
  I taught, mixed/weak elsewhere. **Actual:** coverage reached **11/24** scorable
  (one category, `opposites`, ended up fully scorable at 3/3 too, for 35/48 overall),
  and scores were genuinely mixed: `negation` and `everyday_knowledge` both jumped to
  66.7%, `categories_and_analogies` reached 33.3%, but `opposites` actually *dropped*
  to **0%** trained (from a lucky 1/3 untrained, on the single case that happened to
  be scorable before training) despite being fully in-vocabulary — a harder transfer
  than I expected, discussed below.

## My runs

| | Starter | Expanded |
|---|---|---|
| Completed steps | 3,000 / 3,000 | 3,000 / 3,000 |
| Elapsed time | 9.48s | 9.79s |
| Hardware | Apple Silicon Mac, macOS-15.7.4-arm64, CPU only | same |
| Software | Python 3.13.15, PyTorch 2.14.0 | same |
| Parameters | 111,872 | 132,032 (larger vocabulary embedding table) |
| Vocabulary size | 136 tokens | 451 tokens |
| Train / validation documents | 4,132 / 460 | 4,408 / 490 |
| Unknown-token rate (train / val) | 0.00% / 0.00% | 0.00% / 0.16% |

Both runs use the same seed (42) and settings, so steps/hardware/elapsed are barely
different — the vocabulary and parameter-count differences are entirely from the
added corpus text. Loss values are **not directly comparable between the two runs**:
random-guessing baseline is `ln(vocab_size)`, i.e. 4.91 for the starter's 136-token
vocabulary vs. 6.11 for the expanded run's 451-token vocabulary — exactly what both
runs' step-0 losses show (4.93 and 6.10 respectively).

## Evidence: loss curves and samples

**Starter** ([evidence/starter/training_curves.svg](evidence/starter/training_curves.svg)):

![starter loss curve](evidence/starter/training_curves.svg)

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.9263 | 4.9275 |
| 1500 | 0.6821 | 0.7182 |
| 3000 | 0.6783 | 0.7061 |

**Expanded** ([evidence/expanded/training_curves.svg](evidence/expanded/training_curves.svg)):

![expanded loss curve](evidence/expanded/training_curves.svg)

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 6.0984 | 6.1134 |
| 1500 | 0.9899 | 0.7184 |
| 3000 | 0.8573 | 0.7061 |

Both are fixed panels of ≤20 training + ≤20 validation documents, mean loss over
non-padding next-token targets — small estimates, not full-corpus measurements.

**Samples** (full files in [evidence/starter/samples/](evidence/starter/samples/) and
[evidence/expanded/samples/](evidence/expanded/samples/)), same generation settings
(temperature 0.8, fixed seed) at every checkpoint:

- **Untrained (step 0), both runs:** pure word salad, e.g. *"pear professor bond
  doctor course harvest team physician journey checking buyer delivery..."*
- **Halfway / final (both runs):** grammatical template sentences, e.g. *"the
  consumer compared the merchandise after checking the price ."*, *"our school has
  a question about the new educator and lesson ."* Visible change is almost entirely
  between step 0 and step 1500; step 1500 → 3000 barely moves the generated text in
  either run, even though the expanded run's *loss* keeps changing in that window
  (train 0.99 → 0.86) — the loss is still fitting the larger vocabulary while the
  small fixed sample set has already saturated.

## Token, embedding, gradient and weight update

Tracing the word **`customer`** through the starter run
([evidence/starter/inspection.json](evidence/starter/inspection.json),
[tokenization.json](evidence/starter/tokenization.json)):

- Token → ID: `customer` → **id 28** in the 136-token starter vocabulary. (In the
  expanded run the same word is **id 92**, since the vocabulary — sorted
  alphabetically after 3 special tokens — grew to 451 entries and shifted every ID.)
- Embedding **before** training (first 5 of 64 numbers): `[-0.0576, -0.0048, 0.0426,
  0.0193, 0.0156]`
- Embedding **after** 3,000 steps: `[0.0366, -0.0182, 0.1330, 0.1060, 0.0630]`
- First saved parameter update (coordinate 0 of `customer`'s embedding, the very
  first training step): `before = -0.057592`, `gradient = 0.000693`,
  `learning_rate = 1e-05` (warmup hadn't ramped up yet), `after = -0.057602`.
- Next-token probabilities for prefix **"the customer"**: before training, top guess
  was `customer` itself at only ~1.6% (near-random over 136 tokens); after training,
  the model strongly favors grammatical continuations like `ordered`, `reviewed`,
  `recommended`, `selected`, `compared` — all sensible completions of "the customer
  ___".
- `customer`'s trained nearest neighbors (full 64D cosine similarity, both runs):
  **client, buyer, subscriber, consumer, shopper** — this retail-context cluster
  forms almost immediately and is unaffected by the unrelated extension domains
  added in the expanded run.

## Attention and temperature

Causal self-attention lets each position combine information only from itself and
*earlier* tokens — the measured attention rows for "the customer" show position 0
attending 100% to itself (nothing precedes it) and later positions blending across
earlier ones, never assigning weight to future positions (masked to zero). That
masking is what makes this a left-to-right causal language model.

Temperature divides the logits before the softmax: lower temperature (0.3) sharpens
toward the highest-probability tokens (more repetitive), higher temperature (1.2)
flattens the distribution (more varied, less certain). No weight changes between the
three — `temperature_comparison.json` in both evidence folders reuses the exact same
trained `model.pt` for all three; only the sampling step changes.

## Language evals: how scoring works

An eval is a fixed test — a prompt, four single-word answer choices, an answer key,
and the model's actual result. The runner (`run_evals.py`) feeds **only the prompt**
into the trained nanoGPT (no answer choices, no key), reads the next-token
probabilities, and picks whichever of the 4 choices has the highest probability.
Correct = 1, incorrect or tied = 0. A case is marked `out_of_vocabulary` (scored 0 in
the all-case rate, excluded from "scorable accuracy") if the prompt or **any of the 4
choices** contains a word outside the model's learned vocabulary. A separate,
unconstrained 24-token continuation is also generated and saved — this free text is
**not** what's scored, only inspected. See [evals/README.md](evals/README.md) for the
full methodology.

## Four-experiment eval comparison (required evidence)

All four complete result sets:
[evidence/starter/language_evals/untrained/](evidence/starter/language_evals/untrained/) ·
[evidence/starter/language_evals/final/](evidence/starter/language_evals/final/) ·
[evidence/expanded/language_evals/untrained/](evidence/expanded/language_evals/untrained/) ·
[evidence/expanded/language_evals/final/](evidence/expanded/language_evals/final/)
(each has `eval_cases.json`, `eval_results.json`, `eval_results.csv`,
`eval_summary.json`) — comparison summaries also in
[evidence/starter/language_eval_comparison.json](evidence/starter/language_eval_comparison.json)
and [evidence/expanded/language_eval_comparison.json](evidence/expanded/language_eval_comparison.json).

| Experiment / stage | All-case success | Scorable accuracy | Coverage | `starter_patterns` | `starter_transfer` | `extend_corpus` |
|---|---|---|---|---|---|---|
| Starter, untrained | 18.75% (9/48) | 37.5% | 50.0% | 37.5% | 37.5% | 0.0% |
| Starter, trained | 41.67% (20/48) | 83.3% | 50.0% | **100%** | 50.0% | 0.0% |
| Expanded, untrained | 18.75% (9/48) | 25.7% | 72.9% | 25.0% | 50.0% | 4.2% |
| Expanded, trained | **60.42%** (29/48) | 82.9% | 72.9% | **100%** | **100%** | **20.8%** |

`extend_corpus` category breakdown, expanded run only (trained stage; untrained was
0% everywhere except `opposites` at 33% by chance on 1 lucky case):

| Category | Untrained | Trained | Coverage | Notes |
|---|---|---|---|---|
| opposites | 33.3% (1/3, lucky guess — only 1 of 3 cases was scorable untrained) | **0%** (0/3, all 3 now scorable) | 100% | Training made this *worse*, not better — see failure discussion below |
| negation | 0% | **66.7%** (2/3) | 100% | Biggest real win |
| everyday_knowledge | 0% | **66.7%** (2/3) | 100% | Second biggest win |
| categories_and_analogies | 0% | **33.3%** (1/3) | 66.7% | 1 case still out_of_vocabulary |
| grammar | 0% | 0% | 0% (untouched) | Not attempted — fully out_of_vocabulary, as expected |
| reference | 0% | 0% | 0% (untouched) | Not attempted |
| sequence | 0% | 0% | 0% (untouched) | Not attempted |
| spatial_relations | 0% | 0% | 0% (untouched) | Not attempted |

**Did the corpus extension work?** Partially, and unevenly. `negation` and
`everyday_knowledge` improved substantially — both went from impossible (0%
coverage) to genuinely above the 25% random baseline. `categories_and_analogies`
improved less. `opposites` became scorable but never got a single case right, even
though I taught both the "the opposite of X is Y" sentence structure (with other
word pairs) and the hot/cold, empty/full, noisy/quiet associations (in other
sentence forms) — the model apparently could not combine a learned *structure* with
a learned *association* it had never seen paired together, in only 3,000 steps on a
132K-parameter model. This is a genuine limitation, not a bug: `starter_patterns` and
`starter_transfer`, which test patterns the model saw *already combined* in training,
both reached 100%. The four categories I never touched stayed exactly where the
starter run left them (0%, out_of_vocabulary) — confirming the assignment's own
warning that more training steps cannot supply missing vocabulary or patterns.

## Chat interface

**Launch it yourself:**
```
source .venv/bin/activate   # after: python3 -m venv .venv && pip install -r requirements.txt
python chat.py --model evidence/expanded/model.pt --transcript results/my-chat.json
```
Type any prompt, `/quit` to exit. Each prompt starts a fresh context (no memory
between turns); it never updates the model's weights. `evidence/starter/model.pt` /
`evidence/expanded/model.pt` are the two trained models from this submission —
either can be loaded the same way.

**Evidence of real interactions** (model: `evidence/expanded/model.pt`, run
`20260916T203902_301104Z`, model SHA-256 `d2903c78c64a59df9d8c897d562dee45bf038b1e285bf2ce59f36d173d5fa092`):

![chat screenshot](evidence/chat/chat_screenshot.png)

Full transcript: [evidence/chat/chat_expanded.json](evidence/chat/chat_expanded.json)
(4 turns). For contrast, 2 turns against the *starter* model (before the corpus
extension) are in
[evidence/chat/chat_starter.json](evidence/chat/chat_starter.json) — notably, the
starter model returns `[empty response]` and flags "hot", "opposite" as unknown
words for the exact same prompt the expanded model can at least attempt.

**One concrete chat failure:** prompting `"the weather today is"` against the
expanded model returns `red .` and flags `weather` as an unknown word — the model
has no concept of weather in either corpus, so instead of admitting it doesn't know,
it falls back to a frequent word borrowed from an unrelated learned pattern (the
negation corpus's color sentences). Similarly, `"the opposite of hot is"` returns
`out .`, not `cold` — matching the eval's `opposites` failure above: the model
learned the sentence *shape* and the word *association* separately but cannot
reliably combine them at generation time either.

## What I learned

1. **Corpus and held-out split:** the starter corpus teaches narrow business/retail
   sentence templates; my extension adds four more narrow domains the same way.
   Held-out validation tests recombination of the *same* templates, not
   generalization to unseen domains — the eval suite's `extend_corpus` group is a
   genuinely different, much harder test of exactly that unseen-domain gap.
2. **Token vs. ID vs. embedding:** a token is a text unit (word/punctuation); an ID
   is its arbitrary lookup index (`customer` → 28 in one vocabulary, 92 in another —
   the number itself means nothing); an embedding is the 64-number row the network
   actually learns for that ID, random at init and meaningfully positioned after
   training.
3. **Loss, gradient, and updates:** loss is cross-entropy on the actual next token;
   backpropagation computes the gradient for all ~112K–132K parameters;
   AdamW nudges each one by a tiny, schedule-dependent step. Different vocabulary
   sizes give different random-guessing loss baselines (`ln(vocab_size)`), so the
   starter and expanded runs' losses are not directly comparable — exactly why
   `config.json` in each evidence folder records its own `vocabulary_size`.
4. **Attention and temperature:** causal masking prevents any position from seeing
   the future; temperature reshapes sampling probabilities at inference time without
   ever touching a weight.
5. **Eval separation is mechanical, not semantic:** `reject_eval_leakage` catches
   exact contiguous prompt matches, not paraphrases or "teaching to the test" in
   spirit. I had to self-police beyond the automated check (e.g. avoiding "a puppy
   grows into a dog" verbatim even though a slightly different sentence would have
   technically passed).
6. **Vocabulary coverage is an all-or-nothing gate per case:** a case needs *every
   one* of its 4 choices in-vocabulary to be scored at all — teaching only the
   correct answer's vocabulary is not enough; distractor words matter just as much.
7. **Learning a structure and learning an association are not the same as learning
   to combine them:** `opposites` had full vocabulary coverage and a taught sentence
   pattern, yet scored 0/3 — the clearest evidence in this whole project that this
   tiny model pattern-matches narrowly rather than reasoning.

## One limitation and next experiment

**Limitation:** `opposites` is the sharpest negative result — 100% vocabulary
coverage, a taught general structure, taught word associations, and still 0/3
correct, both on the eval and in live chat (`"the opposite of hot is"` → `out .`).
Coverage and "the words exist in training data" are necessary but clearly not
sufficient for this model to produce a correct answer.

**Next experiment:** rather than teaching structure and association in *separate*
sentences (as I did here), directly combine them for a held-out subset — e.g. teach
"the opposite of wide is narrow", "the opposite of clean is dirty", etc. using the
*same* explicit "the opposite of X is Y" phrasing as the eval, for pairs the eval
does *not* test, while still keeping hot/cold, empty/full, noisy/quiet taught only
through indirect sentences as before. I'd predict this makes the *structure* even
more reliable but would not, by itself, fix `opposites`' 0/3 score, since the actual
gap seems to be combining a memorized structure with a separately-memorized
association — which would need many more varied combined examples, not just a
cleaner version of the same two data points I already provided.

## Reproduce and inspect

1. Clone this repo, then: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (includes `numpy`, required by `run_evals.py`'s model-hashing, in addition to the original `torch`/`pypdf`/`jupyter`).
2. **Starter run:** ensure `corpus/` contains no `extension_*.txt` files, then `python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace custom_llm.ipynb` (or Run All in Jupyter/Colab). Both runs use seed 42 and are fully deterministic — my own starter re-run reproduced identical `history.json` and `language_eval_comparison.json` byte-for-byte.
3. **Expanded run:** put the 5 `extension_*.txt` files (content reconstructable from `evidence/expanded/corpus.txt` / `corpus_manifest.json`, or ask me for the generator script) into `corpus/`, then Run All again — a fresh `llm_runs/` folder is created each time.
4. **Rerun evals standalone** on either saved model: `python run_evals.py --model evidence/starter/model.pt --output results/my-eval --stage final` (swap in `evidence/expanded/model.pt` for the other run; use a fresh `--output` directory each time).
5. **Chat:** `python chat.py --model evidence/expanded/model.pt --transcript results/my-chat.json`.
6. **Maintainer checks** (optional): `python -m unittest test_language_evals test_corpus`.
7. **Embedding viewer:** download `embedding-viewer.html` from this repo, open it locally, and load `evidence/starter/checkpoint.json` or `evidence/expanded/checkpoint.json`.

Supplementary (optional, predates the eval suite): [EXPERIMENTS.md](EXPERIMENTS.md)
covers extra step-count and model-size scaling experiments run under the original,
single-experiment version of this assignment — see
[archive_pre_eval_run/](archive_pre_eval_run/) for that original submission's full
evidence, kept for continuity rather than replaced outright.

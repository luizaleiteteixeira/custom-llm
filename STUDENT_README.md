# My Custom LLM Experiment

Replace the prompts below with your choices, actual outputs, and explanations.

## My choices and prediction

State your corpus and source, training steps, and learning rate. Give a brief reason
for each. What patterns did you expect before running the notebook?

Did you use the classroom corpus, expand it with files in corpus/, or choose
folder-only mode? Name your permitted sources and the unique passages they added.
How did you check extracted PDF text and resolve warnings? Link corpus_manifest.json
when sharing is permitted. Do not publish private source text or derived outputs.

## My run

Link your **executed notebook**, config, and training summary. Record the actual
completed steps, elapsed time, hardware, parameter count, vocabulary size, document
count, and split sizes. Clearly identify interruptions or failures.

Link vocabulary_report.json and report training/held-out unknown-token rates.
Did the 509-type vocabulary retain the words that mattered for your experiment?
The split is by short passage, not source file; explain what that evaluation can test.

## My evidence

Embed your training_curves.svg. Include all measured losses in a table and link
history.json. State the evaluation panel sizes. Show the untrained, halfway, and
final samples and link the full sample files, including empty or garbled outputs.

Link tokenization.json and inspection.json. Show one word, its ID, and its
64-number vector before and after training. Show the saved parameter's before value,
gradient, and after value, plus the probability comparison for the same prefix.

Include the three temperatures' samples and link temperature_comparison.json.

## What I learned

Use actual values from your run to explain:

1. What is my corpus, what can it teach, and what is missing? Why hold data out?
2. How do a token, token ID, vector, and embedding differ?
3. What makes this a neural network? How did loss, gradients, and the optimizer change its weights?
4. What does attention combine, and why can it not look at future tokens?
5. How do probabilities become generated text? What changed with temperature, and did any weights change then?
6. Did the samples and both loss curves support my prediction? What can I honestly conclude?

## One limitation and my next experiment

Describe one observed limitation. Propose one change to data or a setting, explain
why, and predict the effect. A second training run is optional.

## Reproduce and inspect

Explain how to open your notebook and locate the corpus and evidence. Verify the
repository signed out before submitting. Do not clear notebook outputs.

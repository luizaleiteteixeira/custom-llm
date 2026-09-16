# Class 4 Assignment: Building a Custom LLM

Choose data, train a tiny language model, and explain what its numbers and outputs show.

## Overview

Use the ready-made notebook to understand the basics of a language model: corpus, tokens, vectors, embeddings, neural networks, and learning. It uses Andrej Karpathy's actual nanoGPT model, using PyTorch and a small word-token transformer. Choose your corpus, training steps, and learning rate, then submit one public GitHub repository URL with your executed notebook and evidence. This miniature model generates short sentences from a narrow corpus, not general chat answers.

## What You Are Submitting

- Your executed custom_llm.ipynb, saved after the final run with settings, inspections, samples, and plots visible. Submit your own final-run version, not the unexecuted starter.
- Text samples from the untrained, halfway, and final model, using the same generation settings.
- A training/validation loss plot and the full table of measured values from the fixed evaluation panels.
- An inspection of one token's ID and embedding vector before and after training, next-token probabilities, and one real gradient and parameter update.
- A short explanation of the learning process, a temperature comparison, one limitation, and a proposed next experiment.

## Open the Notebook

- [Custom LLM sample project on GitHub](https://github.com/pepealonso95/custom-llm)
- Local Jupyter or VS Code: install requirements.txt and open custom_llm.ipynb with that Python 3 environment. Colab generally includes PyTorch; notebook setup installs the PDF reader if absent. The pinned nanoGPT model source is provided.
- [Open custom_llm.ipynb in Google Colab. The default CPU runtime is enough.](https://colab.research.google.com/github/pepealonso95/custom-llm/blob/main/custom_llm.ipynb)
- Read [Karpathy's nanoGPT explanation](https://github.com/karpathy/nanoGPT) alongside the notebook. The model uses two blocks, four heads, 64-number embeddings, a 48-token context, PyTorch backpropagation, and AdamW. Whole-word tokenization is a classroom choice, not intrinsic to nanoGPT. Classroom additions make data, evaluation, and learning visible.

## Expand Your Corpus with Files

- Put PDF, TXT or Markdown files in corpus/ beside the notebook, such as corpus/report.pdf, corpus/notes.txt or corpus/research/summary.md. Subfolders work. You do not need to paste the text into the code.
- In Colab, run sections 1 and 2 once to create /content/corpus. Refresh the left Files sidebar and upload your files into that folder. Opening a GitHub notebook does not copy its folders or your local files into Colab. Save your source files elsewhere too; runtime storage is temporary.
- Keep CORPUS = "classroom" to add your files to the teaching sentences, or set CORPUS = "folder" to use only your files. Folder-only mode requires at least 100 distinct extracted passages. CORPUS_FOLDER selects the folder; its default is "corpus".
- Select Run All from the top. Section 3 shows the imported files, text previews, passage counts and warnings. After training, load the new ZIP's checkpoint.json in the embedding viewer. Adding files does not instantly update the model: the network must train on their text.
- PDFs need extractable text; run OCR on scans first. Unreadable, encrypted or entirely textless files stop with an error naming the file. Partly textless PDFs produce page warnings. Inspect the extracted text for missing pages or confusing reading order. TXT and MD must use UTF-8; Markdown is plain text, and links/code are not fetched or executed.
- Long documents are split automatically into non-overlapping passages of at most 47 word/punctuation tokens, keeping sentence or line boundaries when possible. A passage is the short training example the notebook calls a document. The model keeps the 509 most frequent training token types; other types become UNK. Inspect both unknown-token rates before assuming a larger corpus adds useful information.
- Review corpus_manifest.json for file sources, previews, warnings and duplicate counts, and vocabulary_report.json for vocabulary coverage. Limits are 50 supported files, 25 MB each, 100 MB total, 200 pages per PDF and 2 million extracted characters per file. Hidden files, symbolic links, unsupported formats and corpus/README.md are ignored.
- Use only material you have permission to use and share. Added files in corpus/ are Git-ignored, but the results ZIP contains extracted text, filenames/hashes and model weights; the executed notebook also exposes examples. Review all artifacts before publishing. Git-ignore is not a privacy guarantee for derived results.

## Your Three Choices

- Corpus: use the supplied synthetic classroom sentences, expand them with files in corpus/, or choose folder-only mode. Explain where the data came from, how many unique passages it added, and what patterns it could teach.
- Training steps: a positive whole number of weight updates. Try 10 for setup, then 3,000 as a starting training budget. A step is not an entire pass through the corpus. Runtime and sample quality depend on your machine, data, and choices.
- Learning rate: the initial size of the optimizer's updates. Start with 0.001. The notebook uses warmup and cosine decay during training. Explain why excessively large or small updates could be a problem.
- Edit those three settings in section 1 and write your prediction before training. You can keep the defaults, provided you explain your choices.
- The 48-token context window keeps this model small enough to inspect. Longer text is split into passages, not silently truncated. Use data you have permission to share, without confidential or personal records.
- Build the vocabulary only from training text and report both training and held-out unknown-token rates. Other settings are optional experiments. Change one at a time and explain it. Complete and understand the baseline notebook first.

## Evaluate the Model Fairly

- Keep the same split, evaluation panels, seed, and baseline generation settings before and after training. Duplicate passages are removed before a 90/10 passage split; validation passages never supply weight updates. Passages from the same source file can appear in both sets, so this does not test generalization to unseen source files.
- The loss plot uses fixed panels of at most 20 training and 20 validation documents, averaging non-padding next-token targets. Report every measured value and both panel sizes. These are small estimates, not full-corpus measurements.
- Show all saved samples, including empty or garbled strings. Use the same starting token and sampling seed for the temperature comparison. Different corpora and vocabularies do not produce directly comparable loss scores.
- Falling training loss alone does not demonstrate generalization. Compare held-out loss and samples too. The synthetic corpus deliberately repeats contexts. Held-out sentences share templates with training, so plausible output does not demonstrate broad knowledge or generalization to new templates. Report lack of improvement honestly.

## Save Your Results

Each Run All creates a new llm_runs/ folder and a ZIP. In Colab, download the ZIP before ending the session. Also download the executed notebook separately after the run; the results ZIP does not contain the currently open notebook.

- The folder contains config.json, corpus.txt, corpus_manifest.json, vocabulary_report.json, split.json, tokenization.json, inspection.json, history.json, training.csv, training_summary.json, checkpoint.json, model.pt, training_curves.svg, the samples/ timeline, and temperature_comparison.json.
- If you interrupt training, continue through the inspection, plot, and download cells. Report the completed steps and interruption. Other errors require fixing the cause and rerunning from the top; a complete results ZIP is not guaranteed after an error.

## README Requirements

The README is the grading entry point. A reader should be able to follow your experiment, inspect the evidence, and understand your explanation without rerunning the notebook.

- A brief overview, the sources and permissions for your corpus files, and instructions to open and run your notebook. Explain how you checked PDF extraction and any warnings.
- Your three choices and reasons, plus the number of unique passages, vocabulary size, training/held-out unknown-token rates, and the train/validation split. Link corpus_manifest.json and vocabulary_report.json when sharing is permitted.
- What you expected before training, followed by what you actually observed in the same run.
- Actual completed steps, elapsed time, hardware, and the model's parameter count. Identify interrupted or failed runs clearly.
- Explain corpus, tokens, IDs, vectors, embeddings, neural-network weights, loss, and learning using actual notebook examples. Trace one word from text to its ID and 64-number vector, then explain one saved gradient and weight update.
- Explain how attention uses earlier context, how probabilities become generated word tokens, and how temperature changes sampling without updating weights. Finish with one observed limitation and one proposed next experiment.

## Suggested Workflow

- Open the sample notebook, save your own copy, and read the explanatory cells as you go.
- Choose your corpus, training steps, and learning rate. Write your reasons and prediction. A 10-step run checks setup; use a meaningful training budget for the final experiment.
- Select Run All. Let the data inspection, untrained evaluation, training, final inspection, and evidence-saving cells finish in order.
- Download embedding-viewer.html from the sample repository, open it locally, and load your checkpoint.json. Compare a word's initial/final 64D vector and neighbors, inspect the first update, and read the sample timeline alongside both loss curves. PCA compresses the map; cosine neighbors use the full vector space. Compare the three temperatures without retraining.
- Save the results ZIP and the executed notebook separately, write your explanation in the README, and publish your notebook and selected evidence on GitHub.

## Starter Prompt for Your AI Assistant

Help me work through custom_llm.ipynb for Class 4. Before training, ask me for my corpus, training steps, and learning rate; explain these choices briefly and wait for my answer. Help me write a prediction. Keep the evaluation settings fixed and run the notebook in order. At each inspection, help me understand the actual data, token IDs, embedding vectors, probabilities, loss, gradients, and weight changes. Ask me to explain them in my own words. Help me save the executed notebook and evidence and write an honest README from my outputs. Do not invent results or substitute a pretrained model.

## Evidence Required in the README

- Show the untrained, halfway, and final text samples, linking the full saved files. Explain at least one visible change or lack of change.
- Embed training_curves.svg and include the full loss table from history.json. State that these are fixed training and validation panels, each with at most 20 documents.
- Link tokenization.json and inspection.json. Include one word-to-ID-to-vector example, the vector before/after, the first parameter's value/gradient/update, and one next-token probability comparison. Explain what each means.
- Link your executed notebook, config.json, training.csv, training_summary.json, and temperature_comparison.json. Explain which data and settings stayed fixed, what changed in training, and what changed only at inference.

## Submission Checklist

- After the final run, save your notebook with all outputs. Upload or push that .ipynb file, your README, and selected results to your public GitHub repository. In Colab, use File → Download → Download .ipynb. Do not clear the outputs. Open the notebook on GitHub and confirm that the final inspections, losses, samples, and plot are visible.
- Open your repository signed out and verify that the notebook, plot, sample files, and evidence links are accessible.
- Keep the complete results ZIP locally. checkpoint.json stores initial/final embeddings for the viewer; model.pt stores the full network for inference. Neither is an exact training-resume file. Include the corpus or a reproducible source link when sharing is permitted.
- [Submit your GitHub repository](https://submissions-portal-eight.vercel.app)

## Learning Focus

The central purpose is understanding how data becomes predictions and how a neural network learns. Your explanation, choices, and visible evidence should support one another. A bigger network, longer training run, or more convincing sample does not replace an explanation. Report limitations rather than claiming the model understands language from a few plausible strings. Losses across different corpora are not a class ranking.

## Definition of Done

- The notebook completes with your chosen corpus and settings, and your README records the actual training budget.
- You compare the untrained, halfway, and final model using fixed evaluation settings and show all measured losses and saved samples.
- You can point to an actual token ID, embedding vector, next-token probability, gradient, and weight update, and explain how they connect.
- You can explain the role of the corpus, the neural network, held-out data, attention/context, and temperature, plus one limitation.

## Single Deliverable

- One public GitHub repository URL through the course submission portal. Its README contains your explanation and evidence, with your executed notebook available to inspect.

## Scope

- Use the supplied nanoGPT model. Writing the network yourself is optional. PyTorch runs the small model on CPU without an API key or pretrained weights. The vocabulary uses words and punctuation, not characters.
- No website, backend, deployment, GPU purchase, or large-model training is required. Keep the first experiment small enough to inspect.
- [Karpathy's nanoGPT source](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py) is the core reference. The larger [GPT video project](https://github.com/karpathy/ng-video-lecture) is an optional explanation of GPT training. The current lab already uses PyTorch and word tokens.
- A proposed next experiment is enough; a second training run is optional.

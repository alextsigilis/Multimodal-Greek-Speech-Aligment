# Revision plan: Parameter-efficient cross-modal alignment for ASR-free Greek spoken-content retrieval

The order matters: audit the pipeline first, since the padding and baseline issues may change every number. Then rerun, add baselines, test non-identical queries, decide the framing, and only then rewrite. Reviewer tags like (R1.4) show which comment each item answers. Items marked *Optional* strengthen the paper but aren't strictly required.

## Phase 0 — Setup

- [ ] **Check the revision deadline** in the decision letter. If this plan won't fit, ask the editor for an extension now rather than at the end.
- [ ] **Tag the current code and results** (e.g. `submission-v1`) so you can trace where every old number came from.
- [ ] **Record exact data provenance:**
  - HParl release and split files.
  - Common Voice version, and which split(s) the 28,211 clips come from.
  - The transcript normalisation you apply (case, tonos/diacritics, punctuation, numbers).

  Normalisation will matter for e5, BM25 and WER alike.

## Phase 1 — Audit the pipeline

- [ ] **Padding (R1.5).** Check whether the downsampling and final mean pooling include Whisper's 30 s padding.
  - The fix: run Whisper on the padded input as usual, then slice the encoder output to the real length, about duration / 20 ms frames, before downsampling.
  - Don't truncate the input itself; Whisper's positional embeddings expect 30 s.
- [ ] **Baseline audit (R1.4, R3).** Identify the ASR model; the "baseline-small" label suggests Whisper-small.
  - Record the decoding settings: forced Greek, beam size, temperature fallback.
  - Rerun the cascade separately on HParl test and Common Voice.
  - Find why its MRR is below the minimum implied by its Recall@k values.
- [ ] **One shared evaluation function** for every system: similarity matrix → ranks → Recall@k and MRR.
  - Define how ties are handled.
  - Add an assertion that MRR falls within the bounds implied by Recall@k.
  - Unit-test it on a toy matrix.
- [ ] **Figure 6 provenance (R2.1).** Find the run that produced it: dataset, downsampling setting, and what "fcnn" means.
  - If it's the (8,4) downsampling result, it becomes a proper ablation in Phase 5.
  - Otherwise, delete it.
- [ ] **Duplicate transcripts.** Count unique normalised transcripts in HParl test and Common Voice, and look at the most frequent ones.
  - Redefine relevance as "all recordings with the same normalised transcript as the query".
  - Use one query per unique transcript.
  - This also gives R1 the multiple-relevant-items setting (R1.1).
- [ ] **Leakage check.** Count test transcripts that also appear verbatim in HParl train; parliament has formulaic lines. You need this before keeping the "not memorisation" sentence in Section 5.2.
- [ ] **Checkpoint selection.** Confirm which checkpoint is evaluated (best validation loss or epoch 40), and that Common Voice played no role in choosing it.

## Phase 2 — Re-establish the core results

- [ ] **Cache frozen-encoder outputs** to disk: masked Whisper frames, plus e5 embeddings for every transcript and query set. Every aligner experiment then trains only the aligner, which makes seeds and ablations cheap on the RTX 3060.
- [ ] **Retrain MLP, CNN and BiLSTM** with at least 3 seeds (5 if time allows), and report mean ± sd.
- [ ] **Paired statistics** on per-query results:
  - Paired bootstrap confidence intervals for MRR differences.
  - McNemar's test for Recall@1.

  Section 6.4 already promises this.
- [ ] **Pool-size control (R1.3).**
  - Draw 10 random Common Voice subsets of 8,679 candidates and report mean ± sd.
  - Plot Recall@1 against candidate-pool size for both datasets.

## Phase 3 — Baselines

- [ ] **ASR quality (R1.2, R2.5).** Report WER and CER on HParl test and Common Voice, using the same normalisation as retrieval.
- [ ] **Cascade across Whisper sizes (R3):** tiny, base, small, medium, and large-v3 if feasible; inference fits on 12 GB in fp16.
  - Plot WER against retrieval performance.
  - This is the direct evidence for your error-propagation motivation.
- [ ] **ASR + BM25 (R2.5)** with Greek-aware normalisation (lowercase, strip tonos).
- [ ] **Oracle baselines (R1.2, R2.5, R3):** e5 and BM25 on the reference transcripts.
  - Under exact queries these will be near 100%. Report them anyway as the sanity anchor.
  - They become meaningful in Phase 4.
- [ ] **Linear-projection aligner (R1.6).** Because mean pooling is linear, this is essentially a linear map on the mean-pooled Whisper vector, so it's very cheap to train.
- [ ] **Vanilla variants (R3):** single-layer MLP, single-kernel CNN, single-layer BiLSTM.
  - For the CNN, also vary kernel width to test temporal context directly, as R3 suggests.
- [ ] **SAMU-XLSR / SONAR (R1.6).** Check whether released checkpoints cover Greek speech.
  - If they do, evaluate zero-shot, using their own text encoder for queries.
  - If not, discuss them in related work and say why they couldn't be evaluated.
- [ ] *Optional (R3):* a cascade with ASR fine-tuned on HParl. Check whether Paraskevopoulos et al. released adapted models before training your own.

## Phase 4 — Non-identical queries

All three reviewers ask for this.

- [ ] **Partial-overlap queries.** Take contiguous spans covering 50% and 30% of each transcript's words, with a fixed seed and a minimum length.
- [ ] **Keyword queries.** Use the top 2–3 content words by TF-IDF, computed on train transcripts. This directly tests R3's keyword-spotting hypothesis.
- [ ] **Paraphrase queries.**
  - Generate Greek paraphrases of test transcripts with an LLM, keeping the original audio as the target.
  - Filter out paraphrases with high word overlap.
  - Have a native speaker judge a random sample of about 200 for meaning preservation and fluency, and report agreement if there are two judges.
  - Disclose the LLM in the methods section and the AI declaration.
- [ ] **Run every system and oracle on every query type**: HParl at minimum, Common Voice if budget allows.
- [ ] *Optional, strongest answer to R1.1:* a small human-written topical query set over HParl test.
  - For example, 50 queries, pooling the top 20 results from all systems.
  - Judge relevance with graded labels.
- [ ] **Decision point.** Compare how much the aligners degrade with how much the oracle degrades, and fix the paper's framing *before* rewriting:
  - If the aligners track the oracle, you can make a careful semantic claim.
  - If they collapse while the oracle holds up, frame the contribution as content matching.

## Phase 5 — Domain shift, ablations and efficiency

- [ ] **Partial encoder adaptation (R1.3).** Unfreeze the top 1–2 Whisper-tiny layers, or add LoRA. Train on HParl, evaluate on both datasets, and report the trainable-parameter count.
- [ ] *Optional:* mixed-domain training using the Common Voice train split, with no sentence or speaker overlap with your evaluation set. This shows whether the gap is about data or about the architecture.
- [ ] **Temperature (R3).**
  - Report τ's initial value and the final learned value.
  - Add a log-parametrisation with clamping.
  - Compare against a fixed τ.
- [ ] **γ ablation (R3):** at least 0.1, 0.5 and 0.9.
- [ ] **Downsampling ablation:** (8,4) vs (32,16) for all aligners.
- [ ] **Efficiency (R1.7)**, measured on the same hardware with batch sizes stated:
  - Trainable parameters.
  - Training wall-clock time.
  - Peak GPU memory (`torch.cuda.max_memory_allocated`).
  - Indexing real-time factor per hour of audio, aligner vs each ASR cascade.
  - Query latency.

## Phase 6 — Analysis

- [ ] **Error analysis (R3 motivation).** Look at queries where the cascade fails and the aligner succeeds, and the reverse. Categorise them: names, numbers, terminology, very short utterances, duplicates.
- [ ] **Breakdowns by ASR WER bucket and by duration bucket.** Do the aligners help most where ASR is worst? This answers R3's "why ASR-free" question with evidence.
- [ ] **Greek-specific analysis (R3).** Greek has heavy vowel homophony: ι/η/υ/ει/οι, ε/αι, ο/ω.
  - Measure how many ASR errors are spelling-only, meaning the same pronunciation with different orthography.
  - Check whether those errors hurt cascade retrieval.
  - This gives you a genuinely language-specific angle.
- [ ] **Qualitative examples:** a few Greek queries with English glosses.

## Phase 7 — Rewrite

Write the abstract and conclusion last, once the numbers are final.

- [ ] **Framing.** Apply the Phase 4 decision:
  - Remove unsupported "semantic" claims from the abstract, introduction, Section 2.2, the Section 2.4 heading and Section 4.3.
  - Add a scope paragraph that defines known-item retrieval.
- [ ] **Section 2.4 (R2.2).** Add a paragraph contrasting your task with prior semantic retrieval setups.
- [ ] **"ASR-free" (R2.3).** Define it as no transcription at indexing or query time, and acknowledge Whisper's ASR-supervised pretraining.
- [ ] **Related work (R1.6):** SAMU-XLSR, SONAR, linear-projection alignment work, and lexical spoken-retrieval baselines.
- [ ] **Method section:**
  - Define every symbol: N, Lᵢ, dₛ, L′ᵢ, dₜ, B (R3).
  - Describe padding handling (R1.5).
  - Give τ's initial value and clamping, and the γ rationale (R3).
  - Justify the architectures with the vanilla variants and kernel-width results (R3).
- [ ] **Setup section:**
  - ASR model and decoding configuration.
  - Normalisation and data versions.
  - Duplicate-aware relevance and the pool-size control.
  - The new query sets.
- [ ] **Results.** Restructure in this order:
  1. Known-item retrieval.
  2. Partial and paraphrased queries.
  3. Cross-domain transfer.
  4. Ablations.
  5. Efficiency.

  Report all numbers with confidence intervals or sd.
- [ ] **Discussion:**
  - Ground the Section 6.1 CNN interpretation in the kernel-width ablation.
  - Update Section 6.3 with the pool-size control and partial-adaptation results.
  - Rewrite the limitations.
- [ ] **Section 8:** delete it, or fold its content into the downsampling ablation.
- [ ] **Leftover text (R1.8, R2 minor 1–2):**
  - Remove the instruction in the Figure 1 caption.
  - Rewrite the AI declaration in past tense with plural authors, and add the paraphrase generation if an LLM was used.
- [ ] **Author list.** Check the fourth author's surname; "Crisis" looks like an autocorrect error.
- [ ] **Code and data availability (R3).** Release code, configs and the new query sets, and add an availability statement.

## Phase 8 — Figures and tables (R2 minor 3)

- [ ] **Figure 1:** verify every element against the actual code, and fix the arrow alignment on the right.
- [ ] **Figures 2 and 4:** overlay the histograms with shared bins and semi-transparent colours. Label the axes "Duration (s)" and "Proportion". Consider merging them into one figure with HParl train, HParl test and Common Voice.
- [ ] **Figures 3 and 5:**
  - Integer x-ticks at the evaluated k values only.
  - y-axis labelled "Recall@k (%)".
  - Confidence-interval bands.
- [ ] **All figures:**
  - Export as vector PDF.
  - Use consistent fonts.
  - Use the same model names everywhere, with no legacy labels.
- [ ] **Tables:** consistent decimals, ± sd, and bold for the best value.

## Phase 9 — Response letter and final checks

- [ ] **Point-by-point response letter.** Quote each comment, give your response, and point to the page and line of each change. Mark the changes in the manuscript, e.g. with latexdiff.
- [ ] **For anything you didn't do** (a fine-tuned ASR cascade, say), explain why and what you did instead.
- [ ] **Consistency pass:**
  - Every number in the abstract, text and conclusion matches the tables.
  - Rerun the MRR bound check on the final tables.
  - Search for "semantic", "phonetic" and "ASR-free" to confirm consistent usage.
- [ ] **Journal requirements.** Check the guide for authors: highlights, data availability statement, CRediT statement.
- [ ] **Sign-off** from all co-authors before resubmitting.

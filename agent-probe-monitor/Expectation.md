# Project Expectations and Related Work

Companion document for `agent-probe-monitor`. Drop this into the repo root.

## 1. What this project claims

Working title: "Do Agents Know They Are Being Hijacked? White-Box Runtime
Monitoring of LLM Agents with Activation Probes."

Central hypothesis: during a multi-step agent trajectory, the model's
residual-stream activations linearly encode (a) that the current observation
contains injected instructions and (b) that the model's next action furthers
the injected goal, and this encoding generalizes to attack types and task
suites never seen at probe-training time.

Intended contributions, in the order a reviewer will weigh them:

1. A step-level activation-probing framework for agent trajectories, with
   two targets (injection_present, attack_furthering) rather than the usual
   single trajectory-level label.
2. Generalization results: leave-one-attack-out and leave-one-suite-out
   AUROC and TPR at 1% and 5% FPR, across at least three model families
   (e.g. Llama-3.1-8B/70B, Qwen-2.5, Mistral).
3. Latency-to-detection: fraction of successful attacks caught before the
   attack-furthering action executes. This metric is the novelty carrier;
   static probe papers cannot report it.
4. Head-to-head comparison against black-box monitors (LlamaGuard-class
   output classifiers, perplexity, action log-prob) under identical splits.
5. At least one causal result, not just correlational: e.g. steering along
   the probe direction changes compliance with the injection, or ablating
   it degrades the model's own refusal behavior. Reviewers in 2026 discount
   pure-AUROC probe papers.

## 2. Success criteria and kill criteria

Success looks like: injection_present LOAO AUROC clearly above 0.9 with the
probe beating all black-box baselines at 1% FPR; attack_furthering LOAO
AUROC above roughly 0.8; a majority of successful attacks caught at or
before the execution step at 5% FPR; results replicated on 3 model families.

Kill / pivot criteria, checked early (by end of September 2026):

- If injection_present is near-ceiling for every method including
  perplexity, the task is too easy and the paper must lean entirely on
  attack_furthering and latency; consider adding adaptive attacks that
  paraphrase injections to defeat string-level detectors.
- If attack_furthering probes do not beat action log-prob, the white-box
  story collapses; pivot to analyzing why (e.g. the signal lives in
  attention patterns, not the residual stream) which is itself a paper if
  done carefully.
- If probes transfer across attacks but not across suites, reframe as a
  study of what the probe actually reads (topic features vs. an abstract
  "off-task" feature), connecting to the task-drift line below.

## 3. Known threats to validity (address these in the paper, reviewers will)

- Label leakage: injection strings are inserted verbatim, so a probe may
  learn surface lexical features. Mitigations: paraphrased-injection test
  set, cross-attack splits, and comparing against a bag-of-words probe on
  token embeddings as a "trivial feature" baseline.
- Segment-boundary artifacts: pooled spans must be computed from exact
  token offsets, not re-tokenization (see README caveat).
- Single-benchmark overfitting: AgentDojo is the primary testbed; include
  at least one out-of-benchmark transfer evaluation (InjecAgent or Agent
  Security Bench style tasks) for the camera-ready.
- The closest prior work is Abdelnabi et al.'s task-drift detection (below);
  the paper must state precisely what is new relative to it: action-level
  (not just drift-level) prediction, pre-execution latency, LOAO/LOSO
  generalization, and causal interventions.

## 4. Timeline expectation (August 2026 start, ICML 2027 target)

- Aug: reconcile AgentDojo API, exact-offset segment tracking, first
  collection run (one suite, one attack, Llama-3.1-8B), layer sweep.
- Sep: full collection grid on 8B; LOAO/LOSO results; go/no-go check.
- Oct: second and third model families; baselines; paraphrase test set.
- Nov: causal experiments (steering/ablation); 70B run on Bender.
- Dec: writing, ablations reviewers ask for (pooling, probe family, layer).
- Jan: internal review (Flek lab, prospective Tübingen supervisors), submit.

## 5. Related work

I did not retrieve these documents while generating the codebase; they are
the literature the design is based on. Verify each reference before citing.
Group A is what you must know cold; a reviewer will assume you do.

### A. Directly load-bearing

- Debenedetti, Zhang, Balunovic, Wolf, Tramer. "AgentDojo: A Dynamic
  Environment to Evaluate Prompt Injection Attacks and Defenses for LLM
  Agents." NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.13352. The
  testbed; read the task/injection-task/attack abstractions before touching
  `agentdojo_env.py`.
- Abdelnabi, Fay, Cherubin, Salem, Fritz, Paverd. "Get my drift? Catching
  LLM Task Drift with Activation Deltas." (also circulated as "Are you
  still on track!?"). SaTML 2025. arXiv:2406.00799. The closest prior
  work: detects task drift from activation differences before/after
  processing external data. Your differentiation must be explicit (see
  Section 3). Also strategically relevant given Abdelnabi is a prospective
  supervisor.
- Zhan, Liang, Ying, Kang. "InjecAgent: Benchmarking Indirect Prompt
  Injections in Tool-Integrated LLM Agents." Findings of ACL 2024.
  arXiv:2403.02691. Out-of-benchmark transfer target.
- Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz. "Not what you've
  signed up for: Compromising Real-World LLM-Integrated Applications with
  Indirect Prompt Injection." AISec 2023. arXiv:2302.12173. The paper that
  defined indirect prompt injection; frames the threat model.
- Andriushchenko et al. "AgentHarm: A Benchmark for Measuring Harmfulness
  of LLM Agents." ICLR 2025. arXiv:2410.09024. Adjacent threat model
  (harmful user tasks rather than injections); candidate transfer eval and
  the other prospective supervisor's core work.

### B. Probing and internal-state detection

- Azaria, Mitchell. "The Internal State of an LLM Knows When It's Lying."
  Findings of EMNLP 2023. arXiv:2304.13734. Origin of the modern
  "classifier on hidden states" line.
- Marks, Tegmark. "The Geometry of Truth: Emergent Linear Structure in LLM
  Representations of True/False Datasets." COLM 2024. arXiv:2310.06824.
  Source of the mass-mean probe implemented in `probes.py`.
- Orgad, Toker, Gekhman, Reichart et al. "LLMs Know More Than They Show:
  On the Intrinsic Representation of LLM Hallucinations." ICLR 2025.
  arXiv:2410.02707. Evidence that error information concentrates in
  specific tokens; motivates careful pooling choices.
- MacDiarmid et al. (Anthropic). "Simple probes can catch sleeper agents."
  Anthropic Alignment blog, 2024. Defection probes; the safety-monitoring
  framing this project inherits.
- Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn (Apollo Research).
  "Detecting Strategic Deception Using Linear Probes." 2025.
  arXiv:2502.03407. Methodologically the closest probe-evaluation
  standard: on/off-policy evaluation, TPR at low FPR. Copy their rigor.
- Zou et al. "Representation Engineering: A Top-Down Approach to AI
  Transparency." 2023. arXiv:2310.01405. Background for reading/steering
  directions; use for the causal experiments.

### C. Defenses and monitors to compare against or discuss

- Inan et al. "Llama Guard: LLM-based Input-Output Safeguard for
  Human-AI Conversations." 2023. arXiv:2312.06674. The output-monitor
  baseline family in `baselines.py`.
- Zou et al. "Improving Alignment and Robustness with Circuit Breakers."
  NeurIPS 2024. arXiv:2406.04313. Representation-level defense; discuss as
  the intervention counterpart to monitoring.
- Chen et al. "StruQ: Defending Against Prompt Injection with Structured
  Queries." USENIX Security 2025. arXiv:2402.06363. And its successor
  SecAlign (arXiv:2410.05451). Training-time defenses; your monitor is
  complementary and should be positioned that way.
- Hung et al. "Attention Tracker: Detecting Prompt Injection Attacks in
  LLMs." 2024. arXiv:2411.00348. Attention-based white-box detection on
  single-turn settings; a natural additional baseline and the fallback
  direction if residual-stream signal is weak.
- Wallace et al. "The Instruction Hierarchy: Training LLMs to Prioritize
  Privileged Instructions." 2024. arXiv:2404.13208. Frames
  instruction-priority as a learned feature, which is plausibly what your
  probe reads.

### D. Context for the safety framing

- Perez, Ribeiro. "Ignore Previous Prompt: Attack Techniques for Language
  Models." 2022. arXiv:2211.09527.
- Greenblatt, Shlegeris et al. "AI Control: Improving Safety Despite
  Intentional Subversion." ICML 2024. arXiv:2312.06942. Runtime monitoring
  as part of a control protocol; useful for the introduction's framing.

### Reading order suggestion

Week 1: AgentDojo, Abdelnabi et al. task drift, Greshake et al.
Week 2: Goldowsky-Dill et al., Marks & Tegmark, Orgad et al., sleeper-agent
probes. Week 3: defenses (StruQ/SecAlign, circuit breakers, Attention
Tracker, instruction hierarchy) and AgentHarm. After week 3 you should be
able to write the related-work section before running a single experiment,
which is the correct order for a gap-driven paper.

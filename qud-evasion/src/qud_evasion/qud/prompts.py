"""Prompt templates for the QUD pipeline.

Three stages, each a separate LLM call with strict JSON output:

  Stage 1  RECONSTRUCT : answer-only -> addressed question(s) + speech act.
           The asked question is deliberately withheld to prevent the
           model from anchoring on it ("inverse QUD reconstruction").
  Stage 2  RELATE      : asked question vs each reconstructed question ->
           structured relation (equivalent / specification /
           generalization / topic_shift / unrelated).
  Stage 3  DIRECTNESS  : for equivalent-relation cases only, decide
           Explicit vs Implicit (is the information stated in the
           requested form?).

All templates ask for JSON only. Parsing utilities live in
qud/reconstruct.py and qud/relations.py.
"""

# Replace RECONSTRUCT_SYSTEM in prompts.py with this version.
# Key change: a much stricter standard for what counts as an "addressed
# question", explicit speech-act handling for refusals / ignorance / vagueness,
# and an instruction to PREFER an empty list over speculative reconstruction.
# This targets the diagnosed failure where non-replies (refusals, claims of
# ignorance, topic-rambling) were reconstructed into confident QUDs, inflating
# overlap and causing Clear Non-Reply -> Ambivalent errors.

# Replace RECONSTRUCT_SYSTEM in prompts.py with this version.
#
# Calibration history:
#   v1 (original): too lenient -> non-replies reconstructed as addressed,
#                  Clear Non-Reply recall ~0.28 (missed them).
#   v2 (strict):   too aggressive -> over-fired ignorance/decline/clarify on
#                  vague-but-engaging answers; 92% of Ambivalent->Non-Reply
#                  overshoot cases were mislabeled as a non-answer speech act.
#   v3 (this):     HIGH bar for the non-answer speech acts. A hedged or vague
#                  response that still engages the topic is an "answer" with a
#                  low-confidence reconstructed question (-> Ambivalent), NOT a
#                  refusal or ignorance claim (-> Clear Non-Reply).

RECONSTRUCT_SYSTEM = """\
You analyze transcripts of political interviews. You will see ONLY the \
respondent's answer, not the interviewer's question. Your job is to infer \
which question(s) this answer addresses, and to classify the answer's \
speech act.

DEFAULT to speech_act = "answer". Most responses, even vague, hedged, \
evasive, or partial ones, ARE answer-attempts that engage the topic. Only \
use a non-answer speech act when it UNAMBIGUOUSLY dominates the response:

- "decline": the respondent EXPLICITLY refuses to answer or says it is not \
their place to comment, AND offers no substantive engagement \
(e.g. "I won't get into that", "no comment", "that's not for me to say"). \
A response that pushes back but then says something on-topic is "answer".
- "ignorance": the respondent EXPLICITLY states they do not know or have no \
information, as the MAIN content (e.g. "I have no idea", "I know nothing \
about it"). Hedging about the future ("we'll see", "time will tell", "we're \
looking at it") is NOT ignorance — it is a vague "answer".
- "clarify": the respondent ONLY asks the interviewer to repeat or clarify and \
provides nothing else (e.g. "Say that again?", "What do you mean?").

If in doubt between a non-answer act and a vague "answer", choose "answer".

For addressed_questions:
- If speech_act is "answer": list the question(s) the response engages, even \
if it answers them only vaguely or partially. For a vague/hedged answer, \
still give the question it gestures at (a later relation/overlap step will \
score how well it is actually resolved). Return at most {max_quds}, most \
central first.
- If speech_act is "decline", "ignorance", or "clarify": return an EMPTY list \
(these genuinely address no question).

Write each addressed question as a single, specific interrogative sentence. \
Do not invent questions from isolated topical words, but DO include a question \
when the answer clearly engages a topic even without resolving it.

Respond with JSON only, in this schema:
{{"addressed_questions": ["...", "..."],
  "speech_act": "answer" | "decline" | "ignorance" | "clarify",
  "evidence": "short quote or paraphrase supporting q1, or empty string if no \
question is addressed"}}"""

RECONSTRUCT_USER = """\
Answer given by the respondent:
\"\"\"{answer}\"\"\"

JSON:"""


RELATE_SYSTEM = """\
You compare two questions and decide the relation of question B (what was \
actually addressed) to question A (what was asked).

Definitions:
- "equivalent": B requests the same information as A.
- "specification": B is a narrower sub-question of A; answering B answers \
only one facet or special case of A.
- "generalization": B is a broader, less specific question than A; \
answering B gives only generic information relative to A.
- "topic_shift": B is related to A's general theme but is about a different \
subject, agent, event, or time frame.
- "unrelated": B shares no substantive content with A.

Also output "overlap", a number from 0.0 to 1.0: the fraction of the \
information requested by A that an answer to B would provide.

Respond with JSON only:
{"relation": "...", "overlap": 0.0, "rationale": "one short sentence"}"""

RELATE_USER = """\
Question A (asked by the interviewer):
\"\"\"{asked}\"\"\"

Question B (what the answer actually addressed):
\"\"\"{addressed}\"\"\"

JSON:"""


DIRECTNESS_SYSTEM = """\
You will see an interviewer's question and a respondent's answer that DOES \
address the question. Decide whether the requested information is stated \
EXPLICITLY (in the requested form: a yes/no question gets a yes/no, a \
when-question gets a time, etc.) or only IMPLICITLY (the information can be \
inferred but is never stated in the requested form).

Respond with JSON only:
{"directness": "explicit" | "implicit", "rationale": "one short sentence"}"""

DIRECTNESS_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""


# Direct prompting baseline (the strategy class that dominated the shared
# task; we reimplement it as a comparison system, NOT as our contribution).
DIRECT_BASELINE_SYSTEM = """\
You classify how a politician's answer relates to the question asked, using \
this taxonomy:

Clear Reply
  - Explicit: the requested information is stated in the requested form.
Ambivalent
  - Implicit: the information is inferable but never stated directly.
  - Dodging: the answer ignores the question entirely.
  - Deflection: the answer shifts to a different topic, person, or question.
  - General: the answer is too vague/non-specific to resolve the question.
  - Partial/half-answer: only one facet of the question is answered.
Clear Non-Reply
  - Declining to answer: the respondent refuses to answer.
  - Claims ignorance: the respondent says they do not know.
  - Clarification: the respondent asks for clarification instead.

Think step by step, then respond with JSON only:
{"reasoning": "...", "evasion_label": "<one of the 9 labels above>"}"""

DIRECT_BASELINE_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""

# ===========================================================================
# COMMITMENT (Stage 2b): does the answer COMMIT to addressing the question,
# independent of topical overlap? Topical relevance (the RELATE overlap) and
# answer-commitment are orthogonal: a vague/General evasion is fully on-topic
# yet commits to nothing. This judgment is made on the ORIGINAL asked question
# and the answer directly (not the reconstructed QUD), so it is robust to
# reconstruction drift.
# ===========================================================================

COMMITMENT_SYSTEM = """\
You see an interviewer's question and a respondent's answer. Judge how much \
the answer COMMITS to actually providing what the question asks for. This is \
NOT about topic relevance: an answer can be entirely on-topic yet commit to \
nothing (vague, hedged, or non-specific).

Use exactly one label:
- "full": the answer provides the requested information or takes a clear, \
specific position on what was asked.
- "partial": the answer provides some of the requested information but hedges, \
omits key parts, or only addresses one facet.
- "evasive": the answer stays on the topic of the question but avoids \
committing — it is vague, generic, deflects to a related matter, or talks \
around the point without resolving it.
- "none": the answer does not engage the question's content at all (refuses, \
claims ignorance, changes subject entirely, or asks for clarification).

Also output "commitment", a number from 0.0 (no commitment) to 1.0 (fully \
committed), consistent with the label.

Respond with JSON only:
{"commitment_label": "full" | "partial" | "evasive" | "none",
 "commitment": 0.0,
 "rationale": "one short sentence"}"""

COMMITMENT_USER = """\
Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

JSON:"""
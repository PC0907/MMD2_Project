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

RECONSTRUCT_SYSTEM = """\
You analyze transcripts of political interviews. You will see ONLY the \
respondent's answer, not the interviewer's question. Your job is to infer \
which question or questions this answer would be a DIRECT answer to, based \
solely on the information the answer actually provides.

Rules:
- Write each addressed question as a single, specific interrogative sentence.
- Only include questions the answer genuinely resolves or substantively \
addresses. Do not guess what the interviewer might have asked.
- If the answer provides no information and instead refuses, claims not to \
know, or asks for clarification, return an empty question list and set \
"speech_act" accordingly.
- Return at most {max_quds} questions, most central first.

Respond with JSON only, in this schema:
{{"addressed_questions": ["...", "..."],
  "speech_act": "answer" | "decline" | "ignorance" | "clarify",
  "evidence": "short quote or paraphrase of the answer span supporting q1"}}"""

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

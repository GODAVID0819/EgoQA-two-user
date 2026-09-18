# Legacy-zero-shot judge prompts and output contracts

> Review snapshot generated from the active prompt builders. All example and placeholder values are English. This document does not contain real videos, private annotations, or deleted samples.

## Scope

The direct six-user path makes four judgments: one qa_formality call, one evidence_groundedness call, one speaker-only answerability call, and one all-six answerability call. Every model decision starts with the same lowercase pass/fail verdict field.

## New final-judge output contract

Both qa_formality and evidence_groundedness use this exact ordered shape:

```json
{
  "verdict": "pass/fail",
  "reason": "specific instance-level failure reason when verdict is fail; null when verdict is pass",
  "fix": "specific repair for that failure when verdict is fail; null when verdict is pass"
}
```

Valid pass example:

```json
{
  "verdict": "pass",
  "reason": null,
  "fix": null
}
```

Valid fail example:

```json
{
  "verdict": "fail",
  "reason": "The correct answer claims a placement that is never visible in the supplied videos.",
  "fix": "Replace the answer with a visibly supported placement or provide evidence that shows the claimed placement."
}
```

The model no longer emits review_passed, checks, nested status, semantic_subchecks, blocking_failures, why_generator_asked_this, feedback_to_generator, or 1-3 quality fields. The merger computes status, blocking failures, overall review_passed, and retry feedback from the authoritative verdict, reason, and fix.

### Code-computed merged record (not a model output contract)

```json
{
  "review_passed": "boolean computed from deterministic checks, both judge verdicts, and the answerability gate",
  "checks": {
    "qa_formality": {
      "status": "PASS/FAIL derived from verdict",
      "reason": "copied from the model failure reason, or a code-generated pass summary",
      "fix": "copied from the model failure fix, or an empty string"
    },
    "evidence_groundedness": {
      "status": "PASS/FAIL derived from verdict",
      "reason": "copied from the model failure reason, or a code-generated pass summary",
      "fix": "copied from the model failure fix, or an empty string"
    },
    "answerability": {
      "status": "PASS/FAIL derived from the asker-only/all-six gate",
      "reason": "code-generated gate reason",
      "fix": "code-generated repair instruction, or an empty string"
    }
  },
  "blocking_failures": "array computed from failed checks",
  "feedback_to_generator": "string assembled by code from failed reasons and fixes"
}
```

## Direct answerability contract

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "verdict",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": [
        "pass",
        "fail"
      ]
    },
    "reason": {
      "type": "string",
      "minLength": 1
    },
    "available_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    },
    "missing_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    }
  }
}
```

For each answerability condition, verdict is pass exactly when the supplied videos are sufficient. The code gate passes only for speaker-only=fail and all-six=pass. available_evidence and missing_evidence remain in the model output for structured audit.

## Judge-model training contract

```json
{
  "contract_version": "verdict_token_bce_sampled_frames_v3",
  "supervision_type": "next_token_pass_fail",
  "binary_label_names": {
    "0": "fail",
    "1": "pass"
  },
  "assistant_prefix": "{\"verdict\":\"",
  "head_type": "none; use the language-model vocabulary logits",
  "binary_logit": "logit(pass) - logit(fail)",
  "loss_function": "BCEWithLogits",
  "class_weighting": "balanced_inverse_frequency_from_training_split_only",
  "class_weight_reduction": "mean-one normalization independently per judge",
  "task_loss_weights": {
    "qa_formality": 0.2,
    "evidence_groundedness": 0.4,
    "answerability": 0.4
  },
  "trainable_parameters": "language attention+MLP LoRA only: q/k/v/o/gate/up/down projections, rank 8, alpha 16",
  "inference_generation": "constrain only the first generated token to the selected pass/fail token, then continue the same generation through the complete JSON contract"
}
```

Training supplies the fixed assistant prefix as input and applies BCE only to the next-token margin logit(pass)-logit(fail). It adds no classifier head and does not supervise archived numerical scores or later JSON fields. At inference, the first generated token is locked from those two logits and the same generation continues through the full JSON contract. Class weights are estimated from the training split independently per judge, then normalized to preserve the 0.2/0.4/0.4 task-loss scale.

## Prompt 1 — qa_formality

```text
You are the qa_formality judge for a six-user multiple-choice question. You are a text-only judge and do not see the videos.

Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.


Judge only the deterministic schema result and the displayed question and options. Do not infer visual truth or use hidden generator intent to rescue unclear wording.

PASS only if all requirements hold:

1. Structure: The deterministic schema branch is PASS. The item has exactly five non-empty A-E options, one valid correct letter, and an answer that exactly matches the selected option. The option strings themselves do not need A./B./C./D./E. prefixes.

2. Perspective: The question is a natural first-person or shared-memory question using I, me, my, we, us, or our. The options do not need first-person pronouns. FAIL third-person questions, questions without an asker perspective, and second-person questions such as "what were you doing?"

3. Information need: The speaker has a plausible reason to ask based on the described experience. FAIL a contrived third-party quiz with no natural information need.

4. Clarity: The question is conversational, concrete, grammatical, and locally unambiguous. References have enough local context to identify their intended referent; global uniqueness across the recording is unnecessary.

5. Options: All five options answer the same question and are mutually exclusive and reasonably parallel. FAIL incompatible option types.

6. Question target: FAIL when the answer is merely one person's activity concurrent with another event, regardless of direction, natural wording, concrete anchoring, or verified overlap. Also FAIL options that encode pairs of concurrent activities. Allow concrete objects, identities, states, locations, placements, outcomes, consequences, explanations, interaction results, and follow-ups.

7. Leakage: FAIL if the question or options directly name a participant; natural descriptive references are allowed. FAIL dataset-facing terms such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, evidence provider, embedding, similarity, or novelty. FAIL clock times, timestamps, timecodes, frame numbers, seconds-from-start, and minute marks. Natural relative wording such as before, after, while, later, last, and most recent is allowed.

Questions about object trajectories, before/after states, revisits and interventions, last-seen events, or cross-user ordering are allowed when naturally and unambiguously phrased. Do not verify their visual truth here, and do not confuse temporal ordering with a prohibited concurrent-activity question.

Set verdict to fail if any requirement fails; otherwise set verdict to pass.

Binary decision contract:
- Return only the fields in the requested JSON schema.
- Do not include reasoning, markdown, or code fences outside the JSON object.
- The first JSON field must be verdict, with exactly one lowercase value: pass or fail.
- Decide verdict before writing reason or fix.
- If verdict is fail, reason must identify the concrete instance-level failure and fix must give a repair targeted to that failure.
- If verdict is pass, reason and fix must both be JSON null.
- Never use a generic placeholder sentence as reason or fix.
- Keep a failure reason and fix to one sentence each and no more than 40 words each.
- Do not assign a numerical score, quality label, rank, quota, or comparison against other candidates.


Deterministic schema/formality branch:
{
  "status": "PASS",
  "errors": []
}

Known participant names for leakage detection only:
{
  "participant_names": [
    "SpeakerUser",
    "ProviderOne",
    "ProviderTwo",
    "ProviderThree",
    "ProviderFour",
    "ProviderFive"
  ]
}

User-facing question-answer item:
{
  "question_type": "neutral",
  "question": "Where was the mug placed after I handed it over?",
  "options": [
    "On the wooden desk",
    "Beside the kitchen sink",
    "Near the front door",
    "On the living-room sofa",
    "Inside a dark backpack"
  ],
  "correct": "B",
  "answer": "Beside the kitchen sink"
}

Return exactly one valid JSON object with this exact shape:
{
  "verdict": "pass/fail",
  "reason": "specific instance-level failure reason when verdict is fail; null when verdict is pass",
  "fix": "specific repair for that failure when verdict is fail; null when verdict is pass"
}
```

## Prompt 2 — evidence_groundedness

```text
You are the evidence_groundedness judge for a six-user multiple-choice question generated from egocentric videos.

Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.


Judge only whether the question and declared answer are visually and temporally grounded. Do not judge wording, first-person style, name leakage, timestamp citations in the question, schema form, or single-user answerability.

PASS only if all requirements hold:

1. Every material object, action, person, identity, state, location, continuity, and temporal claim in the question and declared answer is directly supported by the supplied media or metadata.

2. required_users[0] is the speaker and required_users[1] through required_users[5] are providers. The speaker media must establish the experience or reference that makes the question coherent, and at least one provider or compatible provider combination must establish the answer-bearing external detail. Unused providers are allowed.

3. Same-person and same-object links require visible continuity or distinguishing evidence. Do not infer identity from roles, timing, option wording, lookalikes, similar clothing, or similar objects.

4. State changes require the same object or place and both visible states. A visible difference does not prove an unseen cause or intervention.

5. Handoffs and follow-ups require the exchange, same recipient, same object, and claimed later action, location, or state.

6. Temporal relations must be verified using synchronized original timing or supplied mappings. Do not infer order from equal playback positions, separately sampled inputs, or timestamp proximity.

7. For "last" or "most recent," check all qualifying covered events before the reference event. For object trajectories, revisits, and interventions, verify every required continuity link and claimed event.

8. The declared answer must be supported and exactly one option must remain correct. Incorrect distractors need not appear, but no alternative option may also be supported.

Do not infer missing transitions or actions from adjacent samples. Do not use captions, subtitles, transcripts, filenames, outside knowledge, hidden generator intent, option wording as evidence, or unsupported assumptions.

Set verdict to fail if any required claim or link is missing, ambiguous, contradicted, or inferred. Otherwise set verdict to pass.

Binary decision contract:
- Return only the fields in the requested JSON schema.
- Do not include reasoning, markdown, or code fences outside the JSON object.
- The first JSON field must be verdict, with exactly one lowercase value: pass or fail.
- Decide verdict before writing reason or fix.
- If verdict is fail, reason must identify the concrete instance-level failure and fix must give a repair targeted to that failure.
- If verdict is pass, reason and fix must both be JSON null.
- Never use a generic placeholder sentence as reason or fix.
- Keep a failure reason and fix to one sentence each and no more than 40 words each.
- Do not assign a numerical score, quality label, rank, quota, or comparison against other candidates.


Video set metadata:
{
  "evidence_id": "EVIDENCE_PLACEHOLDER_001",
  "required_users": [
    "SpeakerUser",
    "ProviderOne",
    "ProviderTwo",
    "ProviderThree",
    "ProviderFour",
    "ProviderFive"
  ],
  "role_contract": {
    "speaker_user": "SpeakerUser",
    "evidence_provider_user": "ProviderOne",
    "evidence_provider_users": [
      "ProviderOne",
      "ProviderTwo",
      "ProviderThree",
      "ProviderFour",
      "ProviderFive"
    ],
    "provider_users": [
      "ProviderOne",
      "ProviderTwo",
      "ProviderThree",
      "ProviderFour",
      "ProviderFive"
    ],
    "required_users_order": "required_users[0] is the speaker. required_users[1] through required_users[5] are providers. For generation, the speaker input contains every frame sampled for CLIP clustering (normally 30 one-per-second images), while each provider input contains only sampled members of clusters that survived pruning. The speaker input must naturally motivate the question but remain insufficient to answer it, while the combined six-user image input must support one unique answer. One or more provider views may support the answer; an unused provider does not invalidate the item."
  },
  "prompt_requirement": "Use the visual media directly and write the strongest natural, grounded question supported by the current evidence. Do not cite timestamps in the user-facing question or options.",
  "clips": [
    {
      "user": "SpeakerUser",
      "media_role": "speaker_full_video",
      "is_pruned": false
    },
    {
      "user": "ProviderOne",
      "media_role": "provider_full_video",
      "is_pruned": false
    },
    {
      "user": "ProviderTwo",
      "media_role": "provider_full_video",
      "is_pruned": false
    },
    {
      "user": "ProviderThree",
      "media_role": "provider_full_video",
      "is_pruned": false
    },
    {
      "user": "ProviderFour",
      "media_role": "provider_full_video",
      "is_pruned": false
    },
    {
      "user": "ProviderFive",
      "media_role": "provider_full_video",
      "is_pruned": false
    }
  ]
}

Generated question-answer item:
{
  "qa_id": "QA_PLACEHOLDER_001",
  "question_type": "neutral",
  "question": "Where was the mug placed after I handed it over?",
  "options": [
    "On the wooden desk",
    "Beside the kitchen sink",
    "Near the front door",
    "On the living-room sofa",
    "Inside a dark backpack"
  ],
  "correct": "B",
  "answer": "Beside the kitchen sink",
  "required_users": [
    "SpeakerUser",
    "ProviderOne",
    "ProviderTwo",
    "ProviderThree",
    "ProviderFour",
    "ProviderFive"
  ]
}

Return exactly one valid JSON object with this exact shape:
{
  "verdict": "pass/fail",
  "reason": "specific instance-level failure reason when verdict is fail; null when verdict is pass",
  "fix": "specific repair for that failure when verdict is fail; null when verdict is pass"
}
```

## Prompt 3 — answerability / asker only

```text
You are an evidence-sufficiency judge for an EgoLife multiple-choice question.

Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.


Determine whether the media supplied for this condition directly contains all visual facts needed to distinguish exactly one option. Do not answer the question or reveal which option is correct.

Set verdict to pass only when every required subject, object, action, attribute, location, identity or continuity link, state, and temporal relation is visible and sufficiently clear. Set verdict to fail when any required fact is absent, occluded, ambiguous, contradictory, or requires guessing, outside knowledge, option-wording clues, or omitted media.

Rules:
- The first JSON field must be `verdict`; decide it before generating the later explanation and evidence lists.
- Judge this condition independently. Do not assume speaker_only is insufficient or combined_all_six_users is sufficient.
- Use the question and options only to identify required facts, never as evidence.
- Do not output an option letter, option text, declared answer, or inferred answer.
- Describe evidence using short, answer-neutral fact descriptions.
- Identity and continuity require visible continuity or distinguishing evidence, not roles, timing, lookalikes, similar clothing, or similar objects.
- State changes require the same object or place and both visible states. A visible difference does not prove an unseen cause or intervention.
- Handoffs and follow-ups require visible evidence of the exchange, the same recipient, the same object, and the claimed later action, location, or state.
- Temporal relations require synchronized timing or supplied mappings, not equal playback positions or timestamp proximity.
- "Last" or "most recent" requires checking all qualifying covered events before the reference event.
- `verdict` must be exactly the lowercase string `pass` or `fail`. `reason` must identify the decisive visible support for pass or the first decisive missing, occluded, ambiguous, or contradictory fact for fail. `available_evidence` and `missing_evidence` must be JSON arrays of short strings.
- This condition contains only the full unpruned sampled speaker timeline. Evaluate only what that timeline visibly establishes; do not assume facts from omitted provider views.

Condition:
{
  "condition_id": "speaker_only::SpeakerUser",
  "condition_type": "speaker_only",
  "users": [
    "SpeakerUser"
  ]
}

Generated question:
Where was the mug placed after I handed it over?

Answer options:
A. On the wooden desk
B. Beside the kitchen sink
C. Near the front door
D. On the living-room sofa
E. Inside a dark backpack

Return exactly one JSON object that conforms to the JSON Schema below. Return a data instance, not the schema itself:
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "verdict",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": [
        "pass",
        "fail"
      ]
    },
    "reason": {
      "type": "string",
      "minLength": 1
    },
    "available_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    },
    "missing_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    }
  }
}
```

## Prompt 4 — answerability / all six

```text
You are an evidence-sufficiency judge for an EgoLife multiple-choice question.

Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.


Determine whether the media supplied for this condition directly contains all visual facts needed to distinguish exactly one option. Do not answer the question or reveal which option is correct.

Set verdict to pass only when every required subject, object, action, attribute, location, identity or continuity link, state, and temporal relation is visible and sufficiently clear. Set verdict to fail when any required fact is absent, occluded, ambiguous, contradictory, or requires guessing, outside knowledge, option-wording clues, or omitted media.

Rules:
- The first JSON field must be `verdict`; decide it before generating the later explanation and evidence lists.
- Judge this condition independently. Do not assume speaker_only is insufficient or combined_all_six_users is sufficient.
- Use the question and options only to identify required facts, never as evidence.
- Do not output an option letter, option text, declared answer, or inferred answer.
- Describe evidence using short, answer-neutral fact descriptions.
- Identity and continuity require visible continuity or distinguishing evidence, not roles, timing, lookalikes, similar clothing, or similar objects.
- State changes require the same object or place and both visible states. A visible difference does not prove an unseen cause or intervention.
- Handoffs and follow-ups require visible evidence of the exchange, the same recipient, the same object, and the claimed later action, location, or state.
- Temporal relations require synchronized timing or supplied mappings, not equal playback positions or timestamp proximity.
- "Last" or "most recent" requires checking all qualifying covered events before the reference event.
- `verdict` must be exactly the lowercase string `pass` or `fail`. `reason` must identify the decisive visible support for pass or the first decisive missing, occluded, ambiguous, or contradictory fact for fail. `available_evidence` and `missing_evidence` must be JSON arrays of short strings.
- This condition contains the full unpruned sampled speaker timeline and all five full unpruned sampled provider timelines. Combine visible evidence across them when needed. Some provider views may be irrelevant.

Condition:
{
  "condition_id": "combined_all_six_users::SpeakerUser+ProviderOne+ProviderTwo+ProviderThree+ProviderFour+ProviderFive",
  "condition_type": "combined_all_six_users",
  "users": [
    "SpeakerUser",
    "ProviderOne",
    "ProviderTwo",
    "ProviderThree",
    "ProviderFour",
    "ProviderFive"
  ]
}

Generated question:
Where was the mug placed after I handed it over?

Answer options:
A. On the wooden desk
B. Beside the kitchen sink
C. Near the front door
D. On the living-room sofa
E. Inside a dark backpack

Return exactly one JSON object that conforms to the JSON Schema below. Return a data instance, not the schema itself:
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "verdict",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": [
        "pass",
        "fail"
      ]
    },
    "reason": {
      "type": "string",
      "minLength": 1
    },
    "available_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    },
    "missing_evidence": {
      "type": "array",
      "items": {
        "type": "string",
        "minLength": 1
      }
    }
  }
}
```

## Archived previous output contracts

These contracts are preserved here for audit and possible rollback. They are not the active per-judge production contract.

### Previous qa_formality contract

```json
{
  "review_passed": true,
  "checks": {
    "qa_formality": {
      "status": "PASS/FAIL",
      "reason": "one short explanation based only on this judge's assigned scope",
      "fix": "one specific repair instruction if FAIL; empty string if PASS",
      "semantic_subchecks": {
        "first_person_perspective": {
          "status": "PASS/FAIL",
          "reason": "whether the question uses a natural first-person or shared-memory perspective"
        },
        "naturalness_and_clarity": {
          "status": "PASS/FAIL",
          "reason": "whether the question and options are natural, concrete, clear, exclusive, and parallel"
        },
        "other_person_activity_query": {
          "status": "PASS/FAIL",
          "reason": "whether the answer target is a prohibited concurrent activity report"
        },
        "direct_name_leakage": {
          "status": "PASS/FAIL",
          "reason": "whether the user-facing QA directly names a participant"
        },
        "timestamp_citation": {
          "status": "PASS/FAIL",
          "reason": "whether the user-facing QA cites dataset-like temporal coordinates"
        }
      }
    }
  },
  "blocking_failures": [
    "names of failed checks that should block acceptance"
  ],
  "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed"
}
```

### Previous evidence_groundedness contract

```json
{
  "review_passed": true,
  "checks": {
    "evidence_groundedness": {
      "status": "PASS/FAIL",
      "reason": "one short explanation based only on this judge's assigned scope",
      "fix": "one specific repair instruction if FAIL; empty string if PASS"
    }
  },
  "blocking_failures": [
    "names of failed checks that should block acceptance"
  ],
  "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed"
}
```

### Previous scored check fields

```json
{
  "status": "PASS/FAIL",
  "reason": "one short explanation based only on this judge's assigned scope",
  "fix": "one specific repair instruction if FAIL; empty string if PASS",
  "quality_score": "1/2/3 using the check-specific quality rubric",
  "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
  "quality_reason": "required rationale for the quality score",
  "quota_rebuttal": "required only for an exceptional score of 3 after quota exhaustion; otherwise empty"
}
```

### Previous one-call combined judge contract

```json
{
  "review_passed": true,
  "checks": {
    "qa_formality": {
      "status": "PASS/FAIL",
      "reason": "one short explanation based only on this judge's assigned scope",
      "fix": "one specific repair instruction if FAIL; empty string if PASS",
      "quality_score": "1/2/3 using the check-specific quality rubric",
      "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
      "quality_reason": "required rationale for this attempt's quality score; this does not determine pass/fail status",
      "quota_rebuttal": "required explicit rebuttal only when assigning 3 after the 48-assignment quota is exhausted; otherwise empty string",
      "semantic_subchecks": {
        "first_person_perspective": {
          "status": "PASS/FAIL",
          "reason": "whether the question is written as a natural first-person or shared-memory question using I, me, my, we, us, or our"
        },
        "naturalness_and_clarity": {
          "status": "PASS/FAIL",
          "reason": "whether the question is conversational, concrete, unambiguous, and paired with clear, mutually exclusive, parallel options"
        },
        "other_person_activity_query": {
          "status": "PASS/FAIL",
          "reason": "whether the question asks what one person was doing concurrently with another event instead of asking for a concrete missing object, identity, state, location, placement, outcome, consequence, explanation, interaction result, or follow-up"
        },
        "direct_name_leakage": {
          "status": "PASS/FAIL",
          "reason": "whether the user-facing question or options directly name a required user or another participant"
        },
        "timestamp_citation": {
          "status": "PASS/FAIL",
          "reason": "whether the user-facing question and options avoid clock times, timecodes, timestamps, frame numbers, seconds-from-start, and minute-mark citations"
        }
      }
    },
    "evidence_groundedness": {
      "status": "PASS/FAIL",
      "reason": "one short explanation based only on this judge's assigned scope",
      "fix": "one specific repair instruction if FAIL; empty string if PASS",
      "quality_score": "1/2/3 using the check-specific quality rubric",
      "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
      "quality_reason": "required rationale for this attempt's quality score; this does not determine pass/fail status",
      "quota_rebuttal": "required explicit rebuttal only when assigning 3 after the 48-assignment quota is exhausted; otherwise empty string"
    }
  },
  "blocking_failures": [
    "names of failed checks that should block acceptance"
  ],
  "why_generator_asked_this": "brief explanation of why the generator may have asked this",
  "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed"
}
```

### Previous verbose response sample

```json
{
  "review_passed": false,
  "checks": {
    "evidence_groundedness": {
      "status": "FAIL",
      "reason": "The claimed object placement is not visible.",
      "fix": "Revise the answer to match a visible placement."
    }
  },
  "blocking_failures": [
    "evidence_groundedness"
  ],
  "feedback_to_generator": "Revise the answer to match a visible placement."
}
```

Historical prompt source snapshots also remain under docs/others/historical_prompts; no dataset examples or prior generated QA files are removed by this migration. In the source worktree, the previous ordinal training contract and exact old manifest snapshots are archived under score-only-reward-model/training/grpo_v3/experiments/human_preference_reviewer/ARCHIVED_ORDINAL_REVIEWER_V1.md and its archive/ordinal_v1_manifests directory. The synchronized OneDrive copy is isolated under score-only-reward-model-new so the previous score-only-reward-model directory remains untouched.

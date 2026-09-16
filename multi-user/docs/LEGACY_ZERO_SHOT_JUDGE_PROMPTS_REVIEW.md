# Legacy-zero-shot judge prompts and output contracts

> Review snapshot generated from the active prompt builders. All example and placeholder values are English. This document does not contain real videos, private annotations, or deleted samples.

## Scope

The active six-user RLHF path makes four judgments: one qa_formality call, one evidence_groundedness call, one asker-only answerability call, and one all-six answerability call. The two answerability calls share the same schema and boolean semantics.

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

## Preserved legacy-zero-shot answerability contract

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "answerable",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "answerable": {
      "type": "boolean"
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

The meaning is unchanged: answerable is true exactly when the videos supplied for that condition are sufficient. The target pipeline behavior is asker-only=false and all-six=true. available_evidence and missing_evidence remain in the model output for structured audit.

## Judge-model training contract

```json
{
  "contract_version": "binary_reviewer_v2",
  "supervision_type": "class_weighted_binary_decisions",
  "human_score_to_binary_label": {
    "1": 0,
    "2": 1,
    "3": 1
  },
  "binary_label_names": {
    "0": "fail",
    "1": "pass"
  },
  "head_type": "two_logit_binary_classifier",
  "class_logit_order": [
    "fail",
    "pass"
  ],
  "loss_function": "cross_entropy",
  "class_weighting": "balanced_inverse_frequency_from_training_split_only",
  "class_weight_reduction": "sum_weighted_losses_divided_by_sample_count",
  "head_loss_aggregation": "equal_mean"
}
```

The original human columns stay 1-3 in the archived data. Training maps score 1 to fail and scores 2-3 to pass. For each judge head, class weights are computed only from its training-split fail/pass counts as N / (2 * N_c). Each head uses class-weighted cross-entropy; active head losses are averaged equally. The classification heads do not directly supervise reason/fix text tokens. They can influence deployment reasoning only through the shared LoRA parameters, while the deployment prompt separately enforces verdict-first generation.

### Binary reviewer training input prompt

This is the exact text instruction emitted by the active training prompt builder for the English placeholder candidate. Video content is attached separately as Video A and Video B.

```text
You are reviewing one two-user multiple-choice QA candidate.
Video A (speaker): SpeakerUser
Video B (provider): ProviderUser
Use both synchronized videos and the complete QA below.
Judge visual evidence, which view or views are required, and instruction-following formality.
Do not generate an explanation; return hidden states for the binary fail/pass heads.
Candidate QA:
{"question":"Where was the mug placed after I handed it over?","options":["On the wooden desk","Beside the kitchen sink","Near the front door","On the living-room sofa","Inside a dark backpack"],"correct":"B","answer":"Beside the kitchen sink"}
```

## Prompt 1 — qa_formality

```text
You are the qa_formality judge for a six-user multiple-choice question. You are a pure text-only semantic judge and do not see the videos.

Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.


Judge only the deterministic schema result and the user-facing question and options. Do not use hidden generator intent to rescue unclear wording.

Apply every criterion below before deciding the single overall verdict. Do not return per-criterion or subcheck fields:

1. first_person_perspective
- PASS only when the question sounds like a natural first-person or shared-memory question from someone in the situation and uses I, me, my, we, us, or our.
- The options do not need first-person pronouns.
- FAIL third-person wording or questions with no asker perspective.

2. naturalness_and_clarity
- PASS when the question is conversational, concrete, grammatical, and unambiguous, and the five options answer the same question in mutually exclusive, reasonably parallel forms.
- FAIL vague references, incompatible option types, dataset language such as video/clip/frame/camera/evidence provider, or wording that would be unnatural for someone recalling their experience.
- Judge semantic form only, not whether the described facts are true.
- For a six-user item, PASS only when the question expresses a plausible information need the speaker would naturally have after their own visible experience. FAIL a contrived third-party quiz whose setup gives the speaker no reason to care or ask.

3. other_person_activity_query
- FAIL when the question asks what one person was doing while or when another person was doing something else, and the answer is that person's concurrent activity.
- Apply this restriction in every direction: reject an asker-side event used to query a provider's activity, a provider-side event used to query the asker's activity, and one provider's event used to query another provider's activity.
- FAIL pair-matching questions whose options encode two or more concurrent activities.
- The question still FAILS when the anchor event is concrete, the wording is natural, or the temporal overlap can be verified from synchronized recordings.
- PASS linked task outcomes, interactions, and post-handoff follow-ups only when the answer target is a concrete object, identity, state, location, placement, outcome, consequence, explanation, interaction result, or follow-up rather than a concurrent activity report.
- Do not judge whether the described facts are visually grounded, whether media was cropped, or whether one view is sufficient.

Long-horizon structural wording checks (apply when the item uses one of these optional relations):
- Object trajectory: allow a natural question about where the same object came from, went, or was later found.
- Cross-user before/after state: allow a natural comparison of an earlier and later state without requiring the question to explain an unseen cause.
- Same-user revisit with a cross-user intervention: allow a natural before/leave/intervention/return structure when the provider interaction is the requested missing detail.
- Last-seen or most-recent interaction: allow natural words such as "last" or "most recent" when the question clearly identifies the object, event, and reference point.
- Cross-user temporal ordering: allow a natural first/before/after/between question about clearly identified related events; this is not a prohibited concurrent-activity query.
- These five patterns are optional and equal-status. Do not fail an otherwise natural item merely because it uses one of them. Judge wording and semantic form only; leave visual identity, continuity, state, intervention, recency, and timing truth to the visual judges.


4. direct_name_leakage
- FAIL when the question or any option directly names a required user or another participant. PASS otherwise.
- Natural descriptive references such as "the person in the dark jacket beside the television" are allowed.
- Required-user names below are provided only for this text comparison.

5. timestamp_citation
- FAIL when the question or any option cites a clock time, timestamp, timecode, frame number, seconds-from-start, minute mark, or similar dataset-like temporal coordinate.
- Examples that FAIL include "around 12:53", "at 00:42", "at timestamp 35.2", "during the first 15 seconds", and "near frame 200".
- Natural relative wording such as while, when, before, after, later, at the same time, or a few minutes later is allowed.
- Internal evidence timeframes are outside this judge's scope and are not shown.

6. ambiguous_reference
- Judge whether a reference is resolvable in its local sentence and described situation, not whether its wording would uniquely identify one entity across the entire recording.
- PASS concise natural descriptions such as "the person beside the television" or "the mug I left by the sink" when the local wording makes the intended referent usable.
- FAIL only when two or more equally plausible referents would materially change the meaning or answer, or when the wording supplies no usable identifying context.
- FAIL when the question is asked in a second-person perspective, for example "what were you doing".
- Bare phrases such as "the other room", "the other person", or "the cup" FAIL when the surrounding sentence does not resolve them; they are not automatic failures when local context does resolve them.

Deterministic structure rules:
- The deterministic schema branch must PASS.
- The item must contain exactly five non-empty options in A-E order, one correct letter, and an answer that exactly matches the selected option. The option strings themselves do not need A./B./C./D./E. prefixes.

Decision rules:
- Set verdict to fail when the deterministic schema branch fails or any semantic criterion fails.
- Set verdict to pass only when the deterministic schema branch and every semantic criterion pass.
- On failure, reason must name the actual failed requirement in this candidate and fix must target that failure. Do not return a generic summary of the rubric.

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


You will see the full unpruned sampled visual timeline for every user. This is fuller than the generator input because provider redundancy pruning is not applied to judge media. Judge only visual and temporal grounding. Do not fail for names, missing first-person wording, awkward phrasing, timestamp citations, or schema style. Do not decide whether a single-user condition is sufficient.

evidence_groundedness asks whether the material claims and declared answer are supported by the videos and metadata:
- Infer no hidden generator interpretation; judge the question, declared answer, material option claims, and videos shown.
- Verify every material factual claim in the question stem and declared correct answer against concrete visible moments or supplied metadata.
- Be very strict and verify every claim made in the question.
- Incorrect distractors do not need to occur in the videos for an ordinary object, state, action, or location MCQ; they must simply not make the declared answer ambiguous.
- For a comparison whose options make concrete claims about both operands, verify the declared complete relation and ensure no alternative option is also supported.
- Treat every object, action, person, state, identity, and continuity description as unverified. The generator may hallucinate or misidentify them.
- Do not accept a claimed transition, continuous action, or intermediate event merely because it seems plausible between adjacent sampled images. Require direct support in the supplied unpruned samples; otherwise FAIL.
- Do not use outside knowledge, captions, transcripts, filenames alone, or assumptions not visible in the videos or metadata.
- Treat required_users[0] as the speaker and required_users[1] through required_users[5] as providers. Verify that the full unpruned speaker timeline grounds the specific experience, object, person, or interaction that makes the question natural, and that at least one external provider view or provider combination supplies the answer-bearing continuation or detail. Do not fail merely because an input provider is unused.
- For identity or role linkage, verify enough visible continuity or distinguishing evidence to establish same-person versus different-person rather than inferring identity from roles, timing, or option wording.
- For a post-handoff follow-up, verify the initial exchange, same recipient, same object, and claimed later action/location/state. FAIL links based only on lookalikes, similar objects, or temporal proximity.
- For state verification, verify the exact object and observed state. Accept a claimed change only when both earlier and later states are visible.
- For any temporal claim, verify the claimed events and their relation on the original synchronized timeline. Do not compare equal playback positions in independently pruned videos; use original-video time or supplied pruned-to-original maps.
- FAIL a temporal relation when it is false, vague, or inferred only from timestamp proximity instead of verified synchronized intervals.
- PASS only when the question stem and declared correct answer are clearly supported and exactly one option remains correct.

Long-horizon grounding checks (apply when the generated item uses one of these relations):
- Object trajectory: verify that the observations concern the same physical object and that every claimed handoff, relocation, or endpoint is directly visible. Similar appearance or temporal proximity alone does not establish continuity.
- Cross-user before/after state: verify the same object or place, both distinct visible states, and any claimed intervening action. Different visible states establish a difference, not an unseen cause.
- Same-user revisit with a cross-user intervention: verify both speaker visits and the provider's claimed answer-bearing intervention. Do not infer that intervention merely from a difference between the two visits.
- Last-seen or most-recent interaction: compare all qualifying visible events before the reference event. Accept "last" or "most recent" only when the available coverage rules out a later qualifying event in that interval.
- Cross-user temporal ordering: verify each event and compare original synchronized timing. Never infer cross-user order from equal positions in separately pruned or context-thinned inputs.
- Temporal distance does not compensate for a missing link. Reject a long-horizon claim when identity, continuity, state, intervention, or order is assumed rather than visibly supported.


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


Your task is to determine whether the videos supplied for this condition contain enough visible evidence to produce a grounded answer to the generated question. Do not answer the question yourself.

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

Answer options (for judging whether the evidence resolves the question, not for selecting one):
A. On the wooden desk
B. Beside the kitchen sink
C. Near the front door
D. On the living-room sofa
E. Inside a dark backpack

Rules:
- The first JSON field must be `answerable`; decide this boolean before generating the later explanation and evidence lists.
- Return `answerable: true` only when the supplied videos directly contain the answer-relevant visual facts needed to distinguish one option from the alternatives.
- Return `answerable: false` when a required subject, object, action, attribute, location, identity link, state change, or temporal relation is missing, occluded, too ambiguous, or would require guessing or outside knowledge.
- Do not select an option. Do not output an A-E letter, the final answer, or the text of the option you think is correct.
- Describe evidence availability at the level of needed facts, such as whether the relevant object and action are visible. Do not reveal the answer while explaining the judgment.
- Judge only the visible videos and supplied condition metadata. Do not use the wording of the question or options as evidence.
- Do not assume the speaker-only condition is unanswerable or the six-video condition is answerable. Decide each condition independently from its actual visual evidence.
- Make `reason` specific to the supplied condition: identify the decisive visible support when answerable is true, or the concrete missing, occluded, ambiguous, or contradictory fact when answerable is false. Never use a generic placeholder sentence.
- `answerable` must be a JSON boolean, not a quoted string. `available_evidence` and `missing_evidence` must be JSON arrays of short strings.
- This condition contains only the full unpruned sampled speaker timeline. Evaluate only what that timeline visibly establishes; do not assume facts from omitted provider views.

Long-horizon grounding checks (apply when the generated item uses one of these relations):
- Object trajectory: verify that the observations concern the same physical object and that every claimed handoff, relocation, or endpoint is directly visible. Similar appearance or temporal proximity alone does not establish continuity.
- Cross-user before/after state: verify the same object or place, both distinct visible states, and any claimed intervening action. Different visible states establish a difference, not an unseen cause.
- Same-user revisit with a cross-user intervention: verify both speaker visits and the provider's claimed answer-bearing intervention. Do not infer that intervention merely from a difference between the two visits.
- Last-seen or most-recent interaction: compare all qualifying visible events before the reference event. Accept "last" or "most recent" only when the available coverage rules out a later qualifying event in that interval.
- Cross-user temporal ordering: verify each event and compare original synchronized timing. Never infer cross-user order from equal positions in separately pruned or context-thinned inputs.
- Temporal distance does not compensate for a missing link. Reject a long-horizon claim when identity, continuity, state, intervention, or order is assumed rather than visibly supported.


Return exactly one JSON object that conforms to the JSON Schema below. Return a data instance, not the schema itself:
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "answerable",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "answerable": {
      "type": "boolean"
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


Your task is to determine whether the videos supplied for this condition contain enough visible evidence to produce a grounded answer to the generated question. Do not answer the question yourself.

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

Answer options (for judging whether the evidence resolves the question, not for selecting one):
A. On the wooden desk
B. Beside the kitchen sink
C. Near the front door
D. On the living-room sofa
E. Inside a dark backpack

Rules:
- The first JSON field must be `answerable`; decide this boolean before generating the later explanation and evidence lists.
- Return `answerable: true` only when the supplied videos directly contain the answer-relevant visual facts needed to distinguish one option from the alternatives.
- Return `answerable: false` when a required subject, object, action, attribute, location, identity link, state change, or temporal relation is missing, occluded, too ambiguous, or would require guessing or outside knowledge.
- Do not select an option. Do not output an A-E letter, the final answer, or the text of the option you think is correct.
- Describe evidence availability at the level of needed facts, such as whether the relevant object and action are visible. Do not reveal the answer while explaining the judgment.
- Judge only the visible videos and supplied condition metadata. Do not use the wording of the question or options as evidence.
- Do not assume the speaker-only condition is unanswerable or the six-video condition is answerable. Decide each condition independently from its actual visual evidence.
- Make `reason` specific to the supplied condition: identify the decisive visible support when answerable is true, or the concrete missing, occluded, ambiguous, or contradictory fact when answerable is false. Never use a generic placeholder sentence.
- `answerable` must be a JSON boolean, not a quoted string. `available_evidence` and `missing_evidence` must be JSON arrays of short strings.
- This condition contains the full unpruned sampled speaker timeline and all five full unpruned sampled provider timelines. Combine visible evidence across them when needed. Some provider views may be irrelevant.

Long-horizon grounding checks (apply when the generated item uses one of these relations):
- Object trajectory: verify that the observations concern the same physical object and that every claimed handoff, relocation, or endpoint is directly visible. Similar appearance or temporal proximity alone does not establish continuity.
- Cross-user before/after state: verify the same object or place, both distinct visible states, and any claimed intervening action. Different visible states establish a difference, not an unseen cause.
- Same-user revisit with a cross-user intervention: verify both speaker visits and the provider's claimed answer-bearing intervention. Do not infer that intervention merely from a difference between the two visits.
- Last-seen or most-recent interaction: compare all qualifying visible events before the reference event. Accept "last" or "most recent" only when the available coverage rules out a later qualifying event in that interval.
- Cross-user temporal ordering: verify each event and compare original synchronized timing. Never infer cross-user order from equal positions in separately pruned or context-thinned inputs.
- Temporal distance does not compensate for a missing link. Reject a long-horizon claim when identity, continuity, state, intervention, or order is assumed rather than visibly supported.


Return exactly one JSON object that conforms to the JSON Schema below. Return a data instance, not the schema itself:
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "answerable",
    "reason",
    "available_evidence",
    "missing_evidence"
  ],
  "properties": {
    "answerable": {
      "type": "boolean"
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

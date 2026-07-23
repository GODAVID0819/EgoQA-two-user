# Underrepresented question-family audit

## Scope

This audit measures what generators proposed, so it uses every parseable generation
attempt rather than only accepted rows.

- Direct category-free controls:
  - `qa_mcq.intermediate_with_ratioanle.jsonl`: 48 parseable candidates.
  - `qa_mcq.intermediate_without_rationale.jsonl`: 45 parseable candidates.
  - Combined direct control set: 93 candidates.
- Direct category-steered run:
  - `qa_mcq.intermediate (9).jsonl`: 31 parseable candidates.
- Broader archive:
  - 818 distinct category-free question strings after whitespace/case normalization
    and exclusion of the known category-steered files.

The rare-relation counts below were manually checked against the actual question
wording. Broad temporal/concurrent incidence was used only to establish that those
forms were already common, not to assign a new category to every archived item.

## Findings

| Question family | Direct category-free controls | Broader category-free archive | Categorized-run evidence | Conclusion |
|---|---:|---:|---|---|
| Cross-view comparison or role asymmetry | 0 / 93 clear comparisons | 6 / 818 | The later combined quantity/comparison prompt generated counts, not comparisons | Clearly underrepresented; potentially useful because both views supply comparison operands |
| Cross-view identity or role linkage | 0 / 93 | 1 / 818 | One discovery-mode item asked whether the laptop user was also the food preparer; it was accepted | Extremely underrepresented and inherently relational, but supported by only one categorized example |
| Post-handoff recipient follow-up | 0 / 93 | 1 / 818 clear transfer outcome | Two true handoff-follow-up questions appeared in the older transfer-chain run; both were rejected | Underrepresented, but current evidence shows a grounding/answerability risk |
| Concrete state change or verification | 5 / 93 | Present but uncommon | 5 clean state questions among 31 categorized candidates, 4 accepted | Reliably underrepresented in the controlled comparison (16.1% versus 5.4%), though often answerable from the provider view alone |
| Exact visible count | 0 / 93 | 2 / 818, both pizza-box paraphrases | Two egg-count variants, one accepted | Literally rare, but not a useful family for the intended benchmark; steering removed |
| Strict concurrent activity-pair comparison | 0 / 93 clear pair-matching items | Not observed as an option-pair form in the reviewed archive | Ordinary anchor-to-remote-event questions existed, but not the stricter form where every option pairs one event visible in each view | Useful strict two-view direction: both views contain multiple activities and only original-time alignment identifies which cross-view pair overlaps |

## Not underrepresented

- Object identification and visible attributes dominated the category-free controls.
- Object location/placement and ordinary task details appeared repeatedly without
  category steering.
- Temporal continuation was already common: the direct controls included first-item,
  screen-transition, and later-outcome questions.
- Ordinary concurrent/elsewhere activity wording was also common. Its main issue
  was judge formality and evidence quality, not generator coverage. This is
  distinct from the newly proposed activity-pair comparison, which makes every
  option a pairing of one event visible in each view and is genuinely
  underrepresented.

## Recommended interpretation

The active category-free implicit prompt now includes cross-view
comparison/asymmetry, cross-view identity/role linkage, concrete state
verification, post-handoff recipient follow-up, and strict concurrent
activity-pair comparison as equal-status optional directions. Post-handoff remains
the highest-risk family, so its prompt and judge rules require the exact exchange,
recipient, object, and later outcome to be visually trackable; similarity and
temporal proximity are explicitly insufficient. Exact count remains excluded.

Source: C:\Users\haoya\Downloads\qa_mcq.intermediate_new_prompt_eval.jsonl
Records: 50

Prompt occurrence counts, including nested generation_trace copies:
- generator: 340
- judge_evidence_groundedness: 337
- judge_qa_formality: 337

Unique exact prompt counts by content hash:
- generator: 116
- judge_evidence_groundedness: 117
- judge_qa_formality: 117

Files:
- unique_exact_prompts/: one file per unique exact prompt string
- unique_prompt_manifest.tsv: maps each unique prompt file to first occurrence
- prompt_occurrences.tsv: maps every stored prompt occurrence to its unique hash
- all_attempt_prompts/: non-deduplicated top-level full attempt prompts from the first extraction

# EgoLife binary judge annotation

You have one assignment containing 20 source packets. Please use only the folder assigned to you.

## Start

1. On Windows, double-click `open_rlhf_labeling.cmd`.
2. On macOS or Linux, run `python serve_rlhf_labeling.py` from the assignment folder.
3. Enter your assigned anonymous annotator ID at the top of the page.

The package contains no images or videos. Source clips stream from Hugging Face, so an internet connection is required.

## Label each question

- Select PASS or FAIL for QA formality.
- Select PASS or FAIL for evidence groundedness.
- For answerability, judge `Asker only` and `All six users` independently.
- Common failure-mode checkboxes are optional.
- Manual notes are optional.
- Use Skip only when the question genuinely cannot be assessed, and provide a skip reason.

Question tabs are intentionally anonymized as Q1, Q2, and so on. Do not try to infer generation order. Generator-reported user/time locations are navigation aids, not verified evidence; always confirm against the source video.

## Save and return

Progress saves automatically in your browser. Use Backup periodically. After reopening the package, click `Import backup JSON` and select that backup to repopulate completed and partially completed labels, failure-mode checkboxes, notes, skip state, time metadata, and the annotator ID. The import merges matching candidate IDs into the current assignment and does not clear candidates absent from the backup.

When finished, return:

1. The JSONL file from `Export JSONL`.
2. The JSON backup file from `Backup`.
3. Optionally, the CSV file from `Export CSV`.

Do not edit candidate IDs, packet IDs, or exported provenance fields.

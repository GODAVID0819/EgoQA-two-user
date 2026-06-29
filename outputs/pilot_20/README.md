# Pilot Output Directory

Runtime files are written here:

- `manifest.json`
- `observations.jsonl`
- `observation_prompts.jsonl`
- `evidence_manifest.jsonl`
- `generation_prompts.jsonl`
- `qa_mcq.raw.jsonl`
- `qa_mcq.jsonl`
- `qa_mcq.csv`
- `generation_report.md`

The repository does not include generated QA rows because this local machine has no GPU for `Qwen/Qwen3-VL-8B-Instruct`.

Gaze note: EgoLife EyeGaze CSVs are Aria CPF yaw/pitch/depth, not image-space `gaze_x/gaze_y`. Runtime rows will contain `projection_status=missing_calibration` unless an Aria RGB calibration JSON directory is passed to `observe_clips` or `prepare_evidence`.

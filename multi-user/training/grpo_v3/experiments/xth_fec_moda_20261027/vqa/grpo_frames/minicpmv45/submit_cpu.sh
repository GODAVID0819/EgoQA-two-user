#!/bin/bash
set -euo pipefail
ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
cd "$ROOT/code"
test -f cpu_prepare.sbatch
test ! -e "$ROOT/jobs/cpu_submission.json"
bash -n cpu_prepare.sbatch
JOB=$(sbatch --parsable cpu_prepare.sbatch)
JOB=${JOB%%;*}
test "$JOB" -gt 0
python3 - "$ROOT" "$JOB" <<'PY'
import json,sys,datetime
from pathlib import Path
root=Path(sys.argv[1]);value={'job_id':sys.argv[2],'stage':'cpu_prepare_and_assets','gpu_count':0,
    'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'jobs/cpu_submission.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
PY
squeue -j "$JOB" -o '%.18i %.9T %.10M %R'

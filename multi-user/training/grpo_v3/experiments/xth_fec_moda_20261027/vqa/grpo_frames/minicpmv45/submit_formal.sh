#!/bin/bash
set -euo pipefail
ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
cd "$ROOT/code"
CPU_JOB=$(python3 -c 'import json;print(json.load(open("../jobs/cpu_submission.json"))["job_id"])')
STATE=$(sacct -X -n -j "$CPU_JOB" -o State | head -1 | xargs)
test "$STATE" = COMPLETED
test ! -e "$ROOT/jobs/formal_submission.json"
/scratch/xl6775/envs/minicpmv45_grpo131_20261031/bin/python -m unittest test_native_cache.py
python3 - <<'PY'
import json
r=json.load(open('cpu_ready.json'));assert r['n']==393 and r['max_with_output']<=40960 and not r['missing_frames']
PY
bash -n formal.sbatch
JOB=$(sbatch --parsable formal.sbatch)
JOB=${JOB%%;*}
test "$JOB" -gt 0
python3 - "$ROOT" "$JOB" <<'PY'
import json,sys,datetime
from pathlib import Path
root=Path(sys.argv[1]);value={'job_id':sys.argv[2],'stage':'formal_393','gpu_count':1,'gpu_type':'H200',
    'time_limit_hours':2,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'jobs/formal_submission.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
PY
squeue -j "$JOB" -o '%.18i %.9T %.10M %R'

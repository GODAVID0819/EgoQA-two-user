"""在指定更新步正常保存、验证并结束，保留完整学习率调度长度。"""
import json
import os
from pathlib import Path

try:
    from swift.callbacks import TrainerCallback, callbacks_map
except ModuleNotFoundError as exc:
    if exc.name != 'swift':
        raise
    class TrainerCallback:
        def __init__(self, args, trainer):
            self.args, self.trainer = args, trainer
    callbacks_map = {}


class StageStopCallback(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.stop_step = int(os.environ['EGOQA_GRPO_STOP_AT_STEP'])
        if not 0 < self.stop_step <= args.max_steps:
            raise ValueError('分阶段终点超过调度总步数')

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = True
        return control

    def _audit(self, event, args, state, **kwargs):
        destination = os.environ.get('EGOQA_GRPO_STAGE_AUDIT')
        if not destination or not getattr(state, 'is_world_process_zero', True):
            return
        path = Path(destination)
        record = json.loads(path.read_text()) if path.exists() else {}
        optimizer, scheduler = kwargs.get('optimizer'), kwargs.get('lr_scheduler')
        record[event] = {'global_step': state.global_step, 'schedule_max_steps': args.max_steps,
                         'stop_after_steps': self.stop_step,
                         'learning_rates': [g['lr'] for g in optimizer.param_groups] if optimizer else [],
                         'scheduler_last_epoch': getattr(scheduler, 'last_epoch', None)}
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            raise ValueError('恢复断点已经达到本阶段终点，拒绝重复训练')
        self._audit('train_begin', args, state, **kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._audit('train_end', args, state, **kwargs)
        return control


callbacks_map['egoqa_stage_stop'] = StageStopCallback

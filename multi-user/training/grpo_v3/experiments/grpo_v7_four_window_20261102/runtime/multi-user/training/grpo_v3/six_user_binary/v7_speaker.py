"""冻结的 speaker 盲答；只构造白名单输入，不携带候选隐藏字段。"""
from __future__ import annotations
import re
import time

SPEAKER_INSTRUCTION = (
    "Answer the multiple-choice question using only the speaker's supplied "
    "time-ordered images. Choose the best answer even if the evidence is incomplete. "
    "Output exactly one uppercase letter: A, B, C, D, or E. "
    "Do not output an explanation, punctuation, or any other text.\n\n"
)


def parse_letter(raw):
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    return raw if re.fullmatch('[A-E]', raw) else None


def blind_payload(qa, frames):
    question, options = qa.get('question'), qa.get('options')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('盲答缺少题干')
    if not isinstance(options, list) or len(options) != 5 or any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError('盲答必须保留原五个选项')
    if not frames or any(not isinstance(p, str) or not p for p in frames):
        raise ValueError('盲答必须有真实 speaker 帧')
    return {'prompt': SPEAKER_INSTRUCTION + 'Question:\n' + question + '\n\nOptions:\n'
            + ''.join(f'{letter}. {option}\n' for letter,option in zip('ABCDE', options)),
            'frames': list(frames)}


class FrozenSpeaker:
    """在 Judge 睡眠后懒加载；每批结束必须先释放，才可恢复 Judge/Policy。"""
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.sleeping = True
        self.last_metrics = {}

    def prepare_requests(self, payloads, decoder):
        decoded = {}
        requests = []
        for payload in payloads:
            frames = payload['frames']
            key = (tuple(frames), 3136, 50176, 14)
            content = [dict(type='image', image=frame, min_pixels=3136, max_pixels=50176) for frame in frames]
            content.append(dict(type='text', text=payload['prompt']))
            messages = [dict(role='user', content=content)]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if key not in decoded:
                images, videos = decoder(messages, image_patch_size=14)
                if videos or images is None or len(images) != len(frames):
                    raise RuntimeError('speaker 媒体占位与输入不匹配')
                decoded[key] = images
            requests.append(dict(prompt=prompt, multi_modal_data={'image':decoded[key]}))
        self.last_metrics = dict(batch_size=len(payloads), decoded_timelines=len(decoded),
                                 decoded_frames=sum(len(value) for value in decoded.values()),
                                 timeline_frame_counts=[len(value) for value in decoded.values()],
                                 decoded_image_sizes=[sorted({(getattr(im,'width',None),getattr(im,'height',None)) for im in value}) for value in decoded.values()],
                                 min_pixels=3136,max_pixels=50176,image_patch_size=14)
        return requests

    def predict_many(self, payloads):
        if not payloads:
            return []
        started = time.perf_counter()
        first_load = self.llm is None
        from transformers import AutoConfig, AutoProcessor
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        if self.llm is None:
            if self.sleeping is None:
                raise RuntimeError('speaker 初始化失败后显存状态未知')
            model = self.config['model_id']
            model_config=AutoConfig.from_pretrained(model,local_files_only=True)
            if model_config.model_type!='qwen2_5_vl' or model_config.vision_config.patch_size!=14:
                raise ValueError('盲答必须使用实际 Qwen2.5-VL，禁止继承27B的16像素patch几何')
            self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
            self.sleeping = None
            self.llm = LLM(model=model, dtype='bfloat16', tensor_parallel_size=1,
                max_model_len=self.config.get('max_model_len', 32768),
                max_num_seqs=4, max_num_batched_tokens=8192,
                gpu_memory_utilization=self.config.get('gpu_memory_utilization', .4),
                enable_chunked_prefill=True, enable_sleep_mode=True, enable_prefix_caching=True,
                mm_processor_kwargs={'min_pixels':3136,'max_pixels':50176},
                limit_mm_per_prompt={'image': self.config.get('max_images', 300)})
            self.sleeping = False
        elif self.sleeping is True:
            self.sleeping = None
            self.llm.wake_up()
            self.sleeping = False
        elif self.sleeping is None:
            raise RuntimeError('speaker 显存状态未知')
        load_seconds = time.perf_counter() - started
        prepare_started = time.perf_counter()
        requests = self.prepare_requests(payloads, process_vision_info)
        prepare_seconds = time.perf_counter() - prepare_started
        generation_started = time.perf_counter()
        outputs = self.llm.generate(requests, SamplingParams(temperature=0., max_tokens=8), use_tqdm=False)
        if len(outputs) != len(payloads):
            raise RuntimeError('speaker 返回候选数量不匹配')
        self.last_metrics.update(load_or_wake_seconds=load_seconds, prepare_seconds=prepare_seconds,
                                 generate_seconds=time.perf_counter()-generation_started, cold_start=first_load,
                                 prompt_token_counts=[len(o.prompt_token_ids) if o.prompt_token_ids is not None else None for o in outputs])
        return [o.outputs[0].text for o in outputs]

    def sleep(self):
        if self.llm is None and self.sleeping is None:
            raise RuntimeError('speaker 引擎初始化失败，禁止恢复其他模型')
        if self.llm is not None and self.sleeping is not True:
            self.sleeping = None
            self.llm.reset_prefix_cache()
            self.llm.sleep(level=1)
            self.sleeping = True


"""有界CPU解码/预取及vLLM现有指标读取，不创建GPU上下文。"""
from concurrent.futures import ThreadPoolExecutor
import math


class DecodePipeline:
    def __init__(self,workers=4,chunk_size=16):
        if type(workers) is not int or not 1<=workers<=8:raise ValueError('解码线程需为1..8')
        self.chunk_size=chunk_size
        self.decode_pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='judge-decode')
        self.prefetch_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='judge-prefetch')

    def decode(self,contents,decoder):
        chunks=[contents[i:i+self.chunk_size] for i in range(0,len(contents),self.chunk_size)]
        outputs=list(self.decode_pool.map(decoder,chunks))
        if any(len(a)!=len(b) for a,b in zip(chunks,outputs)):
            raise RuntimeError('并行解码帧数不一致')
        return [image for batch in outputs for image in batch]

    def prefetch(self,function,examples):return self.prefetch_pool.submit(function,examples)

    def close(self):
        self.prefetch_pool.shutdown(wait=True,cancel_futures=True)
        self.decode_pool.shutdown(wait=True,cancel_futures=True)


def engine_metrics(metrics):
    result={}
    for item in metrics:
        name=item.get('name','') if isinstance(item,dict) else getattr(item,'name','')
        value=item.get('value') if isinstance(item,dict) else getattr(item,'value',None)
        if not isinstance(value,(int,float)) or not math.isfinite(value):continue
        if 'num_preemptions' in name:result['preemptions']=result.get('preemptions',0)+value
        elif 'cache_usage_perc' in name:result['kv_usage_ratio']=max(result.get('kv_usage_ratio',0),value)
        elif 'prefix_cache_hits' in name:result['prefix_hits']=result.get('prefix_hits',0)+value
        elif 'prefix_cache_queries' in name:result['prefix_queries']=result.get('prefix_queries',0)+value
    return result


def should_reduce_batch(before,after,batch_size):
    return batch_size>4 and 'preemptions' in before and 'preemptions' in after and after['preemptions']>before['preemptions']

"""在既有作业CPU上只读预取后续真实输入，利用操作系统缓存减少PNG读取等待。"""
import concurrent.futures,json,pathlib,socket,sys,time

def main():
    root=pathlib.Path('/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031')
    out=root/'outputs'/f'formal_{sys.argv[1]}'
    status=json.loads((out/'run_status.json').read_text())
    groups=json.loads((root/'code/groups_manifest.json').read_text())
    start_group=max(0,status.get('encoded_groups',0)+280-64)
    frame_root=pathlib.Path('/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/frames_156')
    paths=list(dict.fromkeys(p for g in groups[start_group:] for p in g['paths']))
    started=time.monotonic();total=0
    def read(path):
        relative=pathlib.PurePosixPath(path)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('路径越界')
        return len((frame_root/relative.with_suffix('.png')).read_bytes())
    print('CPU_PREFETCH_START',socket.gethostname(),len(paths),start_group,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n,size in enumerate(pool.map(read,paths),1):
            total+=size
            if n%10000==0:print('CPU_PREFETCH',n,len(paths),round(time.monotonic()-started,1),flush=True)
    report={'status':'COMPLETE','gpu_count':0,'threads':8,'job_id':sys.argv[1],
        'host':socket.gethostname(),'frames':len(paths),'bytes_read':total,
        'elapsed_seconds':time.monotonic()-started,'source_modified':False}
    (out/'cpu_prefetch_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU_PREFETCH_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()

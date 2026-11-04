"""从已核验的三人缓存抽取十二路异步窗口，并复用原有 CLIP 聚类和相对剪枝。"""
import copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'multi-user'),str(ROOT)]
from training.judge_sft.prepare_real_data import _repo_module
from training.grpo_v3.six_user_binary.data import make_row,validate_row
from training.grpo_v3.six_user_binary.judge import build_examples
from training.grpo_v3.six_user_binary.generation import PROFILE
prep=_repo_module('rlhf_evidence_preprocessing')
cluster_six_user_frame_representatives=prep.cluster_six_user_frame_representatives
clustered_speaker_provider_all_pairs_pruning=prep.clustered_speaker_provider_all_pairs_pruning
build_asker_conditioned_keep_masks=prep.build_asker_conditioned_keep_masks

def read(p): return json.loads(p.read_text(encoding='utf8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def prepare(source,stage_root,plan):
    index={}
    for f in (source/'packets').glob('*/packet.json'):
        p=read(f)
        t=p['time_token']; seconds=int(t[:2])*3600+int(t[2:4])*60+int(t[4:6])
        if seconds in (0,600,1200,1800):
            assert seconds not in index and p['duration_seconds']==600 and p['preprocessing']['user_count']==3
            assert p['selection']['cross_device_sync_verified'] is False
            index[seconds]=(f.parent,p)
    assert sorted(index)==[0,600,1200,1800]
    dataset=stage_root/'dataset'
    ids={}
    for number,groups in enumerate(plan['四包'],1):
        token=f'000000{number:02d}'
        packet_id=f'ARIA12_DAY1_{token}_P{number}'
        dest=dataset/'packets'/packet_id
        dest.mkdir(parents=True,exist_ok=False)
        stage=dest
        ids[number]=packet_id
        sampled_users=[]; embeddings_by_video=[]; source_users=[]; provenance=[]
        for group in groups:
            start=(group-1)*300
            folder,packet=index[start//600*600]
            emb=np.load(folder/'clip_embeddings.f16.npy',allow_pickle=False)
            assert emb.shape==(3,300,512)
            bydir={s['agent_dir']:s for s in packet['source']['users']}
            for ix,user in enumerate(packet['users']):
                assert user['agent_dir']=='user_'+'abc'[ix] and len(user['frames'])==300
                name=str(group)+'ABC'[ix]
                offset=(start%600)//2
                sel=list(range(offset,offset+150,2));assert len(sel)==75
                timeline=copy.deepcopy(bydir[user['agent_dir']])
                seg=timeline['segments']; assert len(seg)==20
                timeline.update(agent_dir=name,agent_id=name,agent_name=name,segments=seg[offset//15:offset//15+10])
                source_users.append(timeline)
                frames=[]
                for local,j in enumerate(sel):
                    old=user['frames'][j]
                    assert old['frame_index']==j and old['timestamp_seconds']==2*j
                    path=(folder/old['path']).resolve()
                    assert path.is_relative_to(folder.resolve()) and path.is_file()
                    frames.append(dict(old,path=str(path),frame_index=local,timestamp_seconds=local*4.,source_segment_index=(local*4)//30))
                provenance.append(dict(window=name,source_user=user['agent_dir'],relative_start_seconds=start,source_packet=folder.name,source_user_index=ix,source_indices=sel))
                sampled_users.append(dict(user=name,clip=timeline,frames=frames))
                embeddings_by_video.append(emb[ix,sel].tolist())
        frames_by_video=[u['frames'] for u in sampled_users]
        user_count=12;sample_interval_seconds=4.
        config=copy.deepcopy(packet['preprocessing']);config.update(user_count=12,duration_seconds=300)
        config['sampling'].update(fps=.25,interval_seconds=4.,expected_frames_per_user=75)
        np.save(stage/'clip_embeddings.f16.npy',np.asarray(embeddings_by_video,dtype=np.float16),allow_pickle=False)
        cluster_config = config["clustering"]
        print(f"rlhf_preprocess packet={packet_id} status=clustering", flush=True)
        clusters_by_video = cluster_six_user_frame_representatives(
            frames_by_video,
            embeddings_by_video,
            user_count=user_count,
            start_seconds=0.0,
            duration_seconds=float(config["duration_seconds"]),
            cluster_count=int(cluster_config["clusters_per_density_window"]),
            split_noncontiguous_clusters=False,
            max_cluster_member_gap_seconds=None,
            cluster_window_seconds=float(cluster_config["density_window_seconds"]),
            temporal_time_weight=float(
                cluster_config["temporal_kmeans_time_weight"]
            ),
            temporal_unit_seconds=float(cluster_config["temporal_unit_seconds"]),
        )
        if any(bool(result.get("split_noncontiguous_clusters")) for result in clusters_by_video):
            raise ValueError("offline clustering unexpectedly split temporal components")

        pruning_config = config["pruning"]
        print(f"rlhf_preprocess packet={packet_id} status=building_masks users={user_count}", flush=True)
        pruning_results = []
        for asker_index in range(user_count):
            pruning_results.append(
                clustered_speaker_provider_all_pairs_pruning(
                    frames_by_video,
                    embeddings_by_video,
                    user_count=user_count,
                    speaker_index=asker_index,
                    start_seconds=0.0,
                    duration_seconds=float(config["duration_seconds"]),
                    sample_interval_seconds=sample_interval_seconds,
                    cluster_count=int(cluster_config["clusters_per_density_window"]),
                    high_similarity_threshold=float(
                        pruning_config["high_similarity_threshold"]
                    ),
                    min_pruned_video_seconds=0.0,
                    pruning_protection_mode="min_percent",
                    min_pruned_video_percent=float(
                        pruning_config["minimum_provider_retention_percent"]
                    ),
                    max_pair_time_difference_seconds=float(
                        pruning_config["max_pair_time_difference_seconds"]
                    ),
                    mutual_nearest_only=False,
                    split_noncontiguous_clusters=False,
                    max_cluster_member_gap_seconds=None,
                    cluster_window_seconds=float(
                        cluster_config["density_window_seconds"]
                    ),
                    precomputed_clusters_by_video=clusters_by_video,
                    temporal_time_weight=float(
                        cluster_config["temporal_kmeans_time_weight"]
                    ),
                    temporal_unit_seconds=float(cluster_config["temporal_unit_seconds"]),
                )
            )
        frame_counts = [len(frames) for frames in frames_by_video]
        masks = build_asker_conditioned_keep_masks(frame_counts, pruning_results, user_count=user_count)
        np.savez_compressed(
            stage / "keep_masks.npz",
            keep_masks=masks,
            frame_counts=np.asarray(frame_counts, dtype=np.int32),
        )


        frames,users=prep._persist_packet_frames(sampled_users,stage)
        clusters=prep._compact_clusters(clusters_by_video,users)
        config['clustering']['global_cluster_count_requested']=clusters['users'][0]['global_cluster_count_requested']
        write(stage/'clusters.json',clusters)
        write(stage/'asker_views.json',prep._compact_asker_views(users,pruning_results,masks))
        new=dict(schema_version=packet['schema_version'],packet_id=packet_id,day='DAY1',time_token=token,
            clip_clock=None,duration_seconds=300,selection=dict(source_time_basis='per_wearer_relative_seconds',cross_device_sync_verified=False,windows=provenance),
            generator_media_mode=packet['generator_media_mode'],preprocessing=config,source={'users':source_users},users=users)
        write(stage/'packet.json',new);prep._write_checksum_manifest(stage);write(stage/'COMPLETE',{'packet_id':packet_id})
        prep.validate_preprocessed_packet(stage,verify_checksums=True)
    rows=[]
    for step in plan['轮换表']:
        for condition in step['条件']:
            n=condition['包'];names=[str(g)+a for g in plan['四包'][n-1] for a in 'ABC']
            asker=names.index(condition['提问窗口'])
            row=make_row(dataset,ids[n],asker,generation_profile=PROFILE)
            assert len(row['required_users'])==12 and row['required_users'][0]==names[asker] and row['generator_frame_counts'][0]==75
            assert len(row['images'])==row['messages'][0]['content'].count('<image>')==sum(row['generator_frame_counts'])
            validate_row(row)
            sample=dict(question='Which object did I ask about?',options=list('ABCDE'),correct='A',answer='A')
            ex=build_examples(dict(row,request_id='cpu-fixture',completion=json.dumps(sample)))
            assert [ex[key].frame_count for key in ('formality','groundedness','combined_answerability')]==[0,900,900]
            if not rows: write(stage_root/'rendered_prompts.json',{'generator':row['messages'][0]['content'],**{k:x.prompt for k,x in ex.items()}})
            rows.append(row)
    assert len(rows)==48 and len({(x['source_packet_id'],x['asker_index']) for x in rows})==48
    data=stage_root/'data'; data.mkdir(exist_ok=True)
    with (data/'train_length_paired.jsonl').open('x',encoding='utf8') as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(stage_root/'condition_definition.json',{'train':[dict(source_packet_id=x['source_packet_id'],speaker_index=x['asker_index'],image_count=len(x['images']),input_view_count=12,duration_seconds=300) for x in rows]})
    return rows
if __name__=='__main__':
    prepare(Path(sys.argv[1]),ROOT,read(ROOT/'schedule.json'))

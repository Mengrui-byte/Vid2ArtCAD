"""Two independently aimed monocular views of an approved motion sequence."""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pipeline.render.stereo_annotation import digest, dump, project
from pipeline.render.mujoco_runtime import set_qpos
from pipeline.render.labels import export_labels


def run(config_path):
    config=json.loads(config_path.read_text());root=config_path.resolve().parent
    source=(root/config['approved_source']).resolve();out=root/config['sample_output']
    if out.exists():raise FileExistsError(out)
    stage=out.with_name(out.name+'.building');stage.mkdir(parents=True)
    shared=['assembly.urdf','motion.npz','timestamps.npy','part_mapping.json','loop_constraints.json',
            'motion_description.json','motion_description.txt','quality.json','scene.xml']
    optional=['kinematic_model.json','joint_runtime.json']
    copied=shared+[f for f in optional if (source/f).exists()]
    source_hashes={f:digest(source/f) for f in copied+['rgb.mp4','camera.json','camera.npz']}
    for f in copied:shutil.copy2(source/f,stage/f)
    (stage/'quality.json').rename(stage/'source_quality.json')
    shutil.copytree(source/'meshes',stage/'meshes')
    shutil.copy2(source/'rgb.mp4',stage/'rgb_cam01.mp4')
    shutil.copy2(source/'camera.json',stage/'camera_cam01.json')
    shutil.copy2(source/'camera.npz',stage/'camera_cam01.npz')
    camera=json.loads((source/'camera.json').read_text());K=np.array(camera['K'])
    T1=np.array(camera['T_world_camera']);W,H=camera['resolution_wh'];fps=camera['fps']
    a=np.load(stage/'motion.npz');poses=a['T_world_link'];ts=a['timestamps'];q=a['q'];N=len(ts)
    assert np.array_equal(ts,np.load(stage/'timestamps.npy'))
    corners=np.array([trimesh.bounds.corners(trimesh.load(stage/'meshes'/f'{n}.obj',process=False).bounds) for n in a['link_order']])
    points=(np.einsum('tpij,pkj->tpki',poses[:,:,:3,:3],corners)+poses[:,:,:3,3][:,:,None,:]).reshape(-1,3)
    target=(points.min(0)+points.max(0))/2
    azimuth=np.deg2rad(config['cam02_azimuth_offset_degrees'])
    back=Rotation.from_rotvec([0,0,azimuth]).apply(-T1[:3,2]);back/=np.linalg.norm(back)
    right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
    Rgl=np.column_stack([right,up,back]);local=np.einsum('nj,jk->nk',points-target,Rgl)
    margin=config['border_fraction'];hx=(W-1)/2-W*margin;hy=(H-1)/2-H*margin
    distance=max(np.max(local[:,2]+np.abs(local[:,0])*K[0,0]/hx),
                 np.max(local[:,2]+np.abs(local[:,1])*K[1,1]/hy))
    eye=target+back*distance;T2=np.eye(4);T2[:3,:3]=Rgl@np.diag([1,-1,-1]);T2[:3,3]=eye
    framing={}
    for name,T in [('cam01',T1),('cam02',T2)]:
        uv,z=project(points,T,K);low,high=uv.min(0),uv.max(0)
        assert z.min()>0 and np.all(low>=20) and np.all(high<np.array([W,H])-20)
        framing[name]=dict(minimum_pixel=low.tolist(),maximum_pixel=high.tolist())
    xml=ET.parse(stage/'scene.xml');world=xml.getroot().find('worldbody')
    original=world.find("camera[@name='annotation']");original.set('name','cam01')
    quat=Rotation.from_matrix(Rgl).as_quat()[[3,0,1,2]]
    ET.SubElement(world,'camera',name='cam02',pos=' '.join(map(str,eye)),quat=' '.join(map(str,quat)),fovy=str(camera['fovy_degrees']))
    ET.indent(xml);xml.write(stage/'scene_multiview.xml',encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str((stage/'scene_multiview.xml').resolve()));data=mujoco.MjData(model)
    runtime=json.loads((stage/'joint_runtime.json').read_text()) if (stage/'joint_runtime.json').exists() else None
    bodies=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,str(n)) for n in a['link_order']]
    camid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'cam02')
    renderer=mujoco.Renderer(model,H,W);fk=0.;rotation=0.;selected=[0,165,285]
    previews={};(stage/'preview').mkdir();writer=imageio.get_writer(stage/'rgb_cam02.mp4',fps=fps,codec='libx264',
        pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-crf','18','-preset','fast'])
    try:
        for i,t in enumerate(ts):
            set_qpos(model,data,q[i],a['joint_order'],runtime);data.time=float(t);mujoco.mj_forward(model,data)
            fk=max(fk,float(np.max(np.abs(data.xpos[bodies]-poses[i,:,:3,3]))))
            rotation=max(rotation,float(np.max(np.abs(data.xmat[bodies].reshape(-1,3,3)-poses[i,:,:3,:3]))))
            actual=np.eye(4);actual[:3,3]=data.cam_xpos[camid];actual[:3,:3]=data.cam_xmat[camid].reshape(3,3)@np.diag([1,-1,-1])
            assert np.allclose(actual,T2,atol=1e-9,rtol=0)
            renderer.update_scene(data,camera='cam02');image=renderer.render();writer.append_data(image)
            if i in selected:imageio.imwrite(stage/'preview'/f'cam02_{i:04d}.png',image)
            if i%60==0:print(f'cam02 frame {i}/{N}',flush=True)
    finally:writer.close();renderer.close()
    assert fk<1e-8 and rotation<1e-8
    videos={}
    for cam in ['cam01','cam02']:
        reader=imageio.get_reader(stage/f'rgb_{cam}.mp4');meta=reader.get_meta_data();n=0
        try:
            for image in reader:
                assert image.shape==(H,W,3)
                if n in selected:
                    previews[cam,n]=np.asarray(image)[::3,::3]
                    imageio.imwrite(stage/'preview'/f'{cam}_{n:04d}.png',image)
                n+=1
        finally:reader.close()
        assert n==N and meta['fps']==fps
        videos[cam]=dict(frame_count=n,fps=fps,resolution_wh=[W,H])
    imageio.imwrite(stage/'preview'/'views_contact_sheet.png',np.concatenate([
        np.concatenate([previews[c,i] for c in ['cam01','cam02']],axis=1) for i in selected],axis=0))
    camera2=dict(resolution_wh=[W,H],fps=fps,K=K.tolist(),T_world_camera=actual.tolist(),
        camera_axes=camera['camera_axes'],world_axes=camera['world_axes'],units='m',distortion=[0,0,0,0,0],
        pixel_center=camera['pixel_center'],fovy_degrees=camera['fovy_degrees'],layout='single_full_assembly_view',
        framing=framing['cam02'],renderer='MuJoCo '+mujoco.__version__,azimuth_offset_degrees=config['cam02_azimuth_offset_degrees'])
    dump(stage/'camera_cam02.json',camera2)
    np.savez_compressed(stage/'camera_cam02.npz',K=np.repeat(K[None],N,axis=0),
        T_world_camera=np.repeat(actual[None],N,axis=0),timestamps=ts)
    separation=float(np.rad2deg(np.arccos(np.clip(np.dot(T1[:3,2],actual[:3,2]),-1,1))))
    back1=-T1[:3,2]; back2=-actual[:3,2]
    azimuth_measured=float(np.rad2deg(np.arccos(np.clip(np.dot(back1[:2],back2[:2]) /
        (np.linalg.norm(back1[:2])*np.linalg.norm(back2[:2])),-1,1))))
    requested=float(np.rad2deg(np.arccos(np.cos(azimuth))))
    if not np.isclose(azimuth_measured,requested,atol=1e-5):
        raise ValueError('Rendered camera azimuth does not match requested separation')
    dump(stage/'views.json',dict(mode='two_independently_aimed_monocular_cameras',rectified=False,
        azimuth_offset_degrees=config['cam02_azimuth_offset_degrees'],optical_axis_angle_degrees=separation,
        measured_azimuth_separation_degrees=azimuth_measured,
        elevation_policy='same downward elevation; 180 degrees refers to opposite azimuths around world vertical',
        T_cam02_cam01=(np.linalg.inv(actual)@T1).tolist(),timestamp_file='timestamps.npy',
        cameras={c:dict(video=f'rgb_{c}.mp4',calibration=f'camera_{c}.json',calibration_arrays=f'camera_{c}.npz') for c in ['cam01','cam02']}))
    unchanged=all(digest(source/f)==h for f,h in source_hashes.items());assert unchanged
    assert digest(stage/'rgb_cam01.mp4')==source_hashes['rgb.mp4']
    dump(stage/'quality.json',dict(status='multiview_replay_passed',videos=videos,framing=framing,
        available_tasks=['rgb', 'motion', 'kinematics', 'depth', 'part_id', 'tracks'],
        measured_azimuth_separation_degrees=azimuth_measured,
        pose_error_m=fk,rotation_error=rotation,cam01_byte_identical_to_approved=True,source_unchanged=unchanged,
        source_limitations=json.loads((stage/'source_quality.json').read_text())['limitations']))
    label_report = export_labels(stage)
    dump(stage/'provenance.json',dict(config=config,source_hashes=source_hashes,exporter_sha256=digest(Path(__file__)),
        renderer='MuJoCo '+mujoco.__version__,states='synthetic_gt',camera_placement='pipeline_policy'))
    shutil.copy2(Path(__file__),stage/'exporter_snapshot.py')
    dump(stage/'sample.json',dict(mode='multiview_monocular',sample_id=out.name,views='views.json',
        urdf='assembly.urdf',meshes='meshes/',motion='motion.npz',timestamps='timestamps.npy',
        scene='scene_multiview.xml',closed_loops='loop_constraints.json',kinematic_model='kinematic_model.json',
        joint_runtime='joint_runtime.json',description='motion_description.json',quality='quality.json',
        labels='labels/quality.json',label_report=label_report))
    dump(stage/'checksums.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    stage.rename(out)
    dump(root/'active_workflow.json',dict(mode='multiview_monocular',config=config_path.name,sample=str(out.relative_to(root)),
        azimuth_offset_degrees=config['cam02_azimuth_offset_degrees'],
        supersedes='Previous parallel-stereo and 90-degree outputs are historical versions'))
    with (root/'manifest.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(sample_id=out.name,path=str(out.relative_to(root)),mode='multiview_monocular',status='multiview_replay_passed'))+'\n')
    print('Completed:',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)

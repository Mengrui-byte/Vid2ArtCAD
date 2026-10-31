"""Replay an approved immutable assembly trajectory through two calibrated cameras."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def project(points, T, K):
    pc = np.einsum('nj,jk->nk', points-T[:3, 3], T[:3, :3])
    uv = pc[:, :2] / pc[:, 2, None]
    uv = uv * [K[0, 0], K[1, 1]] + K[:2, 2]
    return uv, pc[:, 2]


def run(config_path):
    config = json.loads(config_path.read_text())
    root = config_path.resolve().parent
    source = (root / config['approved_source']).resolve()
    target = root / config['sample_output']
    if target.exists():
        raise FileExistsError(f'Output exists; choose a new version: {target}')
    stage = target.with_name(target.name + '.building')
    stage.mkdir(parents=True, exist_ok=False)
    files = ['assembly.urdf', 'scene.xml', 'motion.npz', 'timestamps.npy',
             'camera.json', 'camera.npz', 'part_mapping.json', 'loop_constraints.json',
             'motion_description.json', 'motion_description.txt', 'quality.json']
    files += [str(p.relative_to(source)) for p in sorted((source / 'meshes').glob('*.obj'))]
    source_hashes = {f: digest(source / f) for f in files + ['rgb.mp4']}
    frozen = root / 'approved_reference'
    if frozen.exists():
        for f,h in source_hashes.items():
            if digest(frozen/f) != h:
                raise ValueError('Approved reference changed; explicit re-versioning required: ' + f)
    else:
        frozen.mkdir()
        for f in files + ['rgb.mp4']:
            (frozen/f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/f, frozen/f)
    for f in files:
        (stage/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen/f, stage/f)
    # Preserve single-camera calibration as provenance, publish stereo calibration separately.
    (stage/'camera.json').rename(stage/'reference_camera.json')
    (stage/'camera.npz').rename(stage/'reference_camera.npz')
    (stage/'quality.json').rename(stage/'source_quality.json')
    camera = json.loads((stage/'reference_camera.json').read_text())
    trajectory = np.load(stage/'motion.npz', allow_pickle=False)
    ts = trajectory['timestamps']; q = trajectory['q']; poses = trajectory['T_world_link']
    assert np.array_equal(ts, np.load(stage/'timestamps.npy'))
    assert np.all(np.isfinite(q)) and np.all(np.diff(ts) > 0)
    fps = camera['fps']; count = len(ts); width,height = camera['resolution_wh']
    assert np.allclose(np.diff(ts), 1/fps)
    base = float(config['baseline_m'])
    if not 0 < base < 1:
        raise ValueError('Baseline must be in meters, greater than zero and below one meter')
    K = np.array(camera['K']); Tleft = np.array(camera['T_world_camera'])
    Tright = Tleft.copy(); Tright[:3,3] += Tleft[:3,0]*base
    expected = {'left':Tleft, 'right':Tright}
    bounds = []
    for name in trajectory['link_order']:
        mesh = trimesh.load(stage/'meshes'/f'{name}.obj', process=False)
        bounds.append(trimesh.bounds.corners(mesh.bounds))
    bounds = np.array(bounds)
    envelope = np.einsum('tpij,pkj->tpki', poses[:,:,:3,:3], bounds)+poses[:,:,:3,3][:,:,None,:]
    envelope = envelope.reshape(-1,3)
    framing = {}
    for side,T in expected.items():
        uv,z = project(envelope,T,K)
        low,high = uv.min(0),uv.max(0)
        margin = min(*low, width-1-high[0], height-1-high[1])
        if z.min() <= 0 or margin < config['minimum_border_px']:
            raise ValueError(f'{side} camera framing fails: margin={margin}. Re-version camera configuration.')
        framing[side] = dict(minimum_pixel=low.tolist(),maximum_pixel=high.tolist(),minimum_margin_px=float(margin))
    xml = ET.parse(stage/'scene.xml'); world = xml.getroot().find('worldbody')
    original = world.find("camera[@name='annotation']")
    if original is None:
        raise ValueError('Approved annotation camera missing')
    original.set('name','left')
    right = copy.deepcopy(original); right.set('name','right')
    right.set('pos',' '.join(f'{x:.16g}' for x in Tright[:3,3]));world.append(right)
    ET.indent(xml);xml.write(stage/'scene_stereo.xml',encoding='utf-8',xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str((stage/'scene_stereo.xml').resolve()))
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(n))] for n in trajectory['joint_order']]
    bodyids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,str(n)) for n in trajectory['link_order']]
    assert min(bodyids)>=0 and len(set(qadr))==q.shape[1]
    renderer = mujoco.Renderer(model,height,width)
    writers={}; actual={}; fk_error=0.; rotation_error=0.; previews={}
    try:
        for side in expected:
            writers[side]=imageio.get_writer(stage/f'rgb_{side}.mp4',fps=fps,codec='libx264',
                pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-crf','18','-preset','fast'])
        for i,t in enumerate(ts):
            data.qpos[qadr]=q[i];data.time=float(t);mujoco.mj_forward(model,data)
            fk_error=max(fk_error,float(np.max(np.abs(data.xpos[bodyids]-poses[i,:,:3,3]))))
            rotation_error=max(rotation_error,float(np.max(np.abs(data.xmat[bodyids].reshape(-1,3,3)-poses[i,:,:3,:3]))))
            for side in expected:
                camid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,side)
                T=np.eye(4);T[:3,:3]=data.cam_xmat[camid].reshape(3,3)@np.diag([1,-1,-1]);T[:3,3]=data.cam_xpos[camid]
                assert np.allclose(T,expected[side],atol=1e-9,rtol=0)
                fy=height*.5/np.tan(np.deg2rad(model.cam_fovy[camid])*.5)
                assert np.allclose([fy,fy],[K[0,0],K[1,1]],atol=1e-8,rtol=0)
                actual[side]=T
                renderer.update_scene(data,camera=side);im=renderer.render();writers[side].append_data(im)
                if i in (0,165,285):
                    (stage/'preview').mkdir(exist_ok=True)
                    imageio.imwrite(stage/'preview'/f'{side}_{i:04d}.png',im)
                    previews[(side,i)]=im[::3,::3]
            if i%60==0:print(f'stereo frame {i}/{count}',flush=True)
    finally:
        for writer in writers.values():writer.close()
        renderer.close()
    if fk_error>1e-8 or rotation_error>1e-8:
        raise ValueError('Replay differs from approved trajectory')
    transform_rl=np.linalg.inv(actual['right'])@actual['left']
    assert np.allclose(transform_rl[:3,:3],np.eye(3),atol=1e-9)
    assert np.allclose(transform_rl[:3,3],[-base,0,0],atol=1e-9)
    t=transform_rl[:3,3]; skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    essential=skew@transform_rl[:3,:3]; fundamental=np.linalg.inv(K).T@essential@np.linalg.inv(K)
    pl,z=project(envelope,actual['left'],K);pr,_=project(envelope,actual['right'],K)
    vertical_error=float(np.abs(pl[:,1]-pr[:,1]).max())
    disparity_error=float(np.abs(pl[:,0]-pr[:,0]-K[0,0]*base/z).max())
    assert max(vertical_error,disparity_error)<1e-6
    decoded={}
    for side in expected:
        reader=imageio.get_reader(stage/f'rgb_{side}.mp4'); n=0; first=None;middle=None
        try:
            meta=reader.get_meta_data()
            for frame in reader:
                assert frame.shape==(height,width,3)
                if n==0:first=np.asarray(frame).astype(float)
                if n==165:middle=np.asarray(frame).astype(float)
                n+=1
        finally:reader.close()
        assert n==count and abs(meta['fps']-fps)<1e-8
        decoded[side]=dict(frame_count=n,resolution_wh=[width,height],fps=meta['fps'],
                          first_to_motion_frame_mean_abs_difference=float(np.abs(first-middle).mean()))
    imageio.imwrite(stage/'preview'/'stereo_contact_sheet.png',np.concatenate([
        np.concatenate([previews[(side,i)] for side in ['left','right']],axis=1) for i in [0,165,285]],axis=0))
    calibration=dict(schema_version='1.0',resolution_wh=[width,height],fps=fps,frame_count=count,
        duration_s=count/fps,baseline_m=base,baseline_source='pipeline_policy',
        transform_convention='T_a_b maps column vectors from b coordinates to a coordinates',
        axes='camera x right, y down, z forward; world z up; length m; angle rad',
        pixel_centers='Top-left pixel center (0,0); principal point ((W-1)/2,(H-1)/2)',
        rectified=True,distortion_model='none',synchronization='same q and timestamp; no integration between eyes',
        T_right_left=transform_rl.tolist(),R_right_left=transform_rl[:3,:3].tolist(),t_right_left=t.tolist(),
        E=essential.tolist(),F=fundamental.tolist(),
        P_left=(K@np.eye(4)[:3]).tolist(),P_right=(K@transform_rl[:3]).tolist(),
        disparity_definition='d=u_left-u_right; z_left=fx*baseline/d for corresponding visible points',framing=framing)
    for side in expected:
        calibration[side]=dict(K=K.tolist(),T_world_camera=actual[side].tolist(),distortion=[0,0,0,0,0],video=f'rgb_{side}.mp4')
    dump(stage/'calibration.json',calibration)
    np.savez_compressed(stage/'calibration.npz',timestamps=ts,K_left=np.repeat(K[None],count,axis=0),
        K_right=np.repeat(K[None],count,axis=0),T_world_left=np.repeat(actual['left'][None],count,axis=0),
        T_world_right=np.repeat(actual['right'][None],count,axis=0),T_right_left=transform_rl,E=essential,F=fundamental,baseline_m=base)
    source_quality=json.loads((stage/'source_quality.json').read_text())
    dump(stage/'quality.json',dict(status='stereo_replay_passed',scope='Synchronized synthetic replay and analytic calibration; not physical validation',
        videos=decoded,pose_replay_error_m=fk_error,rotation_matrix_replay_error=rotation_error,
        analytic_vertical_disparity_max_px=vertical_error,analytic_disparity_formula_max_error_px=disparity_error,
        whole_motion_frustum_checks=framing,source_limitations=source_quality['limitations'],
        source_unchanged=all(digest(source/f)==h for f,h in source_hashes.items())))
    provenance=dict(reference='approved_reference/rgb.mp4',source_hashes=source_hashes,
        config=config,exporter_sha256=digest(Path(__file__)),python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ['mujoco','numpy','scipy','trimesh','imageio','imageio-ffmpeg']},
        source_label='dataset_reference',rendered_state_label='synthetic_gt',camera_rig_label='pipeline_policy')
    dump(stage/'provenance.json',provenance)
    shutil.copy2(Path(__file__),stage/'exporter_snapshot.py')
    dump(stage/'sample.json',dict(schema_version='1.0',sample_id=target.name,
        videos={'left':'rgb_left.mp4','right':'rgb_right.mp4'},calibration='calibration.json',
        calibration_arrays='calibration.npz',motion='motion.npz',timestamps='timestamps.npy',urdf='assembly.urdf',
        scene='scene_stereo.xml',loop_constraints='loop_constraints.json',part_mapping='part_mapping.json',
        description='motion_description.json',quality='quality.json',provenance='provenance.json'))
    manifest={str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    dump(stage/'checksums.json',manifest)
    target.parent.mkdir(parents=True,exist_ok=True);stage.rename(target)
    dump(root/'approved_reference_lock.json',dict(source=str(source),sha256=source_hashes))
    with (root/'manifest.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(sample_id=target.name,path=str(target.relative_to(root)),status='stereo_replay_passed'))+'\n')
    print('Completed:',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)

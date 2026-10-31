"""Occurrence-based lamp reconstruction with explicit closed-loop constraints."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh

ASSET = Path('/tmp/vid2artcad_candidate/23226_8d1b5098')
OUTPUT = Path('output/samples/23226_8d1b5098_verified')


def vec(v):
    return np.array([v[k] for k in 'xyz'], dtype=float)


def transform(t):
    T = np.eye(4)
    T[:3, :3] = np.column_stack([vec(t[k + '_axis']) for k in 'xyz'])
    T[:3, 3] = vec(t['origin']) * .01
    return T


def rotate(a, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(a * q).as_matrix()
    return T


def numbers(a):
    return ' '.join(f'{float(x):.12g}' for x in np.asarray(a).ravel())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


class Lamp:
    def __init__(self, asset=ASSET):
        self.asset = asset
        self.source = json.loads((asset / 'assembly.json').read_text())
        self.ids = list(self.source['occurrences'])
        self.occ = list(self.source['occurrences'].values())
        index = {oid: i for i, oid in enumerate(self.ids)}
        self.Tref = np.array([transform(o['transform']) for o in self.occ])
        # World z-up from source y-up; this rotation is a declared convention.
        self.C = np.eye(4)
        self.C[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.Tref = self.C @ self.Tref
        self.edges = []
        for jid, j in self.source['joints'].items():
            a, b = index[j['occurrence_one']], index[j['occurrence_two']]
            g, h = j['geometry_or_origin_one'], j['geometry_or_origin_two']
            p = self.C[:3, :3] @ (vec(g['origin']) * .01)
            axis = self.Tref[a, :3, :3] @ vec(g['primary_axis_vector'])
            other_axis = self.Tref[b, :3, :3] @ vec(h['primary_axis_vector'])
            axis /= np.linalg.norm(axis)
            if abs(np.dot(axis, other_axis)) < .999:
                raise ValueError(f'Inconsistent source axes: {j["name"]}')
            self.edges.append(dict(id=jid, name=j['name'], a=a, b=b, pivot=p, axis=axis))
        # The supplied graph omits the bracket-to-ground connection. Preserve
        # both components and document this fixed mounting as a rendering policy.
        self.mount = (0, 1)
        adjacency = {i: [] for i in range(len(self.ids))}
        for k, e in enumerate(self.edges):
            adjacency[e['a']].append((e['b'], k))
            adjacency[e['b']].append((e['a'], k))
        self.Fref = self.Tref.copy()
        self.tree = []
        seen = {0, 1}
        queue = deque([1])
        used = set()
        while queue:
            p = queue.popleft()
            for c, k in adjacency[p]:
                if c in seen:
                    continue
                seen.add(c); used.add(k); queue.append(c)
                e = self.edges[k]
                J = np.eye(4); J[:3, 3] = e['pivot']
                self.Fref[c] = J
                self.tree.append(dict(parent=p, child=c, edge=k,
                                      A=np.linalg.inv(self.Fref[p]) @ J))
        if len(seen) != len(self.ids):
            raise ValueError('Additional disconnected source components')
        self.loops = [k for k in range(len(self.edges)) if k not in used]
        self.B = np.linalg.inv(self.Fref) @ self.Tref
        self.local_p = []
        self.local_axis = []
        for e in self.edges:
            self.local_p.append([np.linalg.inv(self.Fref[i]) @ np.r_[e['pivot'], 1] for i in [e['a'], e['b']]])
            self.local_axis.append([self.Fref[i, :3, :3].T @ e['axis'] for i in [e['a'], e['b']]])
        self.variables = [k for k, t in enumerate(self.tree)
                          if t['parent'] < 20 and t['child'] < 20
                          and abs(self.edges[t['edge']]['axis'][0]) > .99]
        self.by_name = {self.edges[t['edge']]['name']: k for k, t in enumerate(self.tree)}

    def fk(self, q):
        F = self.Fref.copy()
        for k, t in enumerate(self.tree):
            F[t['child']] = F[t['parent']] @ t['A'] @ rotate(self.edges[t['edge']]['axis'], q[k])
        return F

    def loop_residual(self, F):
        residual = []
        for k in self.loops:
            e = self.edges[k]; a, b = e['a'], e['b']
            p, r = self.local_p[k]
            u, v = self.local_axis[k]
            residual.extend((F[a] @ p - F[b] @ r)[:3])
            residual.extend(.1 * np.cross(F[a, :3, :3] @ u, F[b, :3, :3] @ v))
        return np.array(residual)

    def solve(self, drive, prior=None):
        q = np.zeros(len(self.tree))
        x0 = np.zeros(len(self.variables)) if prior is None else prior[self.variables]
        for name, value in drive.items():
            if self.by_name[name] not in self.variables:
                q[self.by_name[name]] = value
        def residual(x):
            q[self.variables] = x
            constraints = self.loop_residual(self.fk(q))
            drivers = [q[self.by_name[n]] - v for n, v in drive.items()]
            return np.r_[constraints * 100, drivers, 1e-7 * x]
        sol = least_squares(residual, x0, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=150)
        q[self.variables] = sol.x
        F = self.fk(q)
        error = float(np.max(np.abs(self.loop_residual(F))))
        if error > 1e-6:
            raise ValueError(f'Closed loop error {error}')
        return q.copy(), F, error

    def export_geometry(self, out):
        folder = out / 'meshes'; folder.mkdir(parents=True, exist_ok=True)
        geometries = {}
        for i, o in enumerate(self.occ):
            meshes = []
            for bid in o.get('bodies', {}):
                m = trimesh.load(self.asset / (bid + '.obj'), process=False)
                m.vertices *= .01
                meshes.append(m)
            m = trimesh.util.concatenate(meshes)
            m.apply_transform(self.B[i])
            m.export(folder / f'link_{i:02d}.obj')
            geometries[i] = m
        return geometries


def origin_xml(node, T):
    ET.SubElement(node, 'origin', xyz=numbers(T[:3, 3]),
                  rpy=numbers(Rotation.from_matrix(T[:3, :3]).as_euler('xyz')))


def write_xml(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def export_models(lamp, out):
    import mujoco
    meshes = lamp.export_geometry(out)
    robot = ET.Element('robot', name='closed_loop_desk_lamp')
    ET.SubElement(robot, 'link', name='world')
    mj = ET.Element('mujoco', model='closed_loop_desk_lamp')
    ET.SubElement(mj, 'compiler', angle='radian', meshdir='meshes')
    ET.SubElement(mj, 'option', gravity='0 0 0')
    vis = ET.SubElement(mj, 'visual')
    ET.SubElement(vis, 'global', offwidth='1920', offheight='1080')
    ET.SubElement(vis, 'quality', shadowsize='2048', offsamples='4')
    ET.SubElement(vis, 'headlight', ambient='.4 .4 .4', diffuse='.6 .6 .6', specular='.2 .2 .2')
    assets = ET.SubElement(mj, 'asset')
    ET.SubElement(assets, 'texture', name='studio_background', type='skybox',
                  builtin='gradient', rgb1='.78 .81 .84', rgb2='.92 .93 .94',
                  width='256', height='1536')
    world = ET.SubElement(mj, 'worldbody')
    ET.SubElement(world, 'light', pos='1 -1 2', dir='-1 1 -2', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .01', rgba='.92 .93 .94 1', contype='0', conaffinity='0')
    bodies = {}
    specs = [(None, 0, lamp.Fref[0], None),
             (0, 1, np.linalg.inv(lamp.Fref[0]) @ lamp.Fref[1], None)]
    specs += [(t['parent'], t['child'], t['A'], k) for k,t in enumerate(lamp.tree)]
    for p,c,T,k in specs:
        name = f'link_{c:02d}'
        color = [.18,.20,.22,1] if c == 0 else ([.93,.57,.12,1] if c in [2,3,4,7,19] else [.38,.47,.51,1])
        if c in [16,18]: color = [.16,.52,.56,1]
        link = ET.SubElement(robot, 'link', name=name)
        visual = ET.SubElement(link, 'visual'); origin_xml(visual,np.eye(4))
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry,'mesh',filename=f'meshes/{name}.obj')
        material = ET.SubElement(visual,'material',name=f'material_{c}')
        ET.SubElement(material,'color',rgba=numbers(color))
        joint_name = 'world_mount' if c == 0 else ('bracket_mount_policy' if c == 1 else lamp.edges[lamp.tree[k]['edge']]['name'])
        joint = ET.SubElement(robot,'joint', name=joint_name, type='fixed' if k is None else 'continuous')
        ET.SubElement(joint,'parent',link='world' if p is None else f'link_{p:02d}')
        ET.SubElement(joint,'child',link=name); origin_xml(joint,T)
        if k is not None:
            ET.SubElement(joint,'axis',xyz=numbers(lamp.edges[lamp.tree[k]['edge']]['axis']))
        ET.SubElement(assets,'mesh',name=name,file=f'{name}.obj')
        quat = Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
        body = ET.SubElement(world if p is None else bodies[p], 'body',name=name,pos=numbers(T[:3,3]),quat=numbers(quat))
        bodies[c] = body
        if k is not None:
            ET.SubElement(body,'joint',name=joint_name,type='hinge',axis=numbers(lamp.edges[lamp.tree[k]['edge']]['axis']))
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba=numbers(color),contype='0',conaffinity='0',mass='.01')
    constraints=[]
    for k in lamp.loops:
        e=lamp.edges[k]
        constraints.append(dict(source_joint_id=e['id'],name=e['name'],
                                parent=f'link_{e["a"]:02d}',child=f'link_{e["b"]:02d}',
                                anchor_parent=lamp.local_p[k][0][:3].tolist(),anchor_child=lamp.local_p[k][1][:3].tolist(),
                                axis_parent=lamp.local_axis[k][0].tolist(),axis_child=lamp.local_axis[k][1].tolist()))
    dump(out/'loop_constraints.json',dict(constraints=constraints,
        note='URDF stores a spanning tree. These seven edges remain enforced by the trajectory solver; do not independently actuate tree joints.',
        mounting_policy='Source base is grounded but disconnected from bracket; bracket is fixed to base at source reference pose for this clip.'))
    write_xml(out/'assembly.urdf',robot)
    write_xml(out/'scene.xml',mj)
    model=mujoco.MjModel.from_xml_path(str((out/'scene.xml').resolve()))
    return meshes,model


def motion(lamp, frames=360):
    # 12-second scripted manipulation; all angles relative to imported pose.
    times=np.arange(frames)/30
    knots=np.array([0,1,3.5,5.5,7.5,9.5,11,12])
    values=np.array([[0,0,0],[0,0,0],[.26,0,0],[.26,-.22,0],
                     [.26,-.22,.22],[.26,-.22,-.12],[0,0,0],[0,0,0]])
    Q=[]; F=[]; errors=[]; prior=None
    for t in times:
        s=min(np.searchsorted(knots,t,side='right')-1,len(knots)-2)
        u=(t-knots[s])/(knots[s+1]-knots[s]); u=6*u**5-15*u**4+10*u**3
        target=values[s]*(1-u)+values[s+1]*u
        q,f,e=lamp.solve(dict(zip(['Rev2','Rev7','Rev28'],target)),prior)
        prior=q;Q.append(q);F.append(f);errors.append(e)
    return times,np.array(Q),np.array(F),np.array(errors)


def camera_fit(meshes, frames, width, height):
    # Fit the complete motion envelope using a fixed oblique camera.
    points=[]
    for frame in frames:
        for i,m in meshes.items():
            points.append(trimesh.transform_points(trimesh.bounds.corners(m.bounds),frame[i]))
    points=np.vstack(points)
    target=(points.min(0)+points.max(0))/2
    direction=np.array([1.,-1.6,.65]);direction/=np.linalg.norm(direction)
    back=direction;right=np.cross([0,0,1],back);right/=np.linalg.norm(right)
    up=np.cross(back,right); R=np.column_stack([right,up,back])
    fy=.5*height/np.tan(np.deg2rad(38)/2)
    K=np.array([[fy,0,(width-1)/2],[0,fy,(height-1)/2],[0,0,1.]])
    local=np.einsum('ki,ij->kj',points-target,R)
    # Four percent image margin on each edge, over every exported pose.
    distance=max(np.max(local[:,2]+np.abs(local[:,0])*fy/(width*.46)),
                 np.max(local[:,2]+np.abs(local[:,1])*fy/(height*.46)))
    eye=target+back*distance
    T=np.eye(4);T[:3,:3]=R@np.diag([1,-1,-1]);T[:3,3]=eye
    camera_points=np.einsum('ki,ij->kj',points-eye,T[:3,:3])
    pixels=camera_points[:,:2]/camera_points[:,2,None]*fy+K[:2,2]
    lo,hi=pixels.min(0),pixels.max(0)
    assert camera_points[:,2].min()>0
    assert np.all(lo>=0) and np.all(hi<[width,height])
    framing=dict(pose_count=len(frames),minimum_pixel=lo.tolist(),maximum_pixel=hi.tolist(),
                 camera_distance_to_target_m=float(distance),edge_margin_policy=.04,
                 note='Projected bounding-box corners for every part at every frame; all geometry contained.')
    return K,T,R,eye,framing


def export_sample(out=OUTPUT, preview=False):
    import imageio.v2 as imageio
    import mujoco
    out.mkdir(parents=True,exist_ok=True)
    lamp=Lamp(); meshes,model=export_models(lamp,out)
    ts,Q,F,errors=motion(lamp)
    names=[lamp.edges[t['edge']]['name'] for t in lamp.tree]
    np.savez_compressed(out/'motion.npz',timestamps=ts,q=Q,q_valid=np.ones_like(Q,dtype=bool),
                        T_world_link=F,T_world_occurrence=F@lamp.B,joint_order=np.array(names),link_order=np.array([f'link_{i:02d}' for i in range(len(lamp.ids))]))
    np.save(out/'timestamps.npy',ts)
    width,height=1920,1080
    K,T,R,eye,framing=camera_fit(meshes,F,width,height)
    dump(out/'framing_check.json',framing)
    xml=ET.parse(out/'scene.xml');world=xml.getroot().find('worldbody')
    quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    ET.SubElement(world,'camera',name='annotation',pos=numbers(eye),quat=numbers(quat),fovy='38')
    write_xml(out/'scene.xml',xml.getroot())
    model=mujoco.MjModel.from_xml_path(str((out/'scene.xml').resolve())); data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height,width)
    qadr=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]
    bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'link_{i:02d}') for i in range(len(lamp.ids))]
    camid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'annotation')
    actual_T=None;fk_error=0.;images=[]
    writer=None if preview else imageio.get_writer(out/'rgb.mp4',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-crf','18','-preset','fast'])
    chosen=[0,105,165,225,285,359]
    try:
        for f in (chosen if preview else range(len(ts))):
            data.qpos[qadr]=Q[f];mujoco.mj_forward(model,data)
            fk_error=max(fk_error,float(np.max(np.abs(data.xpos[bids]-F[f,:,:3,3]))),float(np.max(np.abs(data.xmat[bids].reshape(-1,3,3)-F[f,:,:3,:3]))))
            renderer.update_scene(data,camera='annotation')
            image=renderer.render()
            if actual_T is None:
                actual_T=np.eye(4);actual_T[:3,:3]=data.cam_xmat[camid].reshape(3,3)@np.diag([1,-1,-1]);actual_T[:3,3]=data.cam_xpos[camid]
                print('camera match',np.max(np.abs(actual_T-T)), 'image std',image.std(),flush=True)
            if writer is not None:writer.append_data(image)
            if f in chosen:
                imageio.imwrite(out/f'frame_{f:03d}.png',image);images.append(image[::3,::3])
            if f%60==0:print('render frame',f,flush=True)
    finally:
        if writer is not None:writer.close()
        renderer.close()
    imageio.imwrite(out/'contact_sheet.png',np.concatenate([np.concatenate(images[:3],axis=1),np.concatenate(images[3:],axis=1)],axis=0))
    np.savez(out/'camera.npz',K=np.repeat(K[None],len(ts),axis=0),T_world_camera=np.repeat(actual_T[None],len(ts),axis=0),timestamps=ts)
    dump(out/'camera.json',dict(resolution_wh=[width,height],fps=30,K=K.tolist(),T_world_camera=actual_T.tolist(),
                               camera_axes='x_right_y_down_z_forward',world_axes='z_up',units='m',distortion=[0,0,0,0,0],
                               pixel_center='integer pixel centers; top-left center (0,0)',fovy_degrees=38,
                               layout='single_full_assembly_view',framing=framing,renderer='MuJoCo '+mujoco.__version__))
    dump(out/'part_mapping.json',dict(parts=[dict(link=f'link_{i:02d}',occurrence_id=lamp.ids[i],name=o['name'],body_ids=list(o.get('bodies',{}))) for i,o in enumerate(lamp.occ)],
                                    joint_order=names,source_to_canonical=lamp.C.tolist(),length_scale=.01))
    text='台灯底座保持固定。0–1 秒保持初始姿态；1–3.5 秒下灯臂平滑调整，平行连杆随动；3.5–5.5 秒上灯臂展开，灯头随整条机构移动；5.5–9.5 秒灯头安装关节往复调节；9.5–11 秒返回初始姿态，11–12 秒停留。角度均相对导入姿态。'
    (out/'motion_description.txt').write_text(text+'\n')
    dump(out/'motion_description.json',dict(description_cn=text,duration_s=12,frame_count=360,fps=30,
         drivers=['Rev2','Rev7','Rev28'],peak_commands_rad=[.26,-.22,.22],source='pipeline_policy',
         source_joint_definitions='dataset_reference',state_source='synthetic_gt',closed_loop_constraints=7,
         mechanical_limits='Canonical source zero unresolved; source bounds are not used as verified canonical limits.'))
    dump(out/'quality.json',dict(status='kinematic_preview',frame_count=360,instances=59,unique_body_geometries=29,
         source_revolute_constraints=64,urdf_revolute_joints=57,loop_edges=7,
         maximum_closure_residual_m=float(errors.max()),mujoco_vs_fk_max_error=fk_error,
         limitations=['Fixed bracket-to-base mount is an explicit policy because the source graph omits this edge.',
                      'Source spring geometries are rigid meshes, not a deformable spring simulation.',
                      'Collision-free continuous motion and real mechanical limits have not been certified.']))
    dump(out/'sample.json',dict(video='rgb.mp4',urdf='assembly.urdf',native_scene='scene.xml',motion='motion.npz',
          camera='camera.npz',description='motion_description.json',closed_loops='loop_constraints.json',quality='quality.json'))
    print('complete',out,'closure',errors.max(),'fk',fk_error,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if not args.probe:
        export_sample(preview=args.preview)
        return
    lamp = Lamp()
    print('instances', len(lamp.ids), 'tree joints', len(lamp.tree), 'closure edges',len(lamp.loops), flush=True)
    print('variables', [lamp.edges[lamp.tree[k]['edge']]['name'] for k in lamp.variables],flush=True)
    prior = None
    for value in [0, .05, .1, .2, -.1]:
        q,F,err = lamp.solve({'Rev2':value, 'Rev7': -.5*value},prior)
        prior=q
        print('drive',value,'closure error m',err,'max movement m',np.max(np.linalg.norm(F[:,:3,3]-lamp.Fref[:,:3,3],axis=1)),flush=True)


if __name__ == '__main__':
    main()

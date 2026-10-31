"""Batch source-grounded tree mechanisms, then reuse the approved multiview exporter."""
from __future__ import annotations
import argparse
import json
import shutil
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from pipeline.render.verified_lamp import transform, vec, rotate, origin_xml, write_xml, numbers, camera_fit
from pipeline.render.stereo_annotation import dump, digest
from pipeline.render.multiview_annotation import run as multiview
from pipeline.kinematics.solver import JointSpec, KinematicSolver
from pipeline.kinematics.source import build_source_model
from pipeline.render.mujoco_runtime import set_qpos

ROOT=Path('annoation')
CONFIGS=[
 dict(id='22218_2f85f6ae',title='滑台与手柄组件',root=1,drive={'Slider8':.008,'Rev2':.5},
      description='背板固定，滑台及其刚性连接件沿导轨小幅移动；手柄绕自身轴线往复转动。两种运动分别激励后组合，再复位。'),
 dict(id='21805_2ba39d00',title='双万向节传动轴',root=1,drive={'Rev1':.18,'Rev4':-.22,'Rev3':-.18,'Rev6':.22},
      description='中央连接轴保持固定，两端万向节绕各自交叉轴依次小幅摆动，叉架和端部轴随动，再组合摆动并复位。'),
 dict(id='20259_de1ab632',title='滑动旋转连杆',root=2,drive={'Rev2':.55,'Slider1':.012},
      description='长轴固定，连杆绕长轴转动并带动短轴位置改变；短轴沿连杆孔轴线往复滑动。先分别运动，再组合运动并复位。'),
 dict(id='20341_6a1588fb',title='鼠标滚轮组件',root=1,drive={'Rev1':2.4},
      description='鼠标外壳固定，滚轮绕原始轴线正向转动、停留、反向转动并回到初始姿态。'),
 dict(id='22043_041e1b7c',title='螺旋支撑旋转组件',root=0,drive={'Rev2':.55,'Rev6':.3},
      description='底座固定，螺杆及其连接零件小幅往复转动，横向手柄随后摆动并复位。')]

def solve_commanded_trajectory(solver, joint_order, targets, commanded_indices, timestamps,
                               tolerance=1e-7):
 targets=np.asarray(targets,dtype=float);timestamps=np.asarray(timestamps,dtype=float)
 if targets.ndim!=2 or targets.shape!=(len(timestamps),len(joint_order)):
  raise ValueError('trajectory target shape does not match timestamps and tree joint order')
 indices=sorted(set(int(index) for index in commanded_indices))
 if any(index<0 or index>=len(joint_order) for index in indices):
  raise ValueError('commanded coordinate index is outside tree joint order')
 warm=np.zeros(solver.dof_count);Q=[];Qall=[];F=[]
 for frame,target in enumerate(targets):
  prescribed={joint_order[index]:float(target[index]) for index in indices}
  result=solver.solve(warm,prescribed=prescribed)
  residual=float(np.linalg.norm(result.residual))
  if not result.success or not np.isfinite(residual) or residual>tolerance:
   raise ValueError(f'loop closure failed during trajectory at frame {frame}: {residual}')
  warm=result.q
  Q.append(warm[:len(joint_order)])
  Qall.append(warm.copy())
  F.append(np.asarray([result.transforms[link] for link in solver.link_ids]))
 return np.asarray(Q),np.asarray(F),np.asarray(Qall)

class Assembly:
 def __init__(self,cfg):
  self.cfg=cfg;self.path=Path(cfg.get('asset_root','data/raw/batch_001'))/cfg['id'];self.d=json.loads((self.path/'assembly.json').read_text())
  self.ids=list(self.d['occurrences']);self.occ=list(self.d['occurrences'].values());ix={x:i for i,x in enumerate(self.ids)}
  assert not any(b.get('is_visible',True) for b in (self.d['root'].get('bodies') or {}).values()), 'Visible root bodies need dedicated adapter'
  assert all(not children for children in self.d['tree']['root'].values()),'Nested instances need recursive adapter'
  self.source_T=np.array([transform(o['transform']) for o in self.occ])
  self.meshes=[]
  for o in self.occ:
   parts=[]
   for bid,b in o['bodies'].items():
    if not b.get('is_visible',True):raise ValueError('Hidden body requires explicit policy')
    m=trimesh.load(self.path/(bid+'.obj'),process=False);m.vertices*=.01;parts.append(m)
   self.meshes.append(trimesh.util.concatenate(parts))
  source_points=np.vstack([trimesh.transform_points(m.vertices,T) for m,T in zip(self.meshes,self.source_T)])
  reference=trimesh.load(self.path/'assembly.obj',process=False);reference.vertices*=.01
  D=np.linalg.norm(reference.extents)
  self.alignment=float(np.quantile(cKDTree(reference.vertices).query(source_points)[0],.95)/D)
  if self.alignment>.03:raise ValueError(f'Reference geometry mismatch {self.alignment}')
  self.C=np.eye(4);self.C[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
  points=trimesh.transform_points(source_points,self.C)
  self.C[:3,3]=[-points[:,0].mean(),-points[:,1].mean(),-points[:,2].min()]
  self.T=self.C@self.source_T;self.root=cfg['root']
  self.model=build_source_model(self.ids, dict(zip(self.ids,self.T)), self.d.get('joints') or {},
   self.ids[self.root], self.C)
  self.solver=self.model.solver
  self.root_indices=[ix[root] for root in self.model.roots]
  self.F=np.asarray([self.solver.forward(np.zeros(self.solver.dof_count))[link] for link in self.ids])
  edge_by_id={edge.id:edge for edge in self.model.edges}
  self.edges=[]
  for edge in self.model.edges:
   e=dict(id=edge.id,name=edge.name,a=ix[edge.parent],b=ix[edge.child],kind=edge.source.get('joint_motion',{}).get('joint_type'),
          canonical_type=edge.type,source=edge.source,axes=edge.spec.axes_in_joint,
          pivot=edge.spec.T_parent_joint[:3,3],raw_limits=None)
   if edge.type!='fixed':
    e['raw_limits']=(edge.source.get('joint_motion') or {}).get('rotation_limits',
      (edge.source.get('joint_motion') or {}).get('slide_limits'))
   self.edges.append(e)
  self.active=[e for e in self.edges if e['canonical_type']!='fixed' and e['id'] in {j.id for j in self.solver.tree_joints}]
  self.renderable_active=[e for e in self.active if e['canonical_type'] in {'revolute','prismatic'} and
                          next(j for j in self.solver.tree_joints if j.id==e['id']).dof_count==1]
  self.tree=[]
  for joint in self.solver.tree_joints:
   edge=edge_by_id[joint.id];e=next(item for item in self.edges if item['id']==joint.id)
   p=ix[joint.parent];c=ix[joint.child]
   dof_start=self.solver.dof_order.index(joint.dof_names[0]) if joint.dof_names else None
   self.tree.append(dict(p=p,c=c,e=e,A=joint.T_parent_joint,qi=dof_start,
                         dof_names=joint.dof_names,joint=joint))
  self.B=np.asarray([np.linalg.inv(self.F[i])@self.T[i] for i in range(len(self.ids))])
  self.jnames=list(self.solver.dof_order[:sum(j.dof_count for j in self.solver.tree_joints)])
  self.joint_names=[e['name'] for e in self.active]
  self.dof_indices_by_joint={joint.id:list(range(self.solver.dof_order.index(joint.dof_names[0]),
    self.solver.dof_order.index(joint.dof_names[0])+joint.dof_count)) for joint in self.solver.tree_joints if joint.dof_names}
  if not set(cfg['drive'])<=set(self.joint_names):
   raise ValueError(f'drive joints not in spanning tree: {set(cfg["drive"])-set(self.joint_names)}')
 def _solve_tree(self,q):
  q=np.asarray(q,dtype=float).reshape(-1)
  tree_dofs=sum(j.dof_count for j in self.solver.tree_joints)
  if q.size!=tree_dofs: raise ValueError(f'expected {tree_dofs} tree q values, got {q.size}')
  if not self.solver.loop_joints:
   return self.solver.forward(q)
  full=np.zeros(self.solver.dof_count)
  full[:tree_dofs]=q
  prescribed={name:float(value) for name,value in zip(self.jnames,q)}
  result=self.solver.solve(full,prescribed=prescribed)
  if not result.success or np.linalg.norm(result.residual)>1e-7:
   raise ValueError(f'kinematic loop solve failed: {result.message}; residual={np.linalg.norm(result.residual)}')
  return result.transforms
 def fk(self,q):
  poses=self._solve_tree(q)
  return np.asarray([poses[link] for link in self.ids])
 def trajectory(self):
  n=len(self.jnames);drives=list(self.cfg['drive']);targets=[np.zeros(n)]
  for name in drives:
   v=np.zeros(n)
   edge=next(e for e in self.active if e['name']==name)
   indices=self.dof_indices_by_joint[edge['id']]
   value=np.asarray(self.cfg['drive'][name],dtype=float).reshape(-1)
   if value.size not in {1,len(indices)}: raise ValueError(f'drive for {name} expects {len(indices)} values')
   v[indices]=value[0] if value.size==1 else value
   targets.extend([v,np.zeros(n)])
  v=np.zeros(n)
  for edge in self.active:
   indices=self.dof_indices_by_joint[edge['id']]
   if edge['name'] in self.cfg['drive']:
    value=np.asarray(self.cfg['drive'][edge['name']],dtype=float).reshape(-1)
    v[indices]=value[0] if value.size==1 else value
  targets.extend([v,-v*.5,np.zeros(n)])
  knots=np.r_[0,np.linspace(1,11,len(targets)),12];values=np.vstack([targets[0],targets,targets[-1]])
  ts=np.arange(360)/30;targets_by_frame=[]
  for t in ts:
   k=min(np.searchsorted(knots,t,side='right')-1,len(knots)-2);u=(t-knots[k])/(knots[k+1]-knots[k]);s=6*u**5-15*u**4+10*u**3;targets_by_frame.append(values[k]*(1-s)+values[k+1]*s)
  commanded_indices=sorted({index for name in self.cfg['drive']
    for index in self.dof_indices_by_joint[next(edge['id'] for edge in self.active if edge['name']==name)]})
  Q,F,Qall=solve_commanded_trajectory(self.solver,self.jnames,targets_by_frame,
                                       commanded_indices,ts)
  return ts,Q,F,Qall

def export(a, out, preview=False, template_scene=None):
 out.mkdir(parents=True,exist_ok=False);(out/'meshes').mkdir()
 ts,Q,F,Qall=a.trajectory();n=len(a.ids);meshes={}
 for i,m in enumerate(a.meshes):
  m=m.copy();m.apply_transform(a.B[i]);m.export(out/'meshes'/f'link_{i:02d}.obj');meshes[i]=m
 # The studio scene is only a renderer template; bodies and meshes are rebuilt from source joints.
 template = Path(template_scene) if template_scene else Path('annoation/approved_reference/scene.xml')
 if not template.exists():
  raise FileNotFoundError(f'MuJoCo scene template not found: {template}')
 xml=ET.parse(template);world=xml.getroot().find('worldbody');assets=xml.getroot().find('asset')
 for node in list(world):
  if node.tag in ['body','camera']:world.remove(node)
 for node in list(assets):
  if node.tag=='mesh':assets.remove(node)
 robot=ET.Element('robot',name=a.cfg['id']);ET.SubElement(robot,'link',name='world');bodies={};links={};runtime=[]
 def add_visual(link,name,color):
  v=ET.SubElement(link,'visual');origin_xml(v,np.eye(4));g=ET.SubElement(v,'geometry');ET.SubElement(g,'mesh',filename=f'meshes/{name}.obj');mat=ET.SubElement(v,'material',name=f'color_{name}');ET.SubElement(mat,'color',rgba=numbers(color))
 def add_mesh_body(parent,name,T,color,link):
  body=ET.SubElement(parent,'body',name=name,pos=numbers(T[:3,3]),quat=numbers(Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]))
  ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,rgba=numbers(color),mass='.01',contype='0',conaffinity='0')
  return body
 def add_urdf_link(name,color=None):
  link=ET.SubElement(robot,'link',name=name)
  if color is not None:add_visual(link,name,color)
  return link
 def add_urdf_joint(name,kind,parent,child,T,axis=None):
  joint=ET.SubElement(robot,'joint',name=name,type=kind);ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=child);origin_xml(joint,T)
  if axis is not None:ET.SubElement(joint,'axis',xyz=numbers(axis))
  return joint
 root_color=[.18,.22,.25,1]
 for root_number,root_index in enumerate(a.root_indices):
  root_name=f'link_{root_index:02d}';root_link=add_urdf_link(root_name,root_color)
  mount='world_mount' if root_number==0 else f'world_mount_{root_number:02d}'
  add_urdf_joint(mount,'fixed','world',root_name,a.F[root_index])
  ET.SubElement(assets,'mesh',name=root_name,file=f'{root_name}.obj')
  bodies[root_index]=add_mesh_body(world,root_name,a.F[root_index],root_color,root_link)
  links[root_index]=root_name
 tree_by_parent={}
 for t in a.tree:tree_by_parent.setdefault(t['p'],[]).append(t)
 for t in a.tree:
  p,c,e=t['p'],t['c'],t['e'];joint=t['joint'];parent_link=links[p];parent_body=bodies[p];name=f'link_{c:02d}';color=[.92,.55,.12,1] if e['canonical_type']!='fixed' else [.20,.52,.56,1]
  link=add_urdf_link(name,color);ET.SubElement(assets,'mesh',name=name,file=f'{name}.obj')
  dofs=list(t['dof_names']);kind=e['canonical_type'];A=joint.T_parent_joint;B=joint.T_joint_child_ref
  if not dofs:
   add_urdf_joint(e['name'],'fixed',parent_link,name,A@B);bodies[c]=add_mesh_body(parent_body,name,A@B,color,link);links[c]=name;continue
  if kind=='ball':
   current_link=parent_link
   for index,dof in enumerate(dofs):
    chain=f'kin_{e["id"]}_ball_{index}';chain_link=add_urdf_link(chain);T=A if index==0 else np.eye(4);add_urdf_joint(dof,'continuous',current_link,chain,T,-np.eye(3)[index]);current_link=chain
   body=ET.SubElement(parent_body,'body',name=f'kin_{e["id"]}_ball',pos=numbers(A[:3,3]),quat=numbers(Rotation.from_matrix(A[:3,:3]).as_quat()[[3,0,1,2]]));ET.SubElement(body,'inertial',pos='0 0 0',mass='0.0001',diaginertia='1e-8 1e-8 1e-8');ET.SubElement(body,'joint',name=dofs[0],type='ball');runtime.append({'q_indices':[a.jnames.index(x) for x in dofs],'sim_name':dofs[0],'kind':'ball','inverse_motion':joint.inverse_motion});add_urdf_joint('fixed_'+e['id'],'fixed',current_link,name,B);bodies[c]=add_mesh_body(body,name,B,color,link);links[c]=name;continue
  current_link=parent_link;current_body=parent_body
  for index,dof in enumerate(dofs):
   kin=f'kin_{e["id"]}_{index}';kin_link=add_urdf_link(kin);T=A if index==0 else np.eye(4);axis=np.array([1.,0.,0.]) if kind=='cylindrical' else np.eye(3)[index if index<3 else 0]
   if joint.inverse_motion:axis=-axis
   joint_kind='continuous' if kind in {'revolute','cylindrical'} and index==0 else ('prismatic' if ((kind=='prismatic') or (kind in {'cylindrical','pin_slot','planar'} and ((kind=='cylindrical' and index==1) or (kind=='pin_slot' and index==1) or (kind=='planar' and index<2)))) else 'continuous')
   add_urdf_joint(dof,joint_kind,current_link,kin,T,axis);body=ET.SubElement(current_body,'body',name=kin,pos=numbers(T[:3,3]),quat=numbers(Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]));ET.SubElement(body,'inertial',pos='0 0 0',mass='0.0001',diaginertia='1e-8 1e-8 1e-8');ET.SubElement(body,'joint',name=dof,type='hinge' if joint_kind=='continuous' else 'slide',axis=numbers(axis));runtime.append({'q_indices':[a.jnames.index(dof)],'sim_name':dof,'kind':'scalar'});current_link=kin;current_body=body
  add_urdf_joint('fixed_'+e['id'],'fixed',current_link,name,B);bodies[c]=add_mesh_body(current_body,name,B,color,link);links[c]=name
 K,T,R,eye,framing=camera_fit(meshes,F,1920,1080)
 ET.SubElement(world,'camera',name='annotation',pos=numbers(eye),quat=numbers(Rotation.from_matrix(R).as_quat()[[3,0,1,2]]),fovy='38')
 dump(out/'joint_runtime.json',dict(joints=runtime,joint_order=a.jnames))
 write_xml(out/'scene.xml',xml.getroot());write_xml(out/'assembly.urdf',robot)
 np.savez_compressed(out/'motion.npz',timestamps=ts,q=Q,q_valid=np.ones_like(Q,dtype=bool),T_world_link=F,T_world_occurrence=F@a.B,joint_order=np.array(a.jnames),all_joint_order=np.array(a.solver.dof_order),q_all=Qall,link_order=np.array([f'link_{i:02d}' for i in range(n)]));np.save(out/'timestamps.npy',ts)
 model=mujoco.MjModel.from_xml_path(str((out/'scene.xml').resolve()));data=mujoco.MjData(model);renderer=mujoco.Renderer(model,1080,1920)
 bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'link_{i:02d}') for i in range(n)];cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'annotation')
 writer=None if preview else imageio.get_writer(out/'rgb.mp4',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-crf','18','-preset','fast'])
 frames=[0,60,120,180,240,300];ims=[];err=0.;actual=None
 try:
  for i in (frames if preview else range(360)):
   set_qpos(model,data,Q[i],np.array(a.jnames),dict(joints=runtime));mujoco.mj_forward(model,data);err=max(err,float(np.abs(data.xpos[bids]-F[i,:,:3,3]).max()),float(np.abs(data.xmat[bids].reshape(-1,3,3)-F[i,:,:3,:3]).max()))
   actual=np.eye(4);actual[:3,3]=data.cam_xpos[cid];actual[:3,:3]=data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);assert np.allclose(actual,T,atol=1e-8)
   renderer.update_scene(data,camera='annotation');im=renderer.render()
   if writer:writer.append_data(im)
   if i in frames:ims.append(im[::3,::3]);imageio.imwrite(out/f'frame_{i:03d}.png',im)
   if i%120==0:print(a.cfg['id'],'view1',i,flush=True)
 finally:
  if writer:writer.close()
  renderer.close()
 assert err<1e-8
 imageio.imwrite(out/'contact_sheet.png',np.concatenate([np.concatenate(ims[:3],1),np.concatenate(ims[3:],1)],0))
 dump(out/'camera.json',dict(resolution_wh=[1920,1080],fps=30,K=K.tolist(),T_world_camera=actual.tolist(),camera_axes='x_right_y_down_z_forward',world_axes='z_up',units='m',distortion=[0]*5,pixel_center='integer; top-left (0,0)',fovy_degrees=38,framing=framing))
 np.savez_compressed(out/'camera.npz',timestamps=ts,K=np.repeat(K[None],360,0),T_world_camera=np.repeat(actual[None],360,0))
 dump(out/'loop_constraints.json',dict(constraints=[dict(id=edge.id,name=edge.name,parent=edge.parent,child=edge.child,type=edge.type,dof_names=list(edge.spec.dof_names),axes_in_joint=edge.spec.axes_in_joint.tolist()) for edge in a.model.loops],topology='spanning_tree_with_closure_edges'))
 dump(out/'kinematic_model.json',dict(links=list(a.ids),root=a.ids[a.root],roots=[a.ids[i] for i in a.root_indices],root_placement_policy='source_occurrence_pose_fixed' if len(a.root_indices)>1 else 'grounded_or_selected_occurrence',dof_order=list(a.solver.dof_order),tree_joints=[dict(id=j.id,parent=j.parent,child=j.child,type=j.type,dof_names=list(j.dof_names),inverse_motion=j.inverse_motion) for j in a.solver.tree_joints],loop_joints=[dict(id=j.id,parent=j.parent,child=j.child,type=j.type,dof_names=list(j.dof_names)) for j in a.solver.loop_joints]))
 dump(out/'part_mapping.json',dict(parts=[dict(link=f'link_{i:02d}',occurrence_id=a.ids[i],name=o['name'],body_ids=list(o['bodies'])) for i,o in enumerate(a.occ)],joint_order=a.jnames,source_to_canonical=a.C.tolist(),length_scale=.01))
 dump(out/'motion_description.json',dict(asset_id=a.cfg['id'],title_cn=a.cfg['title'],description_cn=a.cfg['description'],duration_s=12,source='pipeline_policy',joint_source='dataset_reference',states='synthetic_gt',commands=a.cfg['drive'],joint_units={e['name']:'m' if e['canonical_type']=='prismatic' else 'rad' for e in a.active},mechanical_limits='unknown; disabled source limits are not zero travel'))
 (out/'motion_description.txt').write_text(a.cfg['description']+'\n')
 dump(out/'source_joints.json',a.d['joints'])
 limitations=['Motion amplitudes are synthesis policy; source mechanical limits are disabled.', 'Root fixed by dataset grounded flag or explicit named-part selection.', 'Joint kinematics validated; no physical dynamics or continuous collision-free certification.', 'URDF prismatic limits describe clip sampling envelope, not measured travel.']
 if len(a.root_indices)>1:
  limitations.append('Disconnected source components are fixed independently at their imported occurrence poses; no inter-component joint is inferred.')
 dump(out/'quality.json',dict(status='kinematic_preview',instances=n,active_dofs=len(a.active),root_count=len(a.root_indices),reference_vertex_p95_over_D=a.alignment,mujoco_fk_error=err,limitations=limitations))
 dump(out/'provenance.json',dict(config=a.cfg,appearance_policy='contrasting wheel sector for motion visibility' if a.cfg['id']=='20341_6a1588fb' else 'studio part colors',source_json_sha256=digest(a.path/'assembly.json'),archive='a1.0.0_00.7z',hidden_root_bodies=a.d['root'].get('bodies',{}),source_geometry_hashes={p.name:digest(p) for p in a.path.glob('*.obj')}))
 print('reference complete',out,flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
 report=[]
 for cfg in CONFIGS:
  try:
   a=Assembly(cfg);path=ROOT/('batch_001_previews' if args.preview else 'batch_001_references')/cfg['id']
   if not path.exists():export(a,path,args.preview)
   if not args.preview:
    config=ROOT/f'batch_001_{cfg["id"]}_180_v2.json';dump(config,dict(approved_source=str(path.relative_to(ROOT)),sample_output=f'samples/{cfg["id"]}_multiview',cam02_azimuth_offset_degrees=180,border_fraction=.04,review_basis='batch source geometry and motion preview checked; opposing azimuths requested'))
    dest=ROOT/'samples'/f'{cfg["id"]}_multiview'
    if not dest.exists():multiview(config)
   report.append(dict(asset=cfg['id'],title=cfg['title'],status='preview_ready' if args.preview else 'completed'))
  except Exception as exc:
   report.append(dict(asset=cfg['id'],status='failed',error=str(exc)));print('FAILED',cfg['id'],repr(exc),flush=True)
  dump(ROOT/('batch_001_preview_status.json' if args.preview else 'batch_001_status.json'),report)
 if any(r['status']=='failed' for r in report):raise SystemExit(1)

if __name__=='__main__':main()

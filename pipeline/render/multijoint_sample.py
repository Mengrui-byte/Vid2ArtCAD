"""Render a Fusion assembly using occurrence transforms and a kinematic tree."""
from __future__ import annotations
import argparse,json,math,shutil
from pathlib import Path
import imageio,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,trimesh

def p3(v): return np.array([v['x'],v['y'],v['z']],float)
def mesh(p):
 m=trimesh.load(str(p)); return trimesh.util.concatenate(list(m.geometry.values())) if isinstance(m,trimesh.Scene) else m
def T_from_json(t):
 M=np.eye(4); M[:3,0]=p3(t['x_axis']); M[:3,1]=p3(t['y_axis']); M[:3,2]=p3(t['z_axis']); M[:3,3]=p3(t['origin']); return M
def select_joints(d,n=4):
 out=[]
 for jid,j in (d.get('joints') or {}).items():
  if (j.get('joint_motion') or {}).get('joint_type')!='RevoluteJointType': continue
  g=j.get('geometry_or_origin_two') or {}; l=j.get('joint_motion',{}).get('rotation_limits',{})
  lo=float(l.get('minimum_value',-0.45)); hi=float(l.get('maximum_value',0.45))
  if abs(hi-lo)<1e-6: lo,hi=-0.45,0.45
  out.append((jid,j,g,lo,hi))
  if len(out)>=n: break
 return out
def descendants(edges,root):
 seen=set(); stack=[root]
 while stack:
  x=stack.pop()
  if x in seen: continue
  seen.add(x); stack.extend(c for p,c in edges if p==x)
 return seen
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--asset',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); asset=Path(a.asset); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 d=json.loads((asset/'assembly.json').read_text()); occ=d.get('occurrences') or {}; files={p.stem:p for p in asset.glob('*.obj') if p.stem!='assembly'}
 instances=[]
 for oid,o in occ.items():
  if not o.get('transform'): continue
  T=T_from_json(o['transform'])
  for bid in (o.get('bodies') or {}):
   if bid in files: instances.append((oid,bid,mesh(files[bid]),T))
 joints=select_joints(d,4); edges=[(j[1].get('occurrence_one'),j[1].get('occurrence_two')) for j in joints]
 selected_ids={x[0] for x in joints}; groups=[descendants(edges,j[1].get('occurrence_two')) for j in joints]
 # Reference world geometry from local body mesh + occurrence transform.
 ref=trimesh.util.concatenate([m.copy().apply_transform(T) for _,_,m,T in instances]); b=ref.bounds; c=b.mean(0); e=b[1]-b[0]
 W,H,fps,N=1280,720,30,240; K=np.array([[900,0,W/2],[0,900,H/2],[0,0,1]],float); ts=np.arange(N)/fps; np.savez(out/'camera.npz',K=K,T_world_camera=np.eye(4),timestamps=ts); (out/'camera.json').write_text(json.dumps({'resolution_wh':[W,H],'fps':fps,'K':K.tolist(),'T_world_camera':np.eye(4).tolist(),'axes':'x_right_y_down_z_forward','depth':'z_depth_m'},indent=2))
 writer=imageio.get_writer(str(out/'rgb.mp4'),fps=fps,macro_block_size=1,quality=8)
 for frame in range(N):
  fig=plt.figure(figsize=(12.8,7.2),dpi=100); ax=fig.add_subplot(121,projection='3d'); zoom=fig.add_subplot(122,projection='3d');
  for qax,title,lim,cc in [(ax,'Complete assembly',e,c),(zoom,'Multi-joint motion detail',e*.32,ref.bounds.mean(0))]:
   qax.set_title(title,fontsize=12); qax.set_axis_off(); qax.set_box_aspect(tuple(np.maximum(lim,1e-6))); qax.view_init(elev=18,azim=35); qax.set_xlim(cc[0]-lim[0]*.55,cc[0]+lim[0]*.55); qax.set_ylim(cc[1]-lim[1]*.55,cc[1]+lim[1]*.55); qax.set_zlim(cc[2]-lim[2]*.55,cc[2]+lim[2]*.55)
  transforms={oid:T.copy() for oid,_,_,T in instances}
  for k,(jid,j,g,lo,hi) in enumerate(joints):
   q=lo+(hi-lo)*.5*(1-math.cos(2*math.pi*frame/N + k*.55)); axis=p3(g['primary_axis_vector']); axis/=max(np.linalg.norm(axis),1e-9); R=trimesh.transformations.rotation_matrix(q,axis,p3(g['origin']))
   for oid in groups[k]:
    if oid in transforms: transforms[oid]=R@transforms[oid]
  for oid,bid,m,T0 in instances:
   x=m.copy(); x.apply_transform(transforms[oid]);
   if len(x.faces)>700: x=x.submesh([np.linspace(0,len(x.faces)-1,700,dtype=int)],append=True,repair=False)
   color=(.92,.35,.08) if any(oid in g for g in groups) else (.45,.62,.82)
   ax.plot_trisurf(x.vertices[:,0],x.vertices[:,1],x.vertices[:,2],triangles=x.faces,color=color,linewidth=0,antialiased=False)
   if any(oid in g for g in groups): zoom.plot_trisurf(x.vertices[:,0],x.vertices[:,1],x.vertices[:,2],triangles=x.faces,color=color,linewidth=0,antialiased=False)
  fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)
 writer.close()
 # URDF: one link per occurrence and a spanning-tree subset of source joints.
 links=[]; used=set(); urdf_j=[]; parent_seen=set(); mesh_out=out/'meshes'; mesh_out.mkdir(exist_ok=True)
 for oid,o in occ.items():
  visual=''
  body_ids=[bid for bid in (o.get('bodies') or {}) if bid in files]
  if body_ids:
   visual=f'<visual><geometry><mesh filename="meshes/{body_ids[0]}.obj"/></geometry></visual>'
   for bid in body_ids:
    dst=mesh_out/f'{bid}.obj'
    if not dst.exists(): shutil.copy2(files[bid],dst)
  links.append(f'  <link name="occ_{oid}">{visual}</link>')
 for jid,j,g,lo,hi in select_joints(d,64):
  p=j.get('occurrence_one'); ch=j.get('occurrence_two')
  if p in parent_seen or p==ch: continue
  parent_seen.add(p); axis=p3(g['primary_axis_vector']); axis/=max(np.linalg.norm(axis),1e-9); o=p3(g['origin']); urdf_j.append(f'  <joint name="{jid}" type="revolute"><parent link="occ_{p}"/><child link="occ_{ch}"/><origin xyz="{o[0]} {o[1]} {o[2]}"/><axis xyz="{axis[0]} {axis[1]} {axis[2]}"/><limit lower="{lo}" upper="{hi}" effort="1" velocity="1"/></joint>')
 (out/'assembly.urdf').write_text('<?xml version="1.0"?>\n<robot name="vid2artcad_23226">\n'+'\n'.join(links+urdf_j)+'\n</robot>\n')
 text=f'完整台灯装配体包含 {len(files)} 个几何 body 和 {len(d.get("joints") or {})} 条 Fusion 关节约束。视频选取 {len(joints)} 个旋转关节进行联动演示，运动范围来自源数据；未启用范围的关节采用 pipeline_policy 的 ±0.45 rad 演示范围。左侧为完整装配体，右侧为运动区域放大视图。'
 (out/'motion_description.txt').write_text(text+'\n'); (out/'motion_description.json').write_text(json.dumps({'asset_id':asset.name,'source_joint_count':len(d.get('joints') or {}),'animated_joint_ids':[x[0] for x in joints],'fps':fps,'duration_s':N/fps,'description_cn':text},ensure_ascii=False,indent=2)); (out/'sample.json').write_text(json.dumps({'video':'rgb.mp4','urdf':'assembly.urdf','camera':'camera.json','motion_description':'motion_description.json'},indent=2)); print(out)
if __name__=='__main__': main()

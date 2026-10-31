"""Export a high-resolution sample with URDF, video, camera and motion text."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

def load_mesh(p):
    m=trimesh.load(str(p))
    return trimesh.util.concatenate(list(m.geometry.values())) if isinstance(m,trimesh.Scene) else m
def p3(x): return np.array([x["x"],x["y"],x["z"]],float)
def export_urdf(asset,out):
    d=json.loads((asset/"assembly.json").read_text()); files={p.stem:p for p in asset.glob("*.obj") if p.stem!="assembly"}; links=[]; ids={}
    moving=set()
    if d.get("joints"):
        j=next(iter(d["joints"].values())); moving=set(((d.get("occurrences") or {}).get(j.get("occurrence_two"),{}).get("bodies") or {}).keys())
    for i,b in enumerate(sorted(files)):
        lid=f"body_{i:04d}"; ids[b]=lid; links.append(f'  <link name="{lid}"><visual><geometry><mesh filename="meshes/{files[b].name}"/></geometry></visual></link>')
    joints=[]
    for i,j in enumerate((d.get("joints") or {}).values()):
        e1=j["geometry_or_origin_one"]["entity_one"]; e2=j["geometry_or_origin_two"]["entity_one"]; g=j["geometry_or_origin_two"]; o=p3(g["origin"]); a=p3(g["primary_axis_vector"]); a/=max(np.linalg.norm(a),1e-9); lim=j["joint_motion"]["rotation_limits"]
        parent=ids.get(e1.get("body"),"body_0000"); child=ids.get(e2.get("body"),"body_0000")
        joints.append(f'  <joint name="joint_{i}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{o[0]} {o[1]} {o[2]}"/><axis xyz="{a[0]} {a[1]} {a[2]}"/><limit lower="{lim.get("minimum_value",-1)}" upper="{lim.get("maximum_value",1)}" effort="1" velocity="1"/></joint>')
        for bid in sorted(moving-{e2.get("body")}):
            joints.append(f'  <joint name="fixed_{ids[bid]}" type="fixed"><parent link="{child}"/><child link="{ids[bid]}"/></joint>')
    (out/"meshes").mkdir(parents=True,exist_ok=True)
    for p in files.values():
        dst=out/"meshes"/p.name
        if not dst.exists(): dst.symlink_to(p.resolve())
    (out/"assembly.urdf").write_text('<?xml version="1.0"?>\n<robot name="vid2artcad">\n'+'\n'.join(links+joints)+'\n</robot>\n')
    return len(files),len(joints)
def export_sample(asset,out):
    out.mkdir(parents=True,exist_ok=True); d=json.loads((asset/"assembly.json").read_text()); files={p.stem:p for p in asset.glob("*.obj") if p.stem!="assembly"}; parts=[(b,load_mesh(p)) for b,p in files.items()]; allm=trimesh.util.concatenate([m for _,m in parts]); bounds=allm.bounds; center=bounds.mean(0); extent=bounds[1]-bounds[0]
    jd=None
    if d.get("joints"):
        j=next(iter(d["joints"].values())); g=j["geometry_or_origin_two"]; axis=p3(g["primary_axis_vector"]); axis/=max(np.linalg.norm(axis),1e-9); lim=j["joint_motion"]["rotation_limits"]; moving=set(((d.get("occurrences") or {}).get(j.get("occurrence_two"),{}).get("bodies") or {}).keys()); jd=(p3(g["origin"]),axis,lim.get("minimum_value",-1),lim.get("maximum_value",1),moving)
    W,H,fps,N=1280,720,30,240; K=np.array([[900,0,W/2],[0,900,H/2],[0,0,1]],float); ts=np.arange(N)/fps; np.savez(out/"camera.npz",K=K,T_world_camera=np.eye(4),timestamps=ts); (out/"camera.json").write_text(json.dumps({"resolution_wh":[W,H],"fps":fps,"K":K.tolist(),"T_world_camera":np.eye(4).tolist(),"axes":"x_right_y_down_z_forward","depth":"z_depth_m"},indent=2))
    moving_meshes=[m for bid,m in parts if jd and bid in jd[4]]
    moving_all=trimesh.util.concatenate(moving_meshes) if moving_meshes else allm
    mb=moving_all.bounds; mc=mb.mean(0); me=mb[1]-mb[0]
    writer=imageio.get_writer(str(out/"rgb.mp4"),fps=fps,macro_block_size=1,quality=8)
    for i in range(N):
        fig=plt.figure(figsize=(12.8,7.2),dpi=100)
        ax=fig.add_subplot(121,projection="3d"); ax.set_title("Full assembly",fontsize=12); ax.set_axis_off(); ax.set_box_aspect(tuple(extent)); ax.set_xlim(center[0]-extent[0]*.55,center[0]+extent[0]*.55); ax.set_ylim(center[1]-extent[1]*.55,center[1]+extent[1]*.55); ax.set_zlim(center[2]-extent[2]*.55,center[2]+extent[2]*.55); ax.view_init(elev=18,azim=35)
        if jd: q=jd[2]+(jd[3]-jd[2])*.5*(1-math.cos(2*math.pi*i/(N-1))); R=trimesh.transformations.rotation_matrix(q,jd[1],jd[0])
        for bid,m in parts:
            x=m.copy()
            if jd and bid in jd[4]: x.apply_transform(R)
            if len(x.faces)>1000: x=x.submesh([np.linspace(0,len(x.faces)-1,1000,dtype=int)],append=True,repair=False)
            ax.plot_trisurf(x.vertices[:,0],x.vertices[:,1],x.vertices[:,2],triangles=x.faces,color=(.9,.45,.15) if jd and bid in jd[4] else (.45,.62,.82),linewidth=0,antialiased=False)
        zoom=fig.add_subplot(122,projection="3d"); zoom.set_title("Rev2 motion detail",fontsize=12); zoom.set_axis_off(); zoom.set_box_aspect(tuple(np.maximum(me,1e-6))); pad=np.maximum(me*.35,1e-3); zoom.set_xlim(mc[0]-pad[0],mc[0]+pad[0]); zoom.set_ylim(mc[1]-pad[1],mc[1]+pad[1]); zoom.set_zlim(mc[2]-pad[2],mc[2]+pad[2]); zoom.view_init(elev=20,azim=35)
        for bid,m in parts:
            if not jd or bid not in jd[4]: continue
            z=m.copy(); z.apply_transform(R)
            if len(z.faces)>2500: z=z.submesh([np.linspace(0,len(z.faces)-1,2500,dtype=int)],append=True,repair=False)
            zoom.plot_trisurf(z.vertices[:,0],z.vertices[:,1],z.vertices[:,2],triangles=z.faces,color=(.95,.35,.08),linewidth=0,antialiased=False)
        fig.canvas.draw(); writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)
    writer.close(); nlinks,nj=export_urdf(asset,out); lo,hi=(jd[2],jd[3]) if jd else (0,0); count=len(jd[4]) if jd else 0; text=f"完整装配体包含 {nlinks} 个部件。Rev2 关节驱动 occurrence_two 下的 {count} 个部件作为刚性运动组，绕 Z 轴在 {lo:.3f} 到 {hi:.3f} 弧度之间往复旋转，其余部件保持装配姿态。"; (out/"motion_description.txt").write_text(text+"\n"); (out/"motion_description.json").write_text(json.dumps({"asset_id":asset.name,"link_count":nlinks,"joint_count":nj,"moving_body_count":count,"description_cn":text,"fps":fps,"duration_s":N/fps},ensure_ascii=False,indent=2)); (out/"sample.json").write_text(json.dumps({"video":"rgb.mp4","urdf":"assembly.urdf","camera":"camera.json","motion_description":"motion_description.json"},indent=2)); print(out)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--asset",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); export_sample(Path(a.asset),Path(a.out))

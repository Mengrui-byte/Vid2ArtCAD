"""Version the existing approved samples with opposite-azimuth camera pairs."""
import json
from pathlib import Path

from pipeline.render.multiview_annotation import run
from pipeline.render.stereo_annotation import dump


def main():
    root=Path('annoation')
    old=[root/f'batch_001_{aid}.json' for aid in
         ['22218_2f85f6ae','21805_2ba39d00','20259_de1ab632','20341_6a1588fb','22043_041e1b7c']]
    old.append(root/'multiview.json')
    rows=[]
    for p in old:
        config=json.loads(p.read_text())
        aid=Path(config['sample_output']).name.removesuffix('_multiview')
        config['workflow_version']='multiview_opposite_2.0'
        config['cam02_azimuth_offset_degrees']=180
        config['sample_output']=f'samples/{aid}_multiview'
        config['supersedes']=json.loads(p.read_text())['sample_output']
        new=root/(p.stem+'_180_v2.json')
        dump(new,config)
        target=root/config['sample_output']
        if not target.exists():run(new)
        quality=json.loads((target/'quality.json').read_text())
        rows.append(dict(asset_id=aid,path=config['sample_output'],status=quality['status'],
            azimuth_degrees=quality['measured_azimuth_separation_degrees'],
            cam01_unchanged=quality['cam01_byte_identical_to_approved']))
        dump(root/'opposite_views_status.json',rows)
        print(f'Opposite views completed {len(rows)}/{len(old)}',flush=True)
    dump(root/'active_workflow.json',dict(mode='multiview_monocular',workflow_version='2.0',
        azimuth_offset_degrees=180,border_fraction=.04,samples=rows,
        documentation='多视角单目标注流程_v1.md',supersedes='90-degree and parallel-stereo versions'))


if __name__=='__main__':
    main()

"""Compile an exact binary scene cache; no simulation or renderer is run."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np

HERE=Path(__file__).resolve().parent
XML=HERE.parent/'mujoco_hotel/assets/hotel_b2w.xml'
BINARY=HERE/'hotel_scene.mjb'


def fingerprint():
    parts=[hashlib.sha256(XML.read_bytes()).hexdigest(),mujoco.__version__]
    for asset in ET.parse(XML).find('asset'):
        if asset.get('file'):
            path=XML.parent/asset.get('file');stat=path.stat()
            parts.append(f'{asset.get("file")}:{stat.st_size}:{stat.st_mtime_ns}')
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


if __name__=='__main__':
    m=mujoco.MjModel.from_xml_path(str(XML));mujoco.mj_saveModel(m,str(BINARY))
    restored=mujoco.MjModel.from_binary_path(str(BINARY))
    for key in ['body_mass','body_inertia','geom_pos','geom_dataid','jnt_range','cam_pos','cam_quat','mat_rgba']:
        np.testing.assert_array_equal(getattr(m,key),getattr(restored,key))
    report={'passed':True,'fingerprint':fingerprint(),'ngeom':m.ngeom,'nmesh':m.nmesh,'ntex':m.ntex,
            'bytes':BINARY.stat().st_size,'compiled_only':True}
    (HERE/'scene_cache.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

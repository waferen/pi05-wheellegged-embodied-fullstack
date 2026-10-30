"""No integration steps: compile derived model and check contract geometry/units."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from hotel_runtime import HotelSimulation,MODEL,actor_frame_velocity
from original_teleop import ARMS,LIMITS


def geometry(root):
    root=copy.deepcopy(root)
    for child in list(root.find('asset')):
        if child.tag in ('texture','material'):root.find('asset').remove(child)
    return ET.tostring(root)


def main():
    old=ET.parse(HERE/'backup/assets/hotel_b2w.xml').getroot();new=ET.parse(MODEL).getroot()
    assert geometry(old)==geometry(new),'Material refresh changed physics/geometry/cameras'
    before={x.get('name'):x.attrib for x in old.findall('asset/material')}
    after={x.get('name'):x.attrib for x in new.findall('asset/material')}
    changed=[{'name':k,'before':before[k],'after':v} for k,v in after.items() if before.get(k)!=v and k in before]
    for key in ('mat_c31a6b8d72b262','mat_6124368ed63db9'):
        assert after[key]['rgba']=='0.025 0.025 0.025 1' and 'texture' not in after[key]
    sim=HotelSimulation();m,d=sim.model,sim.data
    joints=[{'name':n,'qpos_address':int(m.joint(n).qposadr[0]),'range':m.joint(n).range.tolist(),
             'type':int(m.joint(n).type[0])} for n in ARMS]
    np.testing.assert_allclose(m.joint('arm_joint7').range,[0,.045],atol=1e-8)
    np.testing.assert_allclose(m.joint('arm_joint8').range,[-.045,0],atol=1e-8)
    # Isaac root_lin_vel_b is root-body COM velocity expressed in actor axes.
    # Compare against the independent COM Jacobian under a rotated root.
    q=Rotation.from_euler('xyz',[.2,-.3,1.1]).as_quat();d.qpos[3:7]=q[[3,0,1,2]]
    d.qvel[:]=np.linspace(-.4,.7,m.nv);mujoco.mj_forward(m,d)
    velocity=actor_frame_velocity(m,d,sim.base)
    jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBodyCom(m,d,jp,jr,sim.base)
    rotation=d.xmat[sim.base].reshape(3,3)
    expected=np.r_[rotation.T@(jr@d.qvel),rotation.T@(jp@d.qvel)]
    np.testing.assert_allclose(velocity,expected,atol=1e-10)
    camera_source=json.loads((HERE.parent/'camera_audit.json').read_text())['source']
    cameras={}
    for label,suffix in [('head','HeadCamera_HeadCamera'),('hand','GripperCamera_GripperCamera')]:
        cid=next(i for i in range(m.ncam) if m.camera(i).name.endswith(suffix));spec=camera_source[label]
        np.testing.assert_allclose(m.cam_pos[cid],spec['position_m'],atol=1e-7)
        np.testing.assert_allclose(m.cam_quat[cid],spec['quaternion_wxyz'],atol=1e-7)
        fx=640*m.cam_intrinsic[cid,0]/m.cam_sensorsize[cid,0]
        assert abs(fx-spec['fx_pixels'])<1e-3
        assert m.cam_resolution[cid].tolist()==[640,480]
        cameras[label]={'name':m.camera(cid).name,'resolution':[640,480],'fx':float(fx)}
    report={'passed':True,'physics_geometry_cameras_unchanged':True,'xml_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'changed_materials':changed,'joint_mapping':joints,'arm_command_limits_rad':LIMITS.tolist(),
            'velocity_com_actor_axes_max_error':float(np.max(np.abs(velocity-expected))),
            'cameras':cameras,'nq':m.nq,'nv':m.nv,'nu':m.nu,'note':'Static compilation and kinematic queries only; no simulation integration.'}
    (HERE/'static_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('changed_materials','joint_mapping')},indent=2))
    print('Changed materials:',len(changed))


if __name__=='__main__':main()

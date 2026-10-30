"""Compare source sensor transforms/intrinsics and render before/after correction."""
import json,math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image
from pxr import UsdGeom
from build_hotel import Builder,name,fmt,quat,HERE
from hotel_runtime import HotelSimulation,MODEL
from camera_calibration import intrinsic_attributes,is_robot_sensor,persist_physical_near

def frames(tag,legacy=False):
    sim=HotelSimulation();m,d=sim.model,sim.data
    if legacy:
        # Reconstruct the recorded pre-fix optics; never write them back to the XML.
        m.vis.map.znear=.01
        for i in range(m.ncam):
            if m.camera(i).name.endswith('HeadCamera_HeadCamera'):old=114.839
            elif m.camera(i).name.endswith('GripperCamera_GripperCamera'):old=103.296
            else:continue
            m.cam_sensorsize[i]=0;m.cam_resolution[i]=1;m.cam_fovy[i]=old
    option=mujoco.MjvOption();option.geomgroup[3]=0
    finger_bodies=[m.body('World_b2w_piper_arm_link'+str(i)).id for i in (7,8)]
    finger_geoms=np.flatnonzero(np.isin(m.geom_bodyid,finger_bodies)&(m.geom_group==0))
    result={}
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        for label,suffix in [('head','HeadCamera_HeadCamera'),('hand','GripperCamera_GripperCamera')]:
            cid=next(i for i in range(m.ncam) if m.camera(i).name.endswith(suffix))
            cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;cam.fixedcamid=cid
            zv=np.zeros(2,np.float32);zh=zv.copy();zc=zv.copy()
            mujoco.mjv_cameraFrustum(zv,zh,zc,m,cam)
            result[label]={'pos':m.cam_pos[cid].tolist(),'quat':m.cam_quat[cid].tolist(),
                'resolution':m.cam_resolution[cid].tolist(),'near_m':float(m.vis.map.znear*m.stat.extent),
                'frustum_vertical':zv.tolist(),'frustum_horizontal':zh.tolist(),'clip_m':zc.tolist(),
                'finger_pixels':{}}
            for state,value in [('open',.045),('closed',.001)]:
                d.joint('arm_joint7').qpos[0]=value;d.joint('arm_joint8').qpos[0]=-value
                mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=cam,scene_option=option)
                rgb=renderer.render();Image.fromarray(rgb).save(HERE/f'camera_{tag}_{label}_{state}.png')
                renderer.enable_segmentation_rendering();renderer.update_scene(d,camera=cam,scene_option=option)
                seg=renderer.render();renderer.disable_segmentation_rendering()
                pixels=int(np.count_nonzero(np.isin(seg[:,:,0],finger_geoms)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))))
                result[label]['finger_pixels'][state]=pixels
    return result

def main():
    report_path=HERE/'camera_audit.json'
    previous=json.loads(report_path.read_text()) if report_path.exists() else None
    before=previous['before'] if previous else frames('before',legacy=True)
    builder=Builder();tree=ET.parse(MODEL);root=tree.getroot();source={}
    for prim in builder.prims:
        if not prim.IsA(UsdGeom.Camera) or not is_robot_sensor(prim.GetPath()):continue
        camera=UsdGeom.Camera(prim);body=builder.body_for(prim)
        matrix=builder.xforms.GetLocalToWorldTransform(prim)*builder.body_world[body].GetInverse()
        node=next(e for e in root.iter('camera') if e.get('name')==name(prim.GetPath()))
        node.attrib.pop('fovy',None)
        node.set('pos',fmt(np.array(matrix)[3,:3]));node.set('quat',quat(matrix))
        node.attrib.update(intrinsic_attributes(camera))
        label='head' if 'HeadCamera' in str(prim.GetPath()) else 'hand'
        f=float(camera.GetFocalLengthAttr().Get());a=float(camera.GetHorizontalApertureAttr().Get())
        source[label]={'prim':str(prim.GetPath()),'body':body,'position_m':np.array(matrix)[3,:3].tolist(),
            'quaternion_wxyz':list(map(float,quat(matrix).split())),
            'focal_mm':f,'horizontal_aperture_mm':a,'authored_vertical_aperture_mm':float(camera.GetVerticalApertureAttr().Get()),
            'effective_vertical_aperture_mm':a*480/640,'fx_pixels':640*f/a,'fy_pixels':640*f/a,
            'horizontal_fov_deg':math.degrees(2*math.atan(a/(2*f))),
            'vertical_fov_deg':math.degrees(2*math.atan(a*480/640/(2*f))),
            'source_clip_m':list(camera.GetClippingRangeAttr().Get()),'resolution':[640,480]}
    ET.indent(tree);tree.write(MODEL,encoding='utf-8',xml_declaration=True)
    persist_physical_near(MODEL)
    after=frames('after')
    for label,spec in source.items():
        actual=after[label]
        actual['position_error_m']=float(np.linalg.norm(np.array(actual['pos'])-spec['position_m']))
        a=np.array(actual['quat']);b=np.array(spec['quaternion_wxyz']);b/=np.linalg.norm(b)
        actual['rotation_error_deg']=math.degrees(2*math.acos(min(1,abs(float(a@b)))))
        near=actual['clip_m'][0];horizontal=actual['frustum_horizontal'];vertical=actual['frustum_vertical']
        actual['horizontal_fov_deg']=math.degrees(math.atan(horizontal[1]/near)+math.atan(horizontal[0]/near))
        actual['vertical_fov_deg']=math.degrees(math.atan(vertical[1]/near)+math.atan(vertical[0]/near))
        assert actual['position_error_m']<1e-7
        assert actual['rotation_error_deg']<1e-4
        assert abs(actual['horizontal_fov_deg']-spec['horizontal_fov_deg'])<1e-3
        assert abs(actual['vertical_fov_deg']-spec['vertical_fov_deg'])<1e-3
        assert abs(actual['near_m']-.01)<1e-8
    assert min(after['hand']['finger_pixels'].values())>100
    report={'passed':True,'source_hotel':builder.source,'before':before,'source':source,'after':after,
        'intrinsic_reference':'https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/sensors/camera/camera.py',
        'note':'Pose compared in effective rigid-body frames. Source USD is unchanged. Rendering uses MuJoCo materials, not RTX pixel equivalence.'}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Calibrated existing robot sensors; USD/OpenGL and MuJoCo share -Z/+Y axes."""
import xml.etree.ElementTree as ET

WIDTH,HEIGHT=640,480
NEAR_METERS=.01
SENSOR_SUFFIXES=('HeadCamera/HeadCamera','GripperCamera/GripperCamera')

def is_robot_sensor(path):
    return str(path).endswith(SENSOR_SUFFIXES)

def intrinsic_attributes(camera):
    # Original IsaacLab capture is 640x480, with fx=fy=W*focal/horizontalAperture.
    # Its camera calibration ignores the authored vertical aperture (IsaacLab 2.3.2).
    focal=float(camera.GetFocalLengthAttr().Get())
    horizontal=float(camera.GetHorizontalApertureAttr().Get())
    vertical=horizontal*HEIGHT/WIDTH
    return {'resolution':f'{WIDTH} {HEIGHT}',
            'sensorsize':f'{horizontal:.12g} {vertical:.12g}',
            'focal':f'{focal:.12g} {focal:.12g}'}

def set_physical_near(model):
    # MuJoCo's visual.map.znear is a fraction of the entire model extent.
    model.vis.map.znear=NEAR_METERS/float(model.stat.extent)

def configure_scene_rendering(scene):
    # Thin imported shells need their interior faces visible too (e.g. vessel).
    import mujoco
    scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE]=False

def persist_physical_near(path):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(path))
    tree=ET.parse(path)
    tree.getroot().find('visual/map').set('znear',f'{NEAR_METERS/float(model.stat.extent):.15g}')
    ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)

def sensor_viewport(width,height,header=0,fill=False):
    """Scale without distortion: fill crops edges; fit shows the complete sensor."""
    available=max(1,height-header)
    units=max((width+3)//4,(available+2)//3) if fill else min(width//4,available//3)
    w=max(4,4*units)
    h=w*HEIGHT//WIDTH
    return ((width-w)//2,(available-h)//2,w,h)

def dual_sensor_viewports(width,height,header=150,gap=12):
    """Two complete 4:3 images below the controls, with no stretch or crop."""
    cell=max(4,(width-gap)//2)
    available=max(3,height-header)
    units=max(1,min(cell//4,available//3))
    w,h=4*units,3*units
    y=max(0,(available-h)//2)
    return [((cell-w)//2,y,w,h),(width-cell+(cell-w)//2,y,w,h)]

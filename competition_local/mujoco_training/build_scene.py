"""Create a small training scene from the migrated, original Piper MJCF."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'cloud_snapshot/PPSUC2026/Piper_ros/src/piper_description/mujoco_model/piper_description.xml'
OUTPUT=ROOT/'mujoco_training/assets/piper_reach.xml'
HOME=[0.0,1.0,-1.2,0.0,0.35,0.0,0.025,-0.025]

def build():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(SOURCE); root=tree.getroot()
    root.set('model','PPSUC Piper local joint reaching')
    for mesh in root.findall('./asset/mesh'):
        original=(SOURCE.parent/mesh.attrib['file']).resolve()
        mesh.set('file',os.path.relpath(original,OUTPUT.parent).replace('\\','/'))
    root.insert(1,ET.Element('option',timestep='0.002',integrator='implicitfast',gravity='0 0 -9.81'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='720')
    ET.SubElement(visual,'headlight',ambient='0.4 0.4 0.4',diffuse='0.7 0.7 0.7')
    world=root.find('worldbody')
    base=ET.Element('body',name='base_link')
    for node in list(world):
        world.remove(node); base.append(node)
    world.append(base)
    contact=ET.SubElement(root,'contact')
    ET.SubElement(contact,'exclude',body1='base_link',body2='link1')
    ET.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1',directional='true')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 0.1',pos='0 0 -0.025',rgba='0.13 0.16 0.20 1')
    ET.SubElement(world,'geom',name='pedestal',type='box',size='0.085 0.085 0.012',pos='0 0 -0.012',rgba='0.28 0.32 0.38 1')
    ET.SubElement(world,'site',name='target',type='sphere',size='0.014',pos='0.4 0 0.4',rgba='0.2 0.9 0.45 0.7')
    ET.SubElement(root.find(".//body[@name='link6']"),'site',name='tcp',type='sphere',size='0.007',pos='0 0 0.17',rgba='1 0.4 0.05 1')
    joints={j.attrib['name']:j for j in root.findall('.//joint')}
    changes=[]
    for actuator in root.findall('./actuator/position'):
        joint=joints[actuator.attrib['joint']]
        if actuator.attrib['ctrlrange']!=joint.attrib['range']:
            changes.append({'joint':joint.attrib['name'],'old_ctrlrange':actuator.attrib['ctrlrange'],'new_ctrlrange':joint.attrib['range']})
        actuator.set('ctrlrange',joint.attrib['range'])
    # Geometry, masses, inertia, link transforms and joint limits remain source values.
    key=ET.SubElement(root,'keyframe')
    ET.SubElement(key,'key',name='home',qpos=' '.join(map(str,HOME)),ctrl=' '.join(map(str,HOME)))
    ET.indent(tree)
    tree.write(OUTPUT,encoding='utf-8',xml_declaration=True)
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(OUTPUT))
    record={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'output':str(OUTPUT.relative_to(ROOT)),'nq':model.nq,'nv':model.nv,'nu':model.nu,
            'timestep':model.opt.timestep,'actuator_range_alignment':changes,
            'collision_fix':'Wrap original world-mounted base mesh in base_link body; exclude only base_link/link1 adjacent self-collision (6 mm overlap in source). Ground and other collisions remain enabled.',
            'scope':'Original fixed-base Piper model; not the mobile B2W assembly or competition hotel.',
            'gripper_limit_m':0.035,'competition_usd_gripper_limit_m':0.045,
            'notes':'The Piper ROS reference and current competition USD have different gripper limits. This no-grasp smoke test retains the source MJCF limit.'}
    OUTPUT.with_suffix('.provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return OUTPUT

if __name__=='__main__':build()

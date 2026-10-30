"""Update already exported XML for inherited source collider APIs (no re-meshing)."""
import copy,json
import xml.etree.ElementTree as ET
from build_hotel import Builder,HERE,OUT,name,collision_enabled

b=Builder()
path=OUT/'hotel_b2w.xml';tree=ET.parse(path);root=tree.getroot()
parents={child:parent for parent in root.iter() for child in parent}
by_name={geom.get('name'):geom for geom in root.findall('.//geom')}
report=json.loads((HERE/'conversion_report.json').read_text(encoding='utf-8'))
added=[]
for record in report['mesh_records']:
    prim=b.stage.GetPrimAtPath(record['source'])
    if not collision_enabled(prim):continue
    record['collision']=True
    collision_name=name(record['source'])+'_collision'
    if collision_name in by_name:continue
    visual=by_name.get(name(record['source'])+'_visual')
    if visual is None:continue
    collider=copy.deepcopy(visual)
    collider.set('name',collision_name);collider.attrib.pop('material',None)
    collider.set('contype','2' if record['robot'] else '1');collider.set('conaffinity','1' if record['robot'] else '2')
    collider.set('group','3');collider.set('friction','1 .005 .0001');collider.set('solref','.01 1');collider.set('rgba','.2 .6 .3 .2')
    parents[visual].append(collider);added.append(record['source'])
root.find('./visual/headlight').set('ambient','.25 .25 .25');root.find('./visual/headlight').set('diffuse','.45 .45 .45')
for light in root.findall('./worldbody/light'):light.set('diffuse','.4 .4 .4')
tree.write(path,encoding='utf-8',xml_declaration=True)
report['inherited_colliders_added']=added
(HERE/'conversion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Mapped inherited collider APIs:',len(added),added)

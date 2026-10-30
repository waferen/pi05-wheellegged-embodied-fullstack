"""Re-export only derived materials using Windows long-path-safe image reads."""
import copy,xml.etree.ElementTree as ET
from pxr import UsdGeom
from build_hotel import Builder,OUT
b=Builder()
for prim in b.prims:
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):b.material(prim)
path=OUT/'hotel_b2w.xml';tree=ET.parse(path);assets=tree.find('asset')
for child in list(assets):
    if child.tag in ('texture','material'):assets.remove(child)
for child in b.assets:
    if child.tag in ('texture','material'):assets.append(copy.deepcopy(child))
tree.write(path,encoding='utf-8',xml_declaration=True)
print('Refreshed',len(assets.findall('material')),'materials and',len(assets.findall('texture')),'textures')

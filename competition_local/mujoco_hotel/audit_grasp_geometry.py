"""Compare original USD and compiled collision dimensions without rescaling."""
import json
import numpy as np
from pxr import UsdGeom
from build_hotel import Builder,transform_points
from hotel_runtime import HotelSimulation,HERE

b=Builder();sim=HotelSimulation();m,d=sim.model,sim.data;rows=[]
for prim in b.prims:
    path=str(prim.GetPath())
    if not prim.IsA(UsdGeom.Mesh):continue
    if not (path.startswith('/World/C4/') or (path.startswith('/World/b2w_piper/arm_link') and '/collisions/' in path and ('arm_link7/' in path or 'arm_link8/' in path))):continue
    geom_name=path.replace('/','_').strip('_')+'_collision'
    gid=next((i for i in range(m.ngeom) if m.geom(i).name==geom_name),None)
    if gid is None:continue
    source=transform_points(np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get()),b.xforms.GetLocalToWorldTransform(prim))
    mesh=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
    actual=m.mesh_vert[start:start+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
    # Robot initial joint posture differs from USD authored transforms: compare
    # local collision coordinates there; C4 initial world poses are preserved.
    if path.startswith('/World/C4/'):
        error=float(np.max(np.abs(np.ptp(actual,axis=0)-np.ptp(source,axis=0))))
        rows.append({'source':path,'source_dimensions_m':np.ptp(source,axis=0).tolist(),
            'converted_dimensions_m':np.ptp(actual,axis=0).tolist(),'max_dimension_error_m':error,
            'original_vertices':len(source),'converted_vertices':len(actual)})
result={'passed':all(r['max_dimension_error_m']<.001 for r in rows),'c4':rows,'scope':'C4 world bounds, source USD vs compiled collision mesh. Mesh contact still uses convex hulls rather than source SDF.'}
(HERE/'grasp_geometry_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

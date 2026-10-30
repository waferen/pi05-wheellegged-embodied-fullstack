"""Inspect competition USD composition without changing the source assets."""
import hashlib,json,os
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Sdf

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
SNAPSHOT=ROOT/'cloud_snapshot'
MODEL=SNAPSHOT/'PPSUC2026/ppsuc_robot_model'

def prepare_layers():
    cache=HERE/'usd_layers';cache.mkdir(parents=True,exist_ok=True)
    aliases=ROOT/'migration/usd_layer_aliases'
    manifest=json.loads((ROOT/'migration/final_source_manifest.json').read_text(encoding='utf-8'))
    paths=[e['path'] for e in manifest['entries'] if Path(e['path']).suffix.lower() in {'.usd','.usda','.usdc'}]
    mapping={str((SNAPSHOT/p).resolve()):cache/(hashlib.sha256(p.encode()).hexdigest()[:20]+'.usdc') for p in paths}
    missing=[]
    for index,relative in enumerate(paths):
        source=SNAPSHOT/relative;target=mapping[str(source.resolve())]
        if target.exists():continue
        alias=aliases/(hashlib.sha256(relative.encode()).hexdigest()[:20]+source.suffix)
        if not alias.exists():os.link(source,alias)
        layer=Sdf.Layer.OpenAsAnonymous(str(alias))
        def remap(asset):
            if not asset or '://' in asset or asset.endswith('.mdl'):return asset
            package=asset.split('[',1)[0]
            resolved=(source.parent/package).resolve()
            if str(resolved) in mapping:return str(mapping[str(resolved)]).replace('\\','/')
            if not resolved.is_file():missing.append({'source':relative,'asset':asset})
            # Geometry composition uses flat cached layers. Texture paths are dealt
            # with during export and may be read with Python's long-path support.
            return str(resolved).replace('\\','/')
        UsdUtils.ModifyAssetPaths(layer,remap)
        layer.Export(str(target))
        if index%25==0:print('cached',index+1,'of',len(paths),flush=True)
    (HERE/'layer_map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (HERE/'unresolved_geometry_assets.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2),encoding='utf-8')
    return mapping

def inspect(stage):
    cache=UsdGeom.XformCache()
    counts={};bodies=[];joints=[];meshes=[];cameras=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        name=prim.GetTypeName();counts[name]=counts.get(name,0)+1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass=UsdPhysics.MassAPI(prim)
            bodies.append({'path':str(prim.GetPath()),'mass':mass.GetMassAttr().Get(),'com':str(mass.GetCenterOfMassAttr().Get()),'world_transform':str(cache.GetLocalToWorldTransform(prim))})
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            joints.append({'path':str(prim.GetPath()),'type':name,'body0':list(map(str,joint.GetBody0Rel().GetTargets())),
                'body1':list(map(str,joint.GetBody1Rel().GetTargets())),
                'attrs':{a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('physics:')}})
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim)
            meshes.append({'path':str(prim.GetPath()),'points':len(mesh.GetPointsAttr().Get() or []),'faces':len(mesh.GetFaceVertexCountsAttr().Get() or []),'purpose':str(mesh.GetPurposeAttr().Get()),'visible':str(mesh.ComputeVisibility()),'collision':prim.HasAPI(UsdPhysics.CollisionAPI)})
        if prim.IsA(UsdGeom.Camera):cameras.append(str(prim.GetPath()))
    bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return {'counts':counts,'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':str(UsdGeom.GetStageUpAxis(stage)),
        'bounds':str(bound),'rigid_bodies':bodies,'joints':joints,'meshes':meshes,'cameras':cameras}

def main():
    mapping=prepare_layers()
    sources={'robot':MODEL/'robot/b2w/b2w_piper.usda','hotel':MODEL/'scene/Competition_hotel/Collected_task03/hotel/task03.usd'}
    for name,source in sources.items():
        print('opening',name,flush=True)
        stage=Usd.Stage.Open(str(mapping[str(source.resolve())]))
        info=inspect(stage)
        info['source']=str(source.relative_to(ROOT))
        (HERE/(name+'_inspection.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,info['counts'],'bodies',len(info['rigid_bodies']),'joints',len(info['joints']),info['bounds'],flush=True)

if __name__=='__main__':main()

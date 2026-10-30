"""Compile, measure free standing contacts, and render inspectable previews."""
import json,time
import numpy as np
import mujoco
from PIL import Image
from hotel_runtime import HotelSimulation,HERE

def main():
    started=time.perf_counter();sim=HotelSimulation();m=sim.model;d=sim.data
    samples=[]
    for i in range(2500):
        sim.step()
        if i%100==0:samples.append({'time':float(d.time),'base':d.xpos[sim.base].tolist(),'contacts':int(d.ncon),'speed':float(np.linalg.norm(d.qvel[:3]))})
    final=d.xpos[sim.base].copy()
    record={'checked_at':time.time(),'compile_and_test_seconds':time.perf_counter()-started,'nq':m.nq,'nv':m.nv,'nu':m.nu,'nbody':m.nbody,'ngeom':m.ngeom,'nmesh':m.nmesh,'ncamera':m.ncam,
        'free_root':True,'simulation_seconds':float(d.time),'floor_z':sim.floor_z,'spawn':sim.spawn.tolist(),'base_final':final.tolist(),
        'height_above_floor':float(final[2]-sim.floor_z),'finite':bool(np.isfinite(d.qpos).all()),'contacts_final':int(d.ncon),'samples':samples,
        'stance_controller':{'kp':600,'kv':20,'source_kp':160,'source_kd':5,'reason':'Static preview without the Isaac locomotion policy; source torque limits unchanged.'},
        'scope':'5 seconds of unpinned B2W/Piper standing with a tuned static stance hold; static hotel collision preview. Not a locomotion/grasping benchmark.'}
    rotation=d.xmat[sim.base].reshape(3,3)
    record['upright_z']=float(rotation[2,2]);record['standing_passed']=bool(record['finite'] and record['height_above_floor']>.35 and rotation[2,2]>.9 and d.ncon>=4)
    start_position=d.xpos[sim.base].copy()
    sim.speed=1.
    for _ in range(1000):sim.step()
    record['wheel_test']={'seconds':2.,'command_rad_s':1.,'displacement':(d.xpos[sim.base]-start_position).tolist(),
        'upright_z':float(d.xmat[sim.base].reshape(3,3)[2,2]),'finite':bool(np.isfinite(d.qpos).all())}
    sim.speed=0.
    for _ in range(250):sim.step()
    (HERE/'validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='samples'},ensure_ascii=False,indent=2),flush=True)
    renderer=mujoco.Renderer(m,height=800,width=1280)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    for name,overview in [('robot_preview.png',False),('hotel_overview.png',True)]:
        renderer.update_scene(d,camera=sim.camera(overview),scene_option=option)
        Image.fromarray(renderer.render()).save(HERE/name)
    renderer.close()
    assert record['standing_passed'],'Free standing check failed; inspect validation.json'
    assert record['wheel_test']['finite'] and np.linalg.norm(record['wheel_test']['displacement'][:2])>.05,'Wheel motion check failed'

if __name__=='__main__':main()

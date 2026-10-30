"""Measure contact force and prolonged sliding before changing product settings."""
import argparse,json,time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from grasp_bench import create_bench
from hotel_runtime import HERE

def trial(settings=None,offset=(0,0,0),model_path=None,approach_deg=-90,release=False,hold_seconds=10):
    m,d=create_bench(offset=offset,model_path=model_path,approach_deg=approach_deg);settings=settings or {}
    if 'kp' in settings:
        m.actuator_gainprm[:,0]=settings['kp'];m.actuator_biasprm[:,1]=-settings['kp']
    if 'impratio' in settings:m.opt.impratio=settings['impratio']
    if 'noslip' in settings:m.opt.noslip_iterations=settings['noslip']
    if 'iterations' in settings:m.opt.iterations=settings['iterations']
    if 'pair' in settings:
        assert m.npair>0,'No explicit contact pairs to adjust'
        for key,value in settings['pair'].items():getattr(m,'pair_'+key)[:]=value
    if 'contact' in settings:
        for g in range(m.ngeom):
            if not m.geom_contype[g]:continue
            if m.body(int(m.geom_bodyid[g])).name=='world':continue
            for key,value in settings['contact'].items():getattr(m,'geom_'+key)[g]=value
    obj=m.body('World_C4_easy_C4_1').id
    grip=m.body('World_b2w_piper_gripper_base').id
    fingers=[m.body('World_b2w_piper_arm_link'+str(i)).id for i in (7,8)]
    force=np.zeros(6);trace=[];start_clock=time.perf_counter()
    origin=d.mocap_pos[0].copy();rotation=Rotation.from_quat(d.mocap_quat[0][[1,2,3,0]])
    def sample(phase):
        normals={b:0. for b in fingers};contacts={b:0 for b in fingers};penetration=0.
        for k in range(d.ncon):
            con=d.contact[k];a,b=map(int,m.geom_bodyid[[con.geom1,con.geom2]])
            other=b if a==obj else a if b==obj else -1
            if other not in fingers:continue
            mujoco.mj_contactForce(m,d,k,force)
            normals[other]+=max(0,float(force[0]));contacts[other]+=1
            penetration=max(penetration,-float(con.dist))
        local=d.xmat[grip].reshape(3,3).T@(d.xipos[obj]-d.xpos[grip])
        trace.append({'phase':phase,'t':float(d.time),'object_z':float(d.xipos[obj,2]),'local':local.tolist(),
            'normals_N':list(normals.values()),'contacts':list(contacts.values()),'penetration_m':penetration,
            'actuator_force_N':d.actuator_force.tolist(),'finger_qpos':[float(d.joint('arm_joint7').qpos[0]),float(d.joint('arm_joint8').qpos[0])]})
    def run(phase,seconds,motion=None):
        steps=round(seconds/m.opt.timestep)
        for k in range(steps):
            if motion:motion((k+1)/steps)
            mujoco.mj_step(m,d)
            if k%25==0 or k==steps-1:sample(phase)
    run('open',.4);d.ctrl[:]=[.001,-.001];run('close',.6)
    run('lift',1,lambda u:d.mocap_pos.__setitem__(0,origin+[0,0,.12*(3*u*u-2*u*u*u)]))
    run('hold',hold_seconds)
    def tilt(u):
        q=(Rotation.from_euler('y',90*(3*u*u-2*u*u*u),degrees=True)*rotation).as_quat()
        d.mocap_quat[0]=q[[3,0,1,2]]
    run('tilt',2,tilt);run('horizontal_hold',3)
    def translate(u):d.mocap_pos[0]=origin+[.05*np.sin(4*np.pi*u),0,.12]
    run('translate',2,translate);run('final_hold',2)
    release_result=None
    if release:
        def untilt(u):
            q=(Rotation.from_euler('y',90*(1-(3*u*u-2*u*u*u)),degrees=True)*rotation).as_quat()
            d.mocap_quat[0]=q[[3,0,1,2]]
        run('untilt',2,untilt);run('before_release',.5)
        release_z=float(d.xipos[obj,2]);d.ctrl[:]=[.045,-.045];run('release',1.5)
        release_result={'fall_m':release_z-float(d.xipos[obj,2]),'finger_positions':[float(d.joint('arm_joint7').qpos[0]),float(d.joint('arm_joint8').qpos[0])]}
    summary={}
    for phase in ('close','lift','hold','tilt','horizontal_hold','translate','final_hold'):
        rows=[r for r in trace if r['phase']==phase]
        local=np.array([r['local'] for r in rows])
        summary[phase]={'slip_m':float(np.linalg.norm(local[-1]-local[0])),
            'max_local_change_m':float(np.linalg.norm(local-local[0],axis=1).max()),
            'min_object_z':min(r['object_z'] for r in rows),
            'two_sided_fraction':float(np.mean([min(r['contacts'])>0 for r in rows])),
            'mean_normal_N':np.mean([r['normals_N'] for r in rows],axis=0).tolist(),
            'max_penetration_m':max(r['penetration_m'] for r in rows)}
    return {'settings':settings,'offset':list(offset),'approach_deg':approach_deg,'summary':summary,'trace':trace,'release':release_result,
        'finite_state':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
        'bad_state_warnings':sum(int(d.warning[w].number) for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)),
        'wall_seconds':time.perf_counter()-start_clock}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sweep',action='store_true');args=parser.parse_args()
    cases={'baseline':{}}
    if args.sweep:cases.update({'impratio10':{'impratio':10},'noslip2':{'noslip':2},
        'hard_contact':{'contact':{'solimp':[.99,.999,.001,.5,2],'solref':[.005,1]}},
        'condim6':{'contact':{'condim':6}},'friction2':{'contact':{'friction':[2,.005,.0001]}}})
    results={}
    for name,settings in cases.items():
        results[name]=trial(settings)
        (HERE/'grasp_diagnosis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(name,json.dumps(results[name]['summary']),flush=True)

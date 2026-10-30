"""Shared scene lighting for preview, collection and policy cameras."""
import json
from pathlib import Path
import numpy as np

SETTINGS=Path(__file__).resolve().parent/'lighting_settings.json'

def settings():
    values=json.loads(SETTINGS.read_text(encoding='utf-8'))
    for key in ('headlight_ambient','headlight_diffuse','headlight_specular','scene_ambient','scene_diffuse'):
        v=np.asarray(values[key],dtype=float)
        if v.shape!=(3,) or not np.isfinite(v).all() or np.any(v<0) or np.any(v>1):
            raise ValueError(f'Invalid lighting RGB: {key}')
    return values

def apply_lighting(model,values=None):
    values=values or settings()
    model.vis.headlight.ambient[:]=values['headlight_ambient']
    model.vis.headlight.diffuse[:]=values['headlight_diffuse']
    model.vis.headlight.specular[:]=values['headlight_specular']
    for i in range(model.nlight):
        model.light_ambient[i]=values['scene_ambient']
        model.light_diffuse[i]=values['scene_diffuse']

def apply_lighting_xml(root):
    values=settings();headlight=root.find('visual/headlight')
    for key in ('ambient','diffuse','specular'):headlight.set(key,' '.join(map(str,values['headlight_'+key])))
    for light in root.findall('worldbody/light'):
        for key in ('ambient','diffuse'):light.set(key,' '.join(map(str,values['scene_'+key])))

"""Resolve authored USD colors and selected literal MDL variant defaults.

This is an albedo approximation for MuJoCo, not an MDL interpreter. Procedural
weathering, glass transmission and full PBR remain outside the native renderer.
"""
import hashlib
import re
from pathlib import Path
import numpy as np
from PIL import Image

COLORS = ('diffuse_color_constant', 'diffuseColor', 'base_color_constant',
          'diffuse_reflection_color', 'base_color', 'diffuse_tint', 'mortar_tint')
TEXTURES = ('diffuse_texture', 'diffuse_reflection_color_image', 'base_color_texture', 'file')

def omnipbr_module(shader,defaults):
    source=shader.GetPrim().GetAttribute('info:mdl:sourceAsset').Get()
    if source and Path(source.path).name in ('OmniPBR.mdl','OmniPBR_Opacity.mdl'):return ''
    path=defaults.get('mdl_file')
    text=Path(path).read_text(encoding='utf-8',errors='replace') if path else ''
    return text if re.search(r'::OmniPBR(?:::|_Opacity::)',text) else None

def reflective_facade_albedo(texture,shader,output,readable_path):
    """Diffuse stand-in for metallic facade reflections, using the source ORM.

    The native renderer has no RTX environment reflections. Keep nonmetal/oxide
    colors, approximate the metallic regions with a dark warm environment.
    """
    value=shader.GetInput('ORM_texture').Get()
    if not value:return texture
    orm=readable_path(value.resolvedPath or value.path)
    if not orm.is_file():return texture
    environment=np.array([.055,.05,.035])
    digest=hashlib.sha256((str(texture)+str(orm)+str(environment)+'facade_orm_v1').encode()).hexdigest()[:18]
    target=output/('facade_reflection_'+digest+'.png')
    if not target.exists():
        with Image.open(texture) as image:
            image=image.convert('RGB');image.thumbnail((1024,1024));size=image.size
            base=np.asarray(image,dtype=float)/255.
        with Image.open(orm) as image:packed=np.asarray(image.convert('RGB').resize(size),dtype=float)/255.
        metallic=packed[:,:,2:3];ao=packed[:,:,:1]
        effective=(base*(1-metallic)+environment[None,None,:]*metallic)*ao
        Image.fromarray(np.uint8(np.clip(effective,0,1)*255)).save(target)
    return target


def literal_color(text, name):
    match = re.search(r'\b' + name + r'\s*:\s*color\s*\(([^)]+)\)', text)
    if not match:
        return None
    try:
        values = [float(v.strip().rstrip('f')) for v in match.group(1).split(',')]
        return values * 3 if len(values) == 1 else values if len(values) == 3 else None
    except ValueError:
        return None


def variant_values(defaults, variant):
    """Select this export, rather than the first color anywhere in its module."""
    path = defaults.get('mdl_file')
    if not path or not variant:
        return {}
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    match = re.search(r'export material\s+' + re.escape(str(variant)) + r'\b', text)
    if not match:
        return {}
    section = text[match.end():].split('export material', 1)[0]
    body = re.search(r'\]\]\s*=\s*\w+\s*\((.*?)\)\s*;', section, re.S)
    if not body:
        return {}
    body = body.group(1)
    result = {name: value for name in COLORS if (value := literal_color(body, name)) is not None}
    for name in ('diffuse_brightness', 'brightness_adjust', 'roughness'):
        value = re.search(r'\b' + name + r'\s*:\s*([-+\d.eE]+)f?', body)
        if value:
            result[name] = float(value.group(1))
    return result


def resolve(shader, defaults, variant, readable_path, output):
    color = np.array(defaults.get('color', [.65, .67, .69]), dtype=float)
    has_color = 'color' in defaults
    texture = readable_path(defaults['diffuse_texture']) if defaults.get('diffuse_texture') else None
    values = variant_values(defaults, variant)
    for name in COLORS:
        if name in values:
            color = np.array(values[name]);has_color = True;break
    for name in COLORS:
        value = shader.GetInput(name).Get()
        if value is not None and hasattr(value, '__len__') and len(value) == 3:
            color = np.array(value, dtype=float);has_color = True;break
    for name in TEXTURES:
        value = shader.GetInput(name).Get()
        if value and hasattr(value, 'path'):
            path = readable_path(value.resolvedPath or value.path)
            if path.is_file():texture = path;break
    pbr=omnipbr_module(shader,defaults)
    if texture is not None and pbr is not None:
        # OmniPBR uses the map OR diffuse_color_constant. diffuse_tint is the
        # map multiplier; multiplying a map by the fallback .2/.5 darkens it.
        tint=shader.GetInput('diffuse_tint').Get()
        if tint is None:tint=values.get('diffuse_tint',literal_color(pbr,'diffuse_tint'))
        color=np.array(tint if tint is not None else [1.,1.,1.]);has_color=True
    if defaults.get('raw_diffuse_texture'):
        # Plaster's packed B channel is diffuse. Bake the selected variant once.
        tint = color if has_color else np.array(defaults.get('bake_tint', [1., 1., 1.]))
        authored = shader.GetInput('diffuse_brightness').Get()
        brightness = float(authored if authored is not None else values.get('diffuse_brightness', defaults.get('bake_brightness', 1.)))
        raw = readable_path(defaults['raw_diffuse_texture'])
        digest = hashlib.sha256((str(raw) + str(tint) + str(brightness)).encode()).hexdigest()[:18]
        texture = output / ('albedo_' + digest + '.png')
        if not texture.exists():
            with Image.open(raw) as image:
                image = image.convert('RGB');image.thumbnail((1024, 1024))
                albedo = np.asarray(image, dtype=float)[:, :, 2:3] / 255.
            linear = np.clip(albedo * tint * brightness, 0, 1)
            srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
            Image.fromarray(np.uint8(srgb * 255)).save(texture)
        color = np.ones(3)
    elif 'mortar_tint' in values:
        # Mortar's R/G/B map is roughness/AO/noise, not an RGB diffuse image.
        # Preserve the selected warm wall tint with a flat albedo approximation.
        texture = None
    elif texture is not None and not has_color:
        color = np.ones(3)
    return np.clip(color, 0, 1), texture, values

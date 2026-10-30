"""Restore same-name MDL defaults/textures from NVIDIA's official public bucket.

Do not execute MDL or modify USD. Store provenance and SHA256 for downloaded files.
"""
import concurrent.futures,hashlib,json,re,time,urllib.parse,urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from build_hotel import Builder,HERE
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/'
DEST=HERE/'material_library';DEST.mkdir(exist_ok=True)
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

def listing(prefix):
    result=[];token=None
    for _ in range(100):
        query={'list-type':'2','prefix':prefix,'max-keys':'1000'}
        if token:query['continuation-token']=token
        data=urllib.request.urlopen(BASE+'?'+urllib.parse.urlencode(query),timeout=30).read()
        root=ET.fromstring(data)
        result.extend(x.text for x in root.findall('s:Contents/s:Key',NS) if x.text.endswith('.mdl'))
        token=root.findtext('s:NextContinuationToken',namespaces=NS)
        if not token:break
    print('Indexed',prefix,len(result),'MDL files',flush=True)
    return result

def download(url):
    suffix=Path(urllib.parse.urlparse(url).path).suffix
    target=DEST/(hashlib.sha256(url.encode()).hexdigest()[:24]+suffix)
    if not target.exists():
        data=urllib.request.urlopen(url,timeout=60).read()
        temp=target.with_suffix(target.suffix+'.part');temp.write_bytes(data);temp.replace(target)
    return target

def load_material(item):
    name,key=item;url=BASE+urllib.parse.quote(key,safe='/')
    try:
        path=download(url);text=path.read_text(encoding='utf-8',errors='replace')
        record={'mdl_url':url,'mdl_file':str(path),'mdl_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for parameter in ('diffuse_texture','base_color_texture','albedo_texture','color_texture'):
            match=re.search(r'\b'+parameter+r'\s*:\s*texture_2d\s*\(\s*"([^\"]+)"',text)
            if match:break
        else:
            match=next((m for m in re.finditer(r'texture_2d\s*\(\s*"([^\"]+)"',text) if re.search(r'basecolor|albedo|diffuse|_diff[._]|_col[._]',m.group(1),re.I)),None)
        if match is None:
            match=re.search(r'texture_2d\s*\(\s*"([^\"]+)"\s*,\s*::tex::gamma_srgb',text)
        if match:
            texture_url=urllib.parse.urljoin(url,match.group(1))
            image=download(texture_url)
            record.update(diffuse_texture=str(image),texture_url=texture_url,texture_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            if '_multi_r_ao_g_grunge_b_diff' in texture_url:
                # This source texture packs AO/grunge/diffuse into RGB; it is not
                # an RGB albedo. Read the actual MDL tint/brightness defaults.
                tint_match=re.search(r'color\s+diffuse_tint\s*=\s*color\(([^)]+)\)',text)
                tint=[float(v.strip().rstrip('f')) for v in tint_match.group(1).split(',')] if tint_match else [1.,1.,1.]
                if len(tint)==1:tint*=3
                brightness_match=re.search(r'float\s+diffuse_brightness\s*=\s*([0-9.]+)f?',text)
                brightness=float(brightness_match.group(1)) if brightness_match else 1.
                baked=image.with_name(image.stem+'_diffuse_baked.png')
                with Image.open(image) as raw:
                    raw=raw.convert('RGB');raw.thumbnail((1024,1024))
                    albedo=np.asarray(raw,dtype=np.float32)[:,:,2:3]/255
                linear=np.clip(albedo*np.array(tint)[None,None,:]*brightness,0,1)
                srgb=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)
                Image.fromarray(np.uint8(np.clip(srgb,0,1)*255)).save(baked)
                record.update(diffuse_texture=str(baked),raw_diffuse_texture=str(image),bake='Source B=diffuse channel, MDL diffuse_tint and diffuse_brightness; procedural weathering approximated.',bake_tint=tint,bake_brightness=brightness)
        for parameter in ('diffuse_color_constant','diffuse_tint','base_color','surface_color'):
            match=re.search(r'\b'+parameter+r'\s*:\s*color\s*\(([^)]+)\)',text)
            if match:
                try:
                    values=[float(v.strip().rstrip('f')) for v in match.group(1).split(',')]
                    if len(values)==1:values*=3
                    if len(values)==3:record['color']=values;break
                except ValueError:pass
        print('Restored',name,'texture',bool(record.get('diffuse_texture')),flush=True)
        return name,record
    except Exception as exc:return name,{'error':str(exc),'mdl_url':url}

def main():
    b=Builder();needed=set()
    for prim in b.prims:
        value=prim.GetAttribute('info:mdl:sourceAsset').Get()
        if value and hasattr(value,'path'):needed.add(Path(value.path).name)
    catalog_file=DEST/'nvidia_mdl_catalog.json'
    if catalog_file.exists():keys=json.loads(catalog_file.read_text())
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            keys=sum(pool.map(listing,['Materials/Base/','Materials/vMaterials_2/']),[])
        catalog_file.write_text(json.dumps(keys,indent=2))
    chosen={}
    for key in keys:
        name=Path(key).name
        if name in needed and name not in chosen:chosen[name]=key
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        defaults=dict(pool.map(load_material,chosen.items()))
    out={'source':'NVIDIA official public material library; exact authored MDL basenames','updated':time.time(),'materials':defaults,
        'not_found':sorted(needed-set(defaults))}
    (HERE/'material_defaults.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',len(defaults),'matched;',len(out['not_found']),'unavailable',flush=True)

if __name__=='__main__':main()

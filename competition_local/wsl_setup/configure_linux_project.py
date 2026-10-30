"""Prepare an independent OpenPI copy for Blackwell JAX + CPU-only PyTorch."""
import argparse,json,re,tomllib
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
    original=args.project/'pyproject.toml';target=args.output or original
    text=original.read_text(encoding='utf-8')
    marker='# PPSUC local Blackwell environment: JAX owns CUDA; PyTorch is data-loading only.'
    if marker not in text:
        config=tomllib.loads(text)
        values=config['tool']['uv'].get('override-dependencies',[])
        values=[v for v in values if not v.startswith('nvidia-')]
        values+=['nvidia-cuda-runtime-cu12>=12.8,<13','nvidia-cuda-nvcc-cu12>=12.8,<13',
                 'nvidia-cuda-cupti-cu12>=12.8,<13','nvidia-cublas-cu12>=12.8,<13',
                 'nvidia-cudnn-cu12>=9.8,<10','nvidia-nvjitlink-cu12>=12.8,<13']
        text,count=re.subn(r'(?m)^override-dependencies\s*=\s*\[[^\]]*\]', 'override-dependencies = '+json.dumps(values),text,count=1)
        if count!=1:raise RuntimeError('Expected one existing uv override list')
        if not any(v.startswith('torchvision') for v in config['project']['dependencies']):
            text=text.replace('dependencies = [\n','dependencies = [\n    "torchvision==0.22.1",\n',1)
        text=text.replace('[tool.uv.sources]\n','[tool.uv.sources]\ntorch = { index = "pytorch-cpu" }\ntorchvision = { index = "pytorch-cpu" }\n',1)
        text+='\n'+marker+'\n[[tool.uv.index]]\nname = "pytorch-cpu"\nurl = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n'
    parsed=tomllib.loads(text)
    assert parsed['tool']['uv']['sources']['torch']['index']=='pytorch-cpu'
    assert parsed['tool']['uv']['sources']['torchvision']['index']=='pytorch-cpu'
    assert 'jax[cuda12]==0.5.3' in parsed['project']['dependencies']
    assert 'flax==0.10.2' in parsed['project']['dependencies']
    if target==original:
        backup=original.with_suffix('.original.toml')
        if not backup.exists():backup.write_bytes(original.read_bytes())
    target.write_text(text,encoding='utf-8')
    print(json.dumps({'prepared':str(target),'source_preserved':str(original),'jax':'0.5.3','flax':'0.10.2','torch':'2.7.1 CPU-only','cuda_runtime_min':'12.8','cudnn_min':'9.8'},indent=2))

if __name__=='__main__':main()

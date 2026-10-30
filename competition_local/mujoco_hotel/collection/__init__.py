"""MuJoCo collection using the original project's episode and stage contract."""
import os,sys
from pathlib import Path

_deps=str(Path(__file__).resolve().parents[2]/('collection_deps' if os.name=='nt' else 'collection_deps_wsl'))
if _deps not in sys.path:sys.path.insert(0,_deps)

"""JPEG-in-HDF5 image storage with transparent RGB reads and legacy support."""
import io
from contextlib import contextmanager
from collections.abc import Mapping
import h5py
import numpy as np
from PIL import Image

IMAGE_SHAPE=(480,640,3)
DEFAULT_JPEG_QUALITY=95


class JPEGDataset:
    def __init__(self,dataset):
        self.raw=dataset
        if (dataset.ndim!=1 or dataset.attrs.get('encoding')!='jpeg'
                or tuple(dataset.attrs.get('image_shape',()))!=IMAGE_SHAPE
                or h5py.check_dtype(vlen=dataset.dtype)!=np.dtype('uint8')):
            raise ValueError(f'Invalid compressed RGB dataset: {dataset.name}')

    @property
    def shape(self):return (len(self.raw),*IMAGE_SHAPE)
    @property
    def dtype(self):return np.dtype('uint8')
    def __len__(self):return len(self.raw)
    def resize(self,size,axis=0):self.raw.resize(size,axis=axis)

    def __getitem__(self,index):
        if not isinstance(index,(int,np.integer)):
            indices=np.arange(len(self))[index]
            return np.stack([self[int(i)] for i in indices]) if len(indices) else np.empty((0,*IMAGE_SHAPE),np.uint8)
        with Image.open(io.BytesIO(self.raw[index].tobytes())) as im:
            if im.format!='JPEG' or im.size!=(640,480):raise ValueError('Invalid encoded camera frame')
            return np.asarray(im.convert('RGB'),dtype=np.uint8).copy()

    def __setitem__(self,index,rgb):
        rgb=np.asarray(rgb)
        if rgb.shape!=IMAGE_SHAPE or rgb.dtype!=np.uint8:raise ValueError('Expected RGB uint8[480,640,3]')
        stream=io.BytesIO()
        Image.fromarray(rgb).save(stream,format='JPEG',quality=int(self.raw.attrs['jpeg_quality']),subsampling=0)
        self.raw[index]=np.frombuffer(stream.getvalue(),np.uint8)


def image_dataset(dataset):
    return JPEGDataset(dataset) if dataset.ndim==1 else dataset


class ImageGroup(Mapping):
    def __init__(self,raw):self.raw=raw
    def __getitem__(self,name):return image_dataset(self.raw[name])
    def __iter__(self):return iter(self.raw)
    def __len__(self):return len(self.raw)


class EpisodeFile:
    """File facade used by the source writer and all local image consumers."""
    def __init__(self,raw,quality=None):
        self.raw=raw;self.quality=quality
        if quality is not None and not 1<=quality<=100:raise ValueError('JPEG quality must be 1..100')

    def __getitem__(self,key):
        key=key.lstrip('/')
        if key=='images':return ImageGroup(self.raw[key])
        if key.startswith('images/'):return image_dataset(self.raw[key])
        return self.raw[key]
    def __contains__(self,key):return key in self.raw
    def __getattr__(self,key):return getattr(self.raw,key)

    def create_dataset(self,name,*args,**kwargs):
        if self.quality is None or not name.startswith('images/'):
            return self.raw.create_dataset(name,*args,**kwargs)
        ds=self.raw.create_dataset(name,shape=(0,),maxshape=(None,),chunks=(128,),dtype=h5py.vlen_dtype(np.uint8))
        ds.attrs['encoding']='jpeg';ds.attrs['jpeg_quality']=self.quality
        ds.attrs['jpeg_subsampling']=0;ds.attrs['image_shape']=IMAGE_SHAPE
        return JPEGDataset(ds)


@contextmanager
def open_episode(path):
    with h5py.File(path,'r') as raw:yield EpisodeFile(raw)

import glob
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from typing import Any, List, Union, Callable
try:
    import pyspng
except ImportError:
    pyspng = None

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import webdataset as wds
from torch.utils.data import default_collate
from braceexpand import braceexpand
import math
import random
from torchvision import transforms
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import sys
from multiprocessing import Value
from ambient_utils.utils import save_images, save_image

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import open_clip



#----------------------------------------------------------------------------
# Abstract base class for datasets.

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        normalize=True,
        only_positive=True,  # whether to return images in [0, 1] or [-1, 1]
        use_other_keys = False, # Load additional keys from dataset.json?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._use_other_keys = use_other_keys
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self.normalize = normalize
        self.only_positive = only_positive

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels
    

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            labels = data.get('labels')
            if labels is None:
                return None
            labels = dict(labels)
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
            labels = np.array(labels)
            labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def load_additional_keys(self):
        """Load additional keys from dataset.json if they exist."""
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            # Exclude 'labels' key and load other keys
            self._other_keys_data = {key: data[key] for key in data if key != 'labels'}
        return self._other_keys_data

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8  # removed to support latent diffusion models
        
        # get array that masks each pixel with probability self.corruption_probability_per_pixel with fixed seed for reproducibility
        np.random.seed(raw_idx)
        torch.manual_seed(raw_idx)
        if self.normalize:
            if self.only_positive:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32) / 127.5 - 1

        return {
            "image": image.copy(),
            "label": self.get_label(idx),
            "raw_idx": raw_idx,
            # this fixes a noise realization per image in the dataset. It is useful for ambient training, but can be ignored otherwise.
            "noise": np.random.randn(*image.shape),
            "filename": self._image_fnames[raw_idx],
            **self.get_other_keys(raw_idx),
        }

    def get_by_filename(self, filename):
        idx = self._image_fnames.index(filename)
        return self[idx]

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()
    
    def get_other_keys(self, idx):
        if self._use_other_keys:
            return {key: self.other_keys_data[key][self._raw_idx[idx]] for key in self.other_keys_data}
        else:
            return {}

    def get_details(self, idx):
        d = EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        d.filename = self._image_fnames[d.raw_idx]
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        must_contain    = None, # Require filenames to contain this substring.
        must_not_contain = None, # Require filenames to NOT contain this substring.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        if must_contain is not None:
            self._all_fnames = {fname for fname in self._all_fnames if must_contain in fname}
        
        if must_not_contain is not None:
            self._all_fnames = {fname for fname in self._all_fnames if must_not_contain not in fname}

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in (PIL.Image.EXTENSION.keys() | {'.npy'}))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

        if self._use_other_keys:
            self.load_additional_keys()

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.npy':
                image = np.load(f)
            elif self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        if self._file_ext(fname) != '.npy':
            image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            labels = data.get('labels')
            if labels is None:
                return None
            labels = dict(labels)
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
            labels = np.array(labels)
            labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def load_additional_keys(self):
        """Load additional keys from dataset.json if they exist."""
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            # Exclude 'labels' key and load other keys
            self._other_keys_data = {key: data[key] for key in data if key != 'labels'}
        return self._other_keys_data

    @property
    def other_keys_data(self):
        """Return additional keys data if use_other_keys is set."""
        if self._use_other_keys:
            return getattr(self, '_other_keys_data', None)
        return None


class SyntheticallyCorruptedImageFolderDataset(ImageFolderDataset):
    def __init__(self, corruption_probability: float = 0.5, 
                 corruptions_dict: EasyDict = {},
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_probability = corruption_probability
        self.corruptions_dict = corruptions_dict
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = item["image"]
        item["original_image"] = image.copy()
        
        # fix seed to idx
        np.random.seed(idx)
        torch.manual_seed(idx)

        if np.random.random() < self.corruption_probability:
            item["corruption_label"] = 1
            # pick one of the corruptions
            corruption_name = np.random.choice(list(self.corruptions_dict.keys()))
            corruption_fn = getattr(__import__('ambient_utils.noise'), "apply_" + corruption_name)
            # clone numpy array image to avoid in-place corruption
            # materialize params
            corruption_params = {key: f() for key, f in self.corruptions_dict[corruption_name].items()}
            item["image"] = corruption_fn(item["image"], **corruption_params)
        else:
            item["corruption_label"] = 0
        return item



#----------------------------------------------------------------------------

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)

def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights

class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

def image_transform(
    image_size: Union[int, Tuple[int, int]],
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_mode: Optional[str] = None,
    interpolation: Optional[str] = None,
    fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    normalize = Normalize(mean=mean, std=std)
    if resize_mode == 'longest':
        transforms = [
            ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
            CenterCropOrPad(image_size, fill=fill_color)
        ]
    elif resize_mode == 'squash':
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transforms = [
            Resize(image_size, interpolation=interpolation_mode),
        ]
    else:
        assert resize_mode == 'shortest'
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        if image_size[0] == image_size[1]:
            # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
            transforms = [
                Resize(image_size[0], interpolation=interpolation_mode)
            ]
        else:
            # resize shortest edge to matching target dim for non-square target
            transforms = [ResizeKeepRatio(image_size)]
        transforms += [CenterCrop(image_size)]

    transforms.extend([
        lambda image: image.convert("RGB"),
        ToTensor(),
        # normalize,
    ])
    return Compose(transforms)

def get_wds_dataset(input_shards, batch_size, is_train=False, 
                    epoch=0, floor=False, num_samples=10000, seed=42, 
                    annotate_fn=None, annotation_keys=[],
                    workers=1, world_size=1):
    resampled = is_train

    num_shards = None
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
   
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=None,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    def get_original_dims(sample):
        sample["original_dims"] = sample["image"].size
        return sample    

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map(get_original_dims),
        wds.map_dict(image=image_transform(image_size=224), text=lambda text: text),
        wds.map(annotate_fn if annotate_fn else lambda x: x),
        wds.to_tuple("image", "text", "original_dims", *annotation_keys),
        wds.batched(batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= workers * world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return dataloader


def clip_annotate_wds(label_texts=None):
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text_features = None

    if label_texts is not None:
        tokenized_labels = tokenizer(label_texts)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_labels)
            text_features /= text_features.norm(dim=-1, keepdim=True)


    def annotate_fn(sample, text_features=None):
        image = sample["image"]

        if text_features is None:
            tokenized_labels = tokenizer(sample["text"])
            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized_labels)
                text_features /= text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            image_features = clip_model.encode_image(image.unsqueeze(0))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            inner_products = (image_features @ text_features.T)
            clip_text_probs = (100.0 * inner_products).softmax(dim=-1)[:, 0]
        sample["clip_text_probs"] = clip_text_probs
        sample["inner_products"] = inner_products
        return sample

    return lambda x: annotate_fn(x, text_features)




def concat_shards_in_path(shards_path):
    shards = glob.glob(os.path.join(shards_path, "*.tar"))
    shards = [os.path.join(shards_path, shard) for shard in shards]
    return "::".join(shards)

if __name__ == "__main__":

    BATCH_SIZE = 256
    NUM_SAMPLES = 10000
    SEED = 42
    WORKERS = 256
    K = 5
    BEST = True

    # demonstration of how to use the webdataset dataloader
    shards = concat_shards_in_path(os.environ.get("DATACOMP_SMALL", "./"))
    # annotate_fn = clip_annotate_wds(["low-quality image", "high-quality image"])
    annotate_fn = clip_annotate_wds()

    dataloader = get_wds_dataset(shards, batch_size=BATCH_SIZE, is_train=True, epoch=0, 
                                 floor=False, num_samples=NUM_SAMPLES, seed=SEED,
                                 # annotation params
                                 annotate_fn=annotate_fn, annotation_keys=["clip_text_probs", "inner_products"],
                                 # hardware params
                                 workers=WORKERS, world_size=1)
    image, text, original_dims, text_probs, inner_products = next(iter(dataloader))
    save_images(image * 2 - 1, "test_webdataset.png")


    # get top-K images
    sorted_indices = inner_products.squeeze(1, 2).argsort(dim=0)
    sorted_indices = sorted_indices[:K] if BEST else sorted_indices[-K:]
    top_K_images = image[sorted_indices].squeeze(1)
    top_K_texts = [text[i] for i in sorted_indices]
    save_images(top_K_images * 2 - 1, "test_top_K_images.png")
    import pdb; pdb.set_trace()
    # # demonstration of how to use the synthetically corrupted image folder dataset

    # corruptions_dict = {
    #     "blur": {
    #         "sigma": lambda: np.random.uniform(0.5, 8.0)
    #     },
    #     "mask": {
    #         "masking_probability": lambda: np.random.uniform(0.05, 0.9)
    #     },
    #     "pixelate": {
    #         "pixel_size": lambda: np.random.randint(10, 200)
    #     },
    #     "saturation": {
    #             # Start of Selection
    #             "saturation_level": lambda: np.random.uniform(0.8, 0.9) if np.random.rand() < 0.5 else np.random.uniform(1.1, 1.2)
    #     },
    #     "color_shift": {
    #         "shift": lambda: np.random.uniform(-0.1, -0.05, size=3) if np.random.rand() < 0.5 else np.random.uniform(0.05, 0.1, size=3)
    #     },
    #     "imagecorruptions": {
    #         "severity": lambda: np.random.randint(1, 5),
    #         # "corruption_name": lambda: np.random.choice(["jpeg_compression"])
    #         "corruption_name": lambda: np.random.choice(["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "motion_blur", "zoom_blur", "snow", "frost", "brightness", "contrast", "elastic_transform", "jpeg_compression"])
    #     }
    # }
    # # TODO(@giannisdaras): de-anonymize this path
    # dataset = SyntheticallyCorruptedImageFolderDataset(path="/scratch/07362/gdaras/datasets/ffhq-256x256_train_split", 
    #                                                     corruption_probability=1.0, 
    #                                                     corruptions_dict=corruptions_dict)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # save_images(next(iter(dataloader))["image"] * 2 - 1, "test_synthetically_corrupted.png")

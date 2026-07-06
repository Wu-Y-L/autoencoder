from archived.diffusion_model_22_2.visualizing import device
import torch
import tifffile
from pathlib import Path
import torchvision 
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key, loader_fn):
        if key in self.cache:
            # move to end (most recent)
            self.cache.move_to_end(key)
            return self.cache[key]
        # load
        value = loader_fn(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)  # pop oldest
        return value

    def clear(self):
        self.cache.clear()

# assigns an index to all file paths and returns a tuple
def _append_files(path, tif: bool = True):

    # read only the stack metadata (no pixel decode) to count slices. This is
    # fast enough to run sequentially, so we no longer need joblib -- a joblib
    # pool created here leaves executor state that breaks the DataLoader's
    # worker processes when they fork.
    with tifffile.TiffFile(path) as handle:
        images_per_file = handle.series[0].shape[0]

    return [(path, i) for i in range(images_per_file)]

def _normalize_per_slice(
    image: torch.Tensor,
    pmin: float = 2.0,
    pmax: float = 99.8,
    dtype=torch.float32,
    eps=1e-08,
    clip=True,
):
    # vectorizes torch tensor to avoid python loop
    image = image.to(dtype=dtype)
    H, W = image.shape
    flat = image.flatten()                        # ( H*W)
    mi = torch.quantile(flat, pmin / 100.0)   # (1)
    ma = torch.quantile(flat, pmax / 100.0)   #
    denom = ma - mi + eps 
    

    normalized = (image - mi) / denom
    
    if clip:
        return torch.clamp(normalized, 0.0, 1.0), mi, ma 

    return normalized, mi, ma 

class load_vae_dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_dir : Path, 
        transform : torchvision.transforms = None, 
        n_jobs = -1, n_patches : int = 8, 
        patch_size : int = 512, 
        cache_size : int = 10,
        device : str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        """
            input_dir is an folder containing tifffiles of image stacks
        """

        self.input_dir = input_dir
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.transform = transform
        self.cache_size = cache_size 
        self.cache = OrderedDict()
        self.device = device 


        self.samples = [] 

        all_files = sorted(list((self.input_dir).glob("*.tif")))

        # sequential, metadata-only index build. No joblib/loky pool here, so
        # nothing conflicts with the DataLoader forking its num_workers > 0
        # worker processes.
        for path in all_files:
            for tup in _append_files(path):
                for _ in range(n_patches):
                    self.samples.append(tup)
    
    def _load_stack(self, file_path):
        stack = tifffile.imread(file_path)
        return torch.from_numpy(stack).float()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        files, idx = self.samples[index]
        
        # LRU cache: load stack if not already cached
        if files not in self.cache:
            # If cache is full, remove oldest item
            if len(self.cache) >= self.cache_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[files] = self._load_stack(files)
        else:
            # Move to end (most recently used)
            self.cache.move_to_end(files)
        stack = self.cache[files]

        _temp = stack[idx]

        current_img, _, _ = _normalize_per_slice(_temp, pmin = 1.0)

        H, W = current_img.shape
        
        # prevent negative

        assert H >= self.patch_size, " image is smaller than patchsize"

        patch_height_idx = torch.randint(0,  H - self.patch_size + 1, size = (1,)).item()
        patch_width_idx = torch.randint(0,  W - self.patch_size + 1 , size = (1,)).item()

        current_patch = current_img[patch_height_idx : patch_height_idx + self.patch_size, patch_width_idx : patch_width_idx + self.patch_size]

        current_patch = current_patch.unsqueeze(0)

        if self.transform:
            # random seed for transformation 
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            current_patch = self.transform(current_patch)

        return current_patch

def _append_files_with_gt(path, tif: bool = True):

    # read only the stack metadata (no pixel decode) to count slices. This is
    # fast enough to run sequentially, so we no longer need joblib -- a joblib
    # pool created here leaves executor state that breaks the DataLoader's
    # worker processes when they fork.
    with tifffile.TiffFile(path) as handle:
        images_per_file = handle.series[0].shape[0] - 1

    return [(path, i) for i in range(images_per_file)]

class ConditionalPatchDataset(torch.utils.data.Dataset):
    """Yields (input_patch, gt_patch), each [1, 512, 512] in [0,1] (VAE domain).
       Stack layout: [input_0, ..., input_{n-1}, GT]  -- last slice is ground truth."""
    def __init__(self, data_dir, patch_size=512, n_patches=8, transform=None, cache_size=2, device : str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.transform = transform
        self.cache = OrderedDict(); self.cache_size = cache_size
        self.samples = []
        self.device = device 
        # build index sequentially (NO joblib): for each tif, register every NON-GT slice
        #   for path in sorted(data_dir.glob("*/*.tif")):
        #       n_slices = <read metadata only, like vae_dataset_loader._append_files>
        #       for input_idx in range(n_slices - 1):     # -1 excludes the GT slice
        #           for _ in range(n_patches): self.samples.append((path, input_idx))
        
        all_files = sorted(list(self.data_dir.glob("*.tif")))

        for path in all_files:
            for tup in _append_files_with_gt(path):
                for _ in range(n_patches):
                    self.samples.append((tup))

    def _load_stack(self, file_path):
        stack = tifffile.imread(file_path)
        return torch.from_numpy(stack).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        files, input_idx = self.samples[index]
            
        # LRU cache: load stack if not already cached
        if files not in self.cache:
            # If cache is full, remove oldest item
            if len(self.cache) >= self.cache_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[files] = self._load_stack(files)
        else:
            # Move to end (most recently used)
            self.cache.move_to_end(files)

        stack = self.cache[files] # [n_slices, H, W]

        inp = stack[input_idx]
        gt = stack[-1]

        inp, _, _ = _normalize_per_slice(inp, pmin = 1.0)
        gt, _, _ = _normalize_per_slice(gt, pmin = 1.0)

        H, W = inp.shape

        # .item() to get the integers out from randint 
        H_idx = torch.randint(
            0, H - self.patch_size + 1, size = (1,)
        ).item()
        W_idx = torch.randint(
            0, W - self.patch_size + 1, size = (1,)
        ).item()

        inp = inp[H_idx: H_idx + self.patch_size, W_idx : W_idx + self.patch_size]
        gt = gt[H_idx: H_idx + self.patch_size, W_idx : W_idx + self.patch_size]
        
        # add channel dim 
        inp = inp.unsqueeze(0)
        gt = gt.unsqueeze(0) 

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            inp = self.transform(inp)
            torch.manual_seed(seed)
            gt = self.transform(gt)

        return inp, gt




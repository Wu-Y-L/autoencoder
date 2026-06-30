import torch 
from joblib import Parallel, delayed
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

    temp_list = []
    
    _temp = tifffile.imread(path)
    (images_per_file, _, _) = _temp.shape
    for i in range(images_per_file):
        temp_list.append((path, i))

    return temp_list

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
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_append_files)(path) for path in all_files
        )

        for x_list in results: 
            for tup in x_list:
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


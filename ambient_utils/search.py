from abc import ABC, abstractmethod
import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
from ambient_utils.dataset import ImageFolderDataset
import os
from torch.utils.data import DataLoader
import ambient_utils
import torch.nn.functional as F
from ambient_utils.loss import from_noise_pred_to_x0_pred_ve



def split_image_into_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split an image into patches.
    
    Args:
        image: torch.Tensor of shape [B, C, H, W]
    Returns:
        patches: torch.Tensor of shape [B, C, H // patch_size, W // patch_size, patch_size, patch_size]
    """
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches

def split_image_into_pixel_level_patches(image: torch.Tensor, patch_size: int, keep_ratio: float = 1.0) -> torch.Tensor:
    """
    Split an image into pixel-level patches.
    
    Creates a patch centered on each pixel in the image. For pixels near the edges,
    the patch is padded with zeros or reflected values to maintain the patch size.
    If patch_size is even, it will be incremented by 1 to make it odd for proper centering.
    
    Args:
        image: torch.Tensor of shape [B, C, H, W]
        patch_size: Size of each patch (will be made odd if even)
    Returns:
        patches: torch.Tensor of shape [B, C, H, W, patch_size, patch_size]
    """
    # Make patch_size odd if it's even
    if patch_size % 2 == 0:
        patch_size += 1
    
    half_patch = patch_size // 2
    
    # Pad the image to handle edge cases
    # Use reflection padding to avoid artifacts at edges
    padded_image = F.pad(image, (half_patch, half_patch, half_patch, half_patch), mode='reflect')

    
    # Use unfold to extract all patches at once
    # First unfold in height dimension
    patches_h = padded_image.unfold(2, patch_size, 1)  # [B, C, H, W, patch_size]

    dropped_h = torch.rand(patches_h.shape[2]) < keep_ratio 
    patches_h = patches_h[:, :, dropped_h, :]


    # Then unfold in width dimension
    patches = patches_h.unfold(3, patch_size, 1)  # [B, C, H, W, patch_size, patch_size]

    dropped_w = torch.rand(patches.shape[3]) < keep_ratio 
    patches = patches[:, :, :, dropped_w]
    
    # Rearrange dimensions to match expected output format
    patches = patches.permute(0, 1, 2, 3, 4, 5)  # [B, C, H, W, patch_size, patch_size]
    
    return patches

def assemble_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Assemble patches into an image.
    
    Args:
        patches: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size]
    Returns: 
        image: torch.Tensor of shape [B, C, H, W]
    """
    B, num_patches_per_row, num_patches_per_col, C, patch_size, _ = patches.shape
    image = patches.permute(0, 3, 1, 4, 2, 5)
    image = image.reshape(B, C, num_patches_per_row * patch_size, num_patches_per_col * patch_size)
    return image


class BaseFAISS(ABC):
    """
    Base class for FAISS-based approximate nearest neighbors search.
    
    This abstract base class provides common functionality for both
    disk-based and in-memory FAISS implementations.
    """
    
    def __init__(self, use_gpu: bool = True, 
                 index_type: str = 'ivf', device: torch.device = None,
                 dtype: torch.dtype = torch.float32,
                 num_clusters: int = 4096):
        """
        Initialize base FAISS neighbors search.
        
        Args:
            use_gpu: Whether to use GPU acceleration for FAISS
            device: torch device for tensor operations
            dtype: data type for tensors
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
        """
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.index = None
        self.num_clusters = num_clusters


    def _create_index(self, dataset_dim: int, pca_dim: int = None):
        """Create FAISS index based on the specified type."""
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatL2(dataset_dim)
        elif self.index_type == 'ivf':
            nlist = self.num_clusters
            quantizer = faiss.IndexFlatL2(dataset_dim)
            self.index = faiss.IndexIVFFlat(quantizer, dataset_dim, nlist)
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dataset_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        if pca_dim is not None:
            pca_matrix = faiss.PCAMatrix(dataset_dim, pca_dim, 0, True)
            pca_index = faiss.IndexPreTransform(pca_matrix, self.index)
            # make sure the distances are computed wrt the true images
            self.index = faiss.IndexRefineFlat(pca_index)
        
 
    def _check_gpu_availability(self):
        """Check FAISS GPU availability and provide diagnostics."""
        try:
            num_gpus = faiss.get_num_gpus()
            print(f"FAISS reports {num_gpus} GPU(s) available")
            
            if num_gpus == 0:
                print("No FAISS GPU support detected. This could be due to:")
                print("  - FAISS compiled without GPU support")
                print("  - No CUDA installation")
                print("  - No compatible GPU drivers")
                return False
                
            # Try to create GPU resources to test availability
            try:
                res = faiss.StandardGpuResources()
                print("FAISS GPU resources created successfully")
                return True
            except Exception as e:
                print(f"Failed to create FAISS GPU resources: {e}")
                return False
                
        except Exception as e:
            print(f"Error checking FAISS GPU availability: {e}")
            return False

    def _move_to_gpu(self):
        """Move FAISS index to GPU if requested and available."""
        if not self.use_gpu:
            return
            
        if not self._check_gpu_availability():
            print("Falling back to CPU")
            self.use_gpu = False
            return
            
        try:
            # Check if index is already on GPU
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                return
                
            # Create GPU resources
            res = faiss.StandardGpuResources()
            
            # Try different GPU cloner configurations, starting with safer options
            # The interleaved_layout=True can cause issues with certain index types
            configs_to_try = [
                {"useFloat16": True, "interleaved_layout": True},    # Most aggressive
                {"useFloat16": True, "interleaved_layout": False},   # Float16 but no interleaving
                {"useFloat16": False, "interleaved_layout": True},   # Interleaving but no float16
                {"useFloat16": False, "interleaved_layout": False},  # Safest option
            ]
            
            for i, config in enumerate(configs_to_try):
                try:                    
                    # Configure GPU cloner options
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = config["useFloat16"]
                    co.interleaved_layout = config["interleaved_layout"]
                    
                    # Try to move index to GPU
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    
                    # Verify the move was successful
                    if hasattr(gpu_index, 'getDevice') and gpu_index.getDevice() >= 0:
                        self.index = gpu_index
                        return
                    else:
                        print(f"GPU move failed verification with config {i+1}")
                        
                except Exception as e:
                    error_msg = str(e)
                    print(f"GPU configuration {i+1} failed: {error_msg}")
                    
                    # If we get a core dump or serious error, stop trying
                    if "Aborted" in error_msg or "core dumped" in error_msg:
                        print("Serious error detected, stopping GPU attempts")
                        break
                    continue
            
            # If all configurations failed
            print("All GPU configurations failed, falling back to CPU")
            self.use_gpu = False
                
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                print(f"CUDA/GPU error when moving FAISS index to GPU: {e}")
                print("Falling back to CPU")
            else:
                print(f"Runtime error when moving FAISS index to GPU: {e}")
            self.use_gpu = False
            
        except ImportError as e:
            print(f"Import error - FAISS GPU support not available: {e}")
            print("Falling back to CPU")
            self.use_gpu = False
            
        except Exception as e:
            print(f"Unexpected error when moving FAISS index to GPU: {e}")
            print("Falling back to CPU")
            self.use_gpu = False

    def _search_neighbors(self, x_t_np: np.ndarray, n_neighbors: int = 100, nprobe: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for nearest neighbors using FAISS.
        
        Args:
            x_t_np: numpy array of shape [B, D] to search for
            n_neighbors: number of nearest neighbors to retrieve
        Returns:
            distances: tensor of shape [B, n_neighbors]
            indices: tensor of shape [B, n_neighbors]
        """
        self.index.nprobe = nprobe
        # faiss.omp_set_num_threads(os.cpu_count())

        distances, indices = self.index.search(x_t_np, n_neighbors)
        distances = torch.from_numpy(distances).to(self.device, dtype=self.dtype)
        indices = torch.from_numpy(indices).to(self.device)
        return distances, indices

    
    def find_nearest_matches(self, generated_samples: torch.Tensor, n_neighbors: int = 100, nprobe: int = 10) -> Tuple[list, list, list]:
        """
        Find the nearest match in the dataset for each generated sample using FAISS.
        
        Args:
            generated_samples: tensor of shape [B, C, H, W] - the generated images
            
        Returns:
            nearest_indices: list of indices of nearest matches in the dataset
            distances: list of L2 distances to the nearest matches
            nearest_samples: list of the actual nearest sample images
        """
        B, C, H, W = generated_samples.shape
        
        # Flatten generated samples
        generated_flat = generated_samples.reshape(B, -1)  # [B, D]
        
        # Convert to numpy for FAISS search
        generated_np = generated_flat.cpu().numpy().astype(np.float32)
        
        # Search for nearest neighbors
        self.index.nprobe = nprobe
        try:
            param = faiss.GpuParameterSpace()
            param.set_index_parameter(self.index, "nprobe", nprobe)
        except Exception as e:
            print(f"Error setting nprobe: {e}. Probably working on CPU.")


        distances, indices = self.index.search(generated_np, n_neighbors)  # [B, n_neighbors]
        
        # Convert back to tensors
        distances = torch.from_numpy(distances).to(self.device, dtype=self.dtype)  # [B, n_neighbors]
        indices = torch.from_numpy(indices).to(self.device)  # [B, n_neighbors]
        
        # Get the actual nearest samples (implemented by subclasses)
        nearest_samples = self._get_nearest_samples(indices, C, H, W)
        
        return indices.cpu().tolist(), distances.cpu().tolist(), nearest_samples
    
    @abstractmethod
    def _get_neighbor_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get neighbor samples for the given indices.
        
        Args:
            indices: tensor of shape [B, n_neighbors]
            
        Returns:
            batch_neighbors: tensor of shape [B, n_neighbors, D]
        """
        pass
    
    @abstractmethod
    def _get_nearest_samples(self, indices: torch.Tensor, C: int, H: int, W: int) -> list:
        """
        Get nearest samples for the given indices.
        
        Args:
            indices: tensor of shape [B, n_neighbors]
            C, H, W: image dimensions
            
        Returns:
            nearest_samples: list of lists of sample tensors, shape [B][n_neighbors]
        """
        pass



class FAISSDiskBased(BaseFAISS):
    """
    FAISS-based approximate nearest neighbors search that works directly from disk.
    
    This class builds FAISS indices directly from dataset files without loading
    the entire dataset into memory, making it suitable for very large datasets.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, pca_dim: int = None):
        """
        Initialize FAISS disk-based approximate nearest neighbors search.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
        """
        super().__init__(use_gpu=use_gpu, index_type=index_type, device=device, dtype=dtype, num_clusters=num_clusters)
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.pca_dim = pca_dim
        if patch_size is None:
            self.patch_size = self.get_dataset().resolution
        else:
            self.patch_size = patch_size
        
        # Default index path if not provided
        if index_path is None:
            dataset_name = Path(dataset_path).stem
            base_path = os.environ.get("SCRATCH", "/scratch/07362/gdaras/")
            index_path = f"{base_path}/faiss_index_{dataset_name}_{index_type}_{self.patch_size}.index"
            print(f"Will be saving index to {index_path}")
        else:
            print(f"Will be saving index to {index_path}")
        self.index_path = index_path
        
        # Dataset info
        self.dataset_dim = None
        self.dataset_size = None
        
        self.dataset = None  # Will be initialized lazily when first needed
        
        # Build or load the index
        self._build_or_load_index()
    
    def _get_dataset(self):
        """Lazy initialization of dataset object."""
        if self.dataset is None:
            self.dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False)
        return self.dataset
    
    def _get_dataset_info(self):
        """Get dataset dimension and size without loading all data."""
        # Create a temporary dataset object to get info
        temp_dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False)
        num_patches_per_image = (temp_dataset.resolution // self.patch_size)**2
        self.dataset_size = len(temp_dataset) * num_patches_per_image
        self.dataset_dim = (temp_dataset.resolution**2) * temp_dataset.num_channels
    
    def _build_index_from_disk(self):
        """Build FAISS index by processing dataset in batches from disk."""
        
        # Create dataset object
        dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        
        # Initialize index
        num_patches_per_image = (dataset.resolution // self.patch_size)**2
        patch_dim = self.dataset_dim // num_patches_per_image
        # faiss.omp_set_num_threads(os.cpu_count())

        self._create_index(patch_dim, self.pca_dim)
        # faiss.omp_set_num_threads(os.cpu_count())
        
        # Process dataset in batches
        first_batch = True
        for batch in tqdm(dataloader, desc="Indexing dataset"):   
            # split into patches

            batch_patches = split_image_into_patches(batch['image'], self.patch_size)
            batch_vectors = batch_patches.reshape(batch_patches.shape[0] * batch_patches.shape[1] * batch_patches.shape[2], -1)


            if self.index_type == 'ivf' and first_batch:
                # Train the index with first batch
                print("Training disk-based FAISS index...")
                self.index.train(batch_vectors)
                first_batch = False
            
            # Add vectors to index
            self.index.add(batch_vectors)
            
        
        # Move to GPU if requested
        self._move_to_gpu()
        
        # Save index to disk
        if self.index_path:
            print(f"Saving FAISS index to: {self.index_path}")
            # Check if index is on GPU and move to CPU before saving
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, self.index_path)
            else:
                faiss.write_index(self.index, self.index_path)
    
    def _load_index_from_disk(self):
        """Load existing FAISS index from disk."""
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        self._move_to_gpu()
    
    def _build_or_load_index(self):
        """Build new index or load existing one from disk."""
        # Get dataset info first
        self._get_dataset_info()
        
        # Check if index already exists
        if os.path.exists(self.index_path):
            try:
                self._load_index_from_disk()
                print("Successfully loaded existing FAISS index")
                return
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                print("Will build new index...")
        
        # Build new index
        self._build_index_from_disk()

    
    def _get_neighbor_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """Get neighbor samples from disk."""
        B = indices.shape[0]
        batch_neighbors = []
        
        for b in range(B):
            # Get the nearest neighbors for this batch item
            neighbor_indices = indices[b]  # [n_neighbors]
            neighbor_samples = []
            
            for idx in neighbor_indices:
                # Load sample from disk
                dataset = self._get_dataset()
                sample = dataset[idx]['image'][:3]
                sample_flat = sample.reshape(-1)  # [D]
                neighbor_samples.append(sample_flat)
            
            neighbor_samples = torch.stack(neighbor_samples)  # [n_neighbors, D]
            batch_neighbors.append(neighbor_samples)
        
        return torch.stack(batch_neighbors)  # [B, n_neighbors, D]
    
    def _get_nearest_samples(self, indices: torch.Tensor, C: int, H: int, W: int) -> list:
        """Get nearest samples from disk."""
        B, n_neighbors = indices.shape
        nearest_samples = []
        
        for b in tqdm(range(B), desc="Getting nearest samples"):
            batch_samples = []
            for n in range(n_neighbors):
                # Load sample from disk
                idx = indices[b, n].item()
                num_patches_per_image = (self._get_dataset().resolution // self.patch_size)**2
                real_dataset_index, patch_index = divmod(idx, num_patches_per_image)  # originally it is indexing the patches
                dataset = self._get_dataset()
                sample = dataset[real_dataset_index]['image'][:3]
                # now find the patch inside the sample
                sample_tensor = torch.from_numpy(sample)
                patch_row = patch_index // (self._get_dataset().resolution // self.patch_size)
                patch_col = patch_index % (self._get_dataset().resolution // self.patch_size)
                sample_patch = sample_tensor[:, patch_row*self.patch_size:(patch_row+1)*self.patch_size, patch_col*self.patch_size:(patch_col+1)*self.patch_size]
                batch_samples.append(sample_patch.numpy())
            nearest_samples.append(batch_samples)
        
        return nearest_samples


class FAISSDiskBasedPixelLevel(FAISSDiskBased):
    """
        Similar to FAISSDiskBased, but creates a separate patch for each pixel (central crop around the pixel).
    """
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, keep_ratio: float = 0.1, pca_dim: int = None):
        self.keep_ratio = keep_ratio
        # Add mapping to track which patches were actually added to the index

        super().__init__(dataset_path=dataset_path, 
                              use_gpu=use_gpu, index_type=index_type,
                              index_path=index_path, device=device,
                              dtype=dtype, batch_size=batch_size,
                              num_clusters=num_clusters, patch_size=patch_size)
    
    def _build_index_from_disk(self):
        """Build FAISS index by processing dataset in batches from disk."""
        
        # Create dataset object
        dataset = ImageFolderDataset(self.dataset_path, use_labels=False, cache=False, only_positive=False)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False, num_workers=16)


        real_patch_size = self.patch_size + 1 if (self.patch_size % 2 == 0) else self.patch_size
        dataset_dim = real_patch_size ** 2 * 3
        self._create_index(dataset_dim, self.pca_dim)
        
        
        # Process dataset in batches
        first_batch = True
        for batch in tqdm(dataloader, desc="Indexing dataset"):   


            batch_patches = split_image_into_pixel_level_patches(batch['image'].to(self.dtype), self.patch_size, self.keep_ratio)  # B, C, H, W, patch_size, patch_size
            real_patch_size = batch_patches.shape[-1]

            batch_patches = batch_patches.permute(0, 2, 3, 1, 4, 5) # B, H, W, C, patch_size, patch_size
            batch_patches = batch_patches.reshape(batch_patches.shape[0] * batch_patches.shape[1] * batch_patches.shape[2], -1) # B * H * W, C * patch_size * patch_size


            if self.index_type == 'ivf' and first_batch:
                # Train the index with first batch
                print("Training disk-based FAISS index...")
                self._move_to_gpu()
                self.index.train(batch_patches)
                first_batch = False
            
            # Add vectors to index
            self.index.add(batch_patches)
            
        
        # Move to GPU if requestedpa
        self._move_to_gpu()
        
        # Save index to disk
        if self.index_path:
            print(f"Saving FAISS index to: {self.index_path}")
            # Check if index is on GPU and move to CPU before saving
            if hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, self.index_path)
            else:
                faiss.write_index(self.index, self.index_path)
            

    def _load_index_from_disk(self):
        """Load existing FAISS index from disk."""
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
                
        self._move_to_gpu()

    def _get_nearest_samples(self, indices: torch.Tensor, C: int, H: int, W: int) -> list:
        """Get nearest samples from disk with caching."""
        B, n_neighbors = indices.shape
        nearest_samples = np.zeros((B, n_neighbors, 3, H, W))
        # print("number of things in index: ", self.index.ntotal)
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        cpu_index.make_direct_map()
        nearest_samples = cpu_index.reconstruct_batch(indices.reshape(-1)).reshape(B, n_neighbors, 3, H, W)
        return nearest_samples
        



class CropScore():
    """
    Compute a per-crop score.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, pca_dim: int = None):
        """
        Initialize CropScore.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
        """
        self.patch_size = patch_size
        self.faiss_disk = FAISSDiskBased(dataset_path=dataset_path, 
                                         index_path=index_path, 
                                         patch_size=patch_size, 
                                         use_gpu=use_gpu, 
                                         index_type=index_type, 
                                         device=device, 
                                         dtype=dtype, 
                                         batch_size=batch_size, 
                                         num_clusters=num_clusters, 
                                         pca_dim=pca_dim)
    

    def find_neighbors(self, image: torch.Tensor, n_neighbors: int = 10, nprobe: int = 20000) -> torch.Tensor:
        """
        Find patch-level neighbors.
        
        Args:
            image: torch.Tensor of shape [B, C, H, W]
        Returns:
            batched_nearest_samples: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
            batched_distances: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors]
        """
        patches = split_image_into_patches(image, self.patch_size)
        num_patches_per_row = image.shape[2] // self.patch_size
        num_patches_per_col = image.shape[3] // self.patch_size
        batched_patches = patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5])
        _, batched_distances, batched_nearest_samples = self.faiss_disk.find_nearest_matches(batched_patches, n_neighbors=n_neighbors, nprobe=nprobe)
        batched_nearest_samples = torch.from_numpy(np.stack(batched_nearest_samples)) 
        batched_nearest_samples = batched_nearest_samples.reshape(image.shape[0], num_patches_per_row, num_patches_per_col, *batched_nearest_samples.shape[1:])
        batched_distances = torch.from_numpy(np.stack(batched_distances))
        batched_distances = batched_distances.reshape(image.shape[0], num_patches_per_row, num_patches_per_col, *batched_distances.shape[1:])
        return batched_nearest_samples, batched_distances


    def __call__(self, x_t: torch.Tensor, 
                sigma_t: torch.Tensor, temperature: float = 1.0, 
                n_neighbors: int = 10, nprobe: int = 10) -> torch.Tensor:
        """
        Compute score from nearest neighbors using softmax weighting.
        
        Args:
            x_t: input tensor [B, C, H, W]
            sigma_t: noise level [B]
            
        Returns:
            score: computed score tensor [B, C, H, W]
        """
        samples, distances = self.find_neighbors(image=x_t, n_neighbors=n_neighbors, nprobe=nprobe) # samples: [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
        batch_size, num_patches_per_row, num_patches_per_col, n_neighbors, _, patch_size, _ = samples.shape
        sigma2 = sigma_t ** 2
        D = x_t[0].numel()
        
        # Compute log-likelihoods for the subset

        log_norm = -0.5 * D * torch.log(2 * torch.pi * sigma2)
        log_likelihoods = log_norm - distances / (2 * sigma2)
        
        # Compute softmax weights over the subset
        weights = F.softmax(log_likelihoods / temperature, dim=-1)  # (B, num_patches_per_row, num_patches_per_col, n_neighbors)
        
        # Compute weighted differences
        samples_perm = samples.permute(0, 3, 1, 2, 4, 5, 6).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size) # [B * n_neighbors, num_patches_per_row, num_patches_per_col, 3, patch_size, patch_size]
        assembled = assemble_patches(samples_perm).reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3]) # [B, n_neighbors, 3, H, W]
        
        diffs = assembled - x_t[:, None] # [B, n_neighbors, 3, H, W]
        patched_diffs = split_image_into_patches(diffs.reshape(batch_size * n_neighbors, 3, x_t.shape[2], x_t.shape[3]), self.patch_size)
        weighted_diffs = weights.permute(0, 3, 1, 2).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col)[:, :, :, None, None, None] * patched_diffs
        weighted_diffs = assemble_patches(weighted_diffs).reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3])
        score = weighted_diffs.sum(dim=1) / sigma2  # [B, 3, patch_size, patch_size]   
        denoised = from_noise_pred_to_x0_pred_ve(x_t, sigma_t, score)
        return denoised

    



class CropScorePixelLevel():
    """
    Compute a per-crop score.
    """
    
    def __init__(self, dataset_path: str, 
                 use_gpu: bool = True, index_type: str = 'ivf',
                 index_path: Optional[str] = None, device: torch.device = None,
                 dtype: torch.dtype = torch.float32, batch_size: int = 16,
                 num_clusters: int = 4096, patch_size: int = None, keep_ratio: float = 0.1, pca_dim: int = None):
        """
        Initialize CropScore.
        
        Args:
            dataset_path: Path to the dataset (directory or zip file)
            use_gpu: Whether to use GPU acceleration for FAISS
            index_type: Type of FAISS index ('ivf', 'flat', 'hnsw')
            index_path: Path to save/load the FAISS index (optional)
            device: torch device for tensor operations
            dtype: data type for tensors
            batch_size: Batch size for processing dataset during index building
        """
        self.patch_size = patch_size
        self.faiss_disk = FAISSDiskBasedPixelLevel(dataset_path=dataset_path, 
                                         index_path=index_path, 
                                         patch_size=patch_size, 
                                         use_gpu=use_gpu, 
                                         index_type=index_type, 
                                         device=device, 
                                         dtype=dtype, 
                                         batch_size=batch_size, 
                                         num_clusters=num_clusters, 
                                         keep_ratio=keep_ratio,
                                         pca_dim=pca_dim)
    
    

    def find_neighbors(self, image: torch.Tensor, n_neighbors: int = 10, nprobe: int = 20000) -> torch.Tensor:
        """
        Find patch-level neighbors.
        
        Args:
            image: torch.Tensor of shape [B, C, H, W]
        Returns:
            batched_nearest_samples: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
            batched_distances: torch.Tensor of shape [B, num_patches_per_row, num_patches_per_col, n_neighbors]
        """
        patches = split_image_into_pixel_level_patches(image, self.patch_size) # B, C, H, W, patch_size, patch_size
        patches_perm = patches.permute(0, 2, 3, 1, 4, 5) # B, H, W, C, patch_size, patch_size
        batched_patches = patches_perm.reshape(patches_perm.shape[0] * patches_perm.shape[1] * patches_perm.shape[2], 3, patches_perm.shape[-2], patches_perm.shape[-1]) # B * H * W, 3, patch_size, patch_size
        _, batched_distances, batched_nearest_samples = self.faiss_disk.find_nearest_matches(batched_patches, n_neighbors=n_neighbors, nprobe=nprobe)

        batched_nearest_samples = torch.from_numpy(np.stack(batched_nearest_samples))  # B * H * W, n_neighbors, 3, patch_size, patch_size
        batched_nearest_samples = batched_nearest_samples.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_nearest_samples.shape[1:]) # B, H, W, n_neighbors, 3, patch_size, patch_size
        batched_distances = torch.from_numpy(np.stack(batched_distances))
        batched_distances = batched_distances.reshape(image.shape[0], image.shape[-2], image.shape[-1], *batched_distances.shape[1:])  # B, H, W, n_neighbors
        return batched_nearest_samples, batched_distances


    def __call__(self, x_t: torch.Tensor, 
                sigma_t: torch.Tensor, temperature: float = 1.0, 
                n_neighbors: int = 10, nprobe: int = 10) -> torch.Tensor:
        """
        Compute score from nearest neighbors using softmax weighting.
        
        Args:
            x_t: input tensor [B, C, H, W]
            sigma_t: noise level [B]
            
        Returns:
            score: computed score tensor [B, C, H, W]
        """
        samples, distances = self.find_neighbors(image=x_t, n_neighbors=n_neighbors, nprobe=nprobe) # samples: [B, num_patches_per_row, num_patches_per_col, n_neighbors, 3, patch_size, patch_size]
        batch_size, num_patches_per_row, num_patches_per_col, n_neighbors, _, patch_size, _ = samples.shape
        sigma2 = sigma_t ** 2
        D = x_t[0].numel()
        
        # Compute log-likelihoods for the subset

        log_norm = -0.5 * D * torch.log(2 * torch.pi * sigma2)
        log_likelihoods = log_norm - distances / (2 * sigma2)
        
        # Compute softmax weights over the subset
        weights = F.softmax(log_likelihoods / temperature, dim=-1)  # (B, num_patches_per_row, num_patches_per_col, n_neighbors)
        
        # Compute weighted differences
        real_patch_size = self.patch_size + 1 if self.patch_size % 2 == 0 else self.patch_size
        samples_perm = samples.permute(0, 3, 4, 1, 2, 5, 6).reshape(batch_size * n_neighbors, 3, x_t.shape[2], x_t.shape[3], real_patch_size, real_patch_size)
        diffs = samples_perm - x_t[:, :, :, :, None, None]

        weights = weights.permute(0, 3, 1, 2).reshape(batch_size * n_neighbors, num_patches_per_row, num_patches_per_col)

        diffs_on_central_pixel = diffs[:, :, :, :, real_patch_size // 2, real_patch_size // 2] # (B * n_neighbors, 3, H, W)
        weighted_diffs = weights[:, None] * diffs_on_central_pixel  # (B * n_neighbors, 3, H, W)
        weighted_diffs = weighted_diffs.reshape(batch_size, n_neighbors, 3, x_t.shape[2], x_t.shape[3])
        score = weighted_diffs.sum(dim=1) / sigma2  # [B, 3, H, W]   
        denoised = from_noise_pred_to_x0_pred_ve(x_t, sigma_t, score)
        return denoised

if __name__ == "__main__":
    base_path = os.environ.get("SCRATCH", "/scratch/07362/gdaras/")
    # image_path = "/scratch/07362/gdaras/datasets/afhqv2-64x64/00000/img00000000.png"
    image_path = "/home1/07362/gdaras/freefusion/images/ood_cat.png"
    # image_path = "/scratch/07362/gdaras/datasets/cifar10-32x32/00000/img00000000.png"

    # dataset_path = "datasets/cifar10-32x32"
    # dataset_name = "cifar10-32x32"
    # dataset_resolution = 32
    # dataset_size = 50_000

    dataset_path = "datasets/afhqv2-64x64"
    dataset_name = "afhqv2"
    dataset_resolution = 64
    dataset_size = 16_000


    patch_size = 64
    keep_ratio = 0.1
    n_probe = 12
    n_neighbors = 64
    sigma_value = 0.2
    batch_size = 4096



    num_clusters = int(np.sqrt(dataset_size * dataset_resolution ** 2))  # it needs to be sqrt(number_of_training_points)
    pca_dim = None

    test_mode = "crop_score_pixel_level" # choose between disk_based, disk_based_pixel_level, scrop_score, crop_score_pixel_level
    # test_mode = "crop_score"


    test_image = ambient_utils.load_image(image_path, device=torch.device("cpu"))[:, :3] * 2 - 1

    if test_mode == "disk_based":
        faiss_disk = FAISSDiskBased(dataset_path=os.path.join(base_path, dataset_path), 
                                    use_gpu=True, index_type='ivf', 
                                    index_path=os.path.join(base_path, f"datasets/faiss/faiss_index_{dataset_name}_ivf_{patch_size}.index"),
                                    num_clusters=num_clusters, patch_size=patch_size, batch_size=batch_size)    

        # get the central patch of size patch_size
        top_left_patch = test_image[:, :, :patch_size, :patch_size]
        ambient_utils.save_images(top_left_patch, "top_left_patch.png")
        _, _, samples = faiss_disk.find_nearest_matches(top_left_patch, n_neighbors=n_neighbors, nprobe=n_probe)
        samples = torch.from_numpy(np.stack(samples[0]))  # Stack the neighbors for the first batch item
        ambient_utils.save_images(samples, "nearest_samples.png")
    elif test_mode == "crop_score":
        sigma_t = torch.tensor(sigma_value).unsqueeze(0)
        crop_score = CropScore(dataset_path=os.path.join(base_path, dataset_path), 
                                use_gpu=True, index_type='ivf', 
                                index_path=os.path.join(base_path, f"datasets/faiss/faiss_index_{dataset_name}_ivf_{patch_size}.index"),
                                num_clusters=128, patch_size=patch_size, device=torch.device("cpu"), batch_size=batch_size)    
        noisy_test_image = test_image + sigma_t * torch.randn_like(test_image)
        ambient_utils.save_images(noisy_test_image, f"noisy_test_image.png")
        denoised = crop_score(noisy_test_image, sigma_t=sigma_t)
        ambient_utils.save_images(denoised, f"denoised_patch_size_{patch_size}_sigma_{sigma_value}.png")
    elif test_mode == "disk_based_pixel_level":
        test_image_patches = split_image_into_pixel_level_patches(test_image, patch_size)
        patch_level_of_interest = test_image_patches[:, :, 16, 16]
        ambient_utils.save_images(patch_level_of_interest, "patch_level_of_interest.png")
        faiss_disk = FAISSDiskBasedPixelLevel(dataset_path=os.path.join(base_path, dataset_path), 
                                             use_gpu=True, index_type='ivf', 
                                             index_path=os.path.join(base_path, f"datasets/faiss/faiss_pixel_level_index_{dataset_name}-{dataset_resolution}x{dataset_resolution}_ivf_patch_size_{patch_size}.index"),
                                             num_clusters=num_clusters, patch_size=patch_size, keep_ratio=keep_ratio, batch_size=batch_size, pca_dim=pca_dim)    

        _, _,samples = faiss_disk.find_nearest_matches(patch_level_of_interest, n_neighbors=n_neighbors, nprobe=n_probe)
        samples = torch.from_numpy(np.stack(samples[0]))
        ambient_utils.save_images(samples, "nearest_samples.png")
    elif test_mode == "crop_score_pixel_level":
        sigma_t = torch.tensor(sigma_value).unsqueeze(0)
        crop_score = CropScorePixelLevel(dataset_path=os.path.join(base_path, dataset_path), 
                                use_gpu=True, index_type='ivf', 
                                index_path=os.path.join(base_path, f"datasets/faiss/faiss_pixel_level_index_{dataset_name}-{dataset_resolution}x{dataset_resolution}_ivf_patch_size_{patch_size}.index"),
                                num_clusters=num_clusters, patch_size=patch_size, device=torch.device("cpu"),
                                batch_size=batch_size,
                                keep_ratio=keep_ratio, pca_dim=pca_dim, dtype=torch.float16)
        noisy_test_image = test_image + sigma_t * torch.randn_like( test_image)
        ambient_utils.save_images(noisy_test_image, f"noisy_test_image.png")
        denoised = crop_score(noisy_test_image, sigma_t=sigma_t, nprobe=n_probe, n_neighbors=n_neighbors)
        ambient_utils.save_images(denoised, f"denoised_patch_size_{patch_size}_sigma_{sigma_value}.png")
        print("Number of patches in the index: ", crop_score.faiss_disk.index.ntotal)
        print("Number of clusters in the index: ", crop_score.faiss_disk.index.nlist)
        print("Average cluster size: ", crop_score.faiss_disk.index.ntotal // crop_score.faiss_disk.index.nlist)
    else:
        raise ValueError(f"Invalid test mode: {test_mode}")




from PIL import Image
import os, torch
from torch.utils.data import Dataset
from torchvision import transforms


class CTGTDataset(Dataset):
    """
    CrossTalk ↔ Ground-Truth dataset with optional, deterministic gain/gamma jitter.

    Assumes the file list is arranged as pairs:
      base frame k → index 2*k = RAW-resized, index 2*k+1 = TILE+shuffle version.

    Parameters
    ----------
    input_dir : str
        Folder with the “crosstalk” input frames.
    target_dir : str
        Folder with the clean ground-truth frames (same filenames).
    train : bool, default True
        If True, allow jitter (per jitter_mode); off for val/test.
    jitter_range : (float, float), default (0.85, 1.25)
        Uniform range [γ_min, γ_max] for multiplicative brightness/gamma factor.
    jitter_mode : {"both", "raw_only", "tile_only", "none"}, default "both"
        Which indices in each pair get jitter:
          - "both": apply to RAW and TILE
          - "raw_only": apply only to even indices (RAW)
          - "tile_only": apply only to odd indices (TILE)
          - "none": disable jitter regardless of `train`
    deterministic_pairs : bool, default True
        If True, the *same* γ is used for both frames in a (RAW, TILE) pair.
    base_seed : int, default 1337
        Base seed to make jitter deterministic across runs/models.
    transform : torchvision Transform, optional
        Defaults to ToTensor() (0–255 → 0–1 float).
    """

    def __init__(self,
                 input_dir,
                 target_dir,
                 train: bool = True,
                 jitter_range: tuple[float, float] = (0.5, 1.0),
                 jitter_mode: str = "both",
                 deterministic_pairs: bool = True,
                 base_seed: int = 1337,
                 transform = None):

        self.input_dir  = input_dir
        self.target_dir = target_dir
        self.train      = train

        self.g0, self.g1 = jitter_range
        self.jitter_mode = jitter_mode.lower()
        assert self.jitter_mode in {"both", "raw_only", "tile_only", "none"}

        self.deterministic_pairs = deterministic_pairs
        self.base_seed = int(base_seed)

        # --- match filenames between the two folders ---------------------
        self.list_input  = sorted(os.listdir(self.input_dir))
        self.list_target = sorted(os.listdir(self.target_dir))
        common = sorted(set(self.list_input) & set(self.list_target))
        self.list_input  = [f for f in self.list_input  if f in common]
        self.list_target = [f for f in self.list_target if f in common]
        if len(self.list_input) != len(self.list_target):
            raise ValueError("Input/target counts mismatch after filtering.")

        self.transform = transform or transforms.ToTensor()  # 0–255 → 0–1

    def __len__(self):
        return len(self.list_input)

    def _pair_id(self, index: int) -> int:
        """Map index to its (RAW,TILE) pair id: 0,0→0; 1,1→0; 2,3→1; etc."""
        return index // 2

    def _is_raw(self, index: int) -> bool:
        """Even index = RAW, odd index = TILE (by your dataset convention)."""
        return (index % 2) == 0

    def _should_jitter(self, index: int) -> bool:
        if self.jitter_mode == "none":
            return False
        if self.jitter_mode == "both":
            return True
        if self.jitter_mode == "raw_only":
            return self._is_raw(index)
        if self.jitter_mode == "tile_only":
            return not self._is_raw(index)
        return False

    def _sample_gamma(self, index: int) -> torch.Tensor:
        """
        Return a deterministic gamma in [g0,g1].
        If deterministic_pairs=True, gamma is tied to the pair id so RAW/TILE share it.
        """
        if self.deterministic_pairs:
            seed_key = self._pair_id(index)
        else:
            seed_key = index
        g = torch.Generator()
        g.manual_seed(self.base_seed + int(seed_key))
        gamma = self.g0 + (self.g1 - self.g0) * torch.rand(1, generator=g)
        return gamma

    def __getitem__(self, index: int):
        fname = self.list_input[index]
        inp_path  = os.path.join(self.input_dir,  fname)
        targ_path = os.path.join(self.target_dir, fname)

        inp  = self.transform(Image.open(inp_path ).convert('L'))  # [1,H,W] in [0,1]
        targ = self.transform(Image.open(targ_path).convert('L'))

        # Paired photometric jitter (exposure-like), applied as configured
        if self.train and self.jitter_mode != "none" and self._should_jitter(index):
            gamma = self._sample_gamma(index)
            # gamma-correct both input and target (paired), clamp for safety
            inp  = (inp * gamma).clamp_(0.0, 1.0)
            targ = (targ * gamma).clamp_(0.0, 1.0)

        return inp, targ
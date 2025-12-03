import os
import numpy as np
import torch
import torch.utils.data as data
from scipy import io
import glob

class iBims(data.Dataset):
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        # Tìm file imagelist.txt
        candidates = glob.glob(os.path.join(self.dataset_path, "**/imagelist.txt"), recursive=True)
        if not candidates:
            raise FileNotFoundError("Không tìm thấy imagelist.txt")
        self.list_path = candidates[0]

        with open(self.list_path, "r") as f:
            self.image_names = [x.strip() for x in f.readlines()]

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        name_base = os.path.splitext(file_name)[0]

        # Tìm file .mat
        mat_files = glob.glob(os.path.join(self.dataset_path, "**", file_name), recursive=True)
        if not mat_files:
            raise FileNotFoundError(f"Không tìm thấy file mat: {file_name}")
        mat_path = mat_files[0]

        # Tìm calib
        calib_files = glob.glob(os.path.join(self.dataset_path, "**/calib/" + name_base + ".txt"), recursive=True)
        K = np.loadtxt(calib_files[0], delimiter=",") if calib_files else np.eye(3)

        # Load dữ liệu
        mat_data = io.loadmat(mat_path)["data"]
        rgb = mat_data["rgb"][0][0] / 255.0
        depth = mat_data["depth"][0][0]

        # Tạo mask
        mask_valid = np.ones_like(depth, dtype=np.float32)
        for key in ["mask_invalid","mask_transp"]:
            if key in mat_data:
                mask_valid *= mat_data[key][0][0].astype(np.float32)
        mask_valid *= (depth != 0).astype(np.float32)

        # Chuyển sang Tensor
        rgb_t = torch.tensor(rgb.astype(np.float32).transpose(2,0,1))
        depth_t = torch.tensor((depth * mask_valid).astype(np.float32))[None, ...]
        sparse_dep = self.get_sparse_depth(depth_t, 1000)
        K_t = torch.tensor(K.astype(np.float32))

        return {"rgb": rgb_t, "dep": sparse_dep, "gt": depth_t, "K": K_t, "filename": file_name}

    def __len__(self):
        return len(self.image_names)

    def get_sparse_depth(self, dep, num_sample):
        c,h,w = dep.shape
        idx = torch.nonzero(dep>0.001).flatten()
        if idx.numel()==0: return torch.zeros_like(dep)
        sample = idx[torch.randperm(len(idx))[:num_sample]]
        mask = torch.zeros_like(dep.view(-1))
        mask[sample] = 1
        mask = mask.view(c,h,w)
        return dep * mask

def valset(args):
    return iBims(args)

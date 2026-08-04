import os
import re
import torch
import torch.nn.functional as F
import pydicom
from torch.utils.data import Dataset
import pandas as pd


class OCTDataset(Dataset):
    """
    Base dataset: loads ALL OCT images under directory
    """
    def __init__(self, root_dir,
                 target_depth=15, target_h=512, target_w=512, augment=False):

        self.root_dir     = root_dir
        self.target_depth = target_depth
        self.target_h     = target_h
        self.target_w     = target_w
        self.augment      = augment

        self.image_paths = self._scan_dicom_files()
        self.meta_rows   = [self._parse_meta(p) for p in self.image_paths]

        print(f"[OCTDataset] Total images: {len(self.image_paths)}")

        import torchvision.transforms as T
        self.aug = T.RandomHorizontalFlip(p=0.5)

    # -------------------------
    # Key builder
    # -------------------------
    def build_key(self, subj_id, date, time):
        return (subj_id, date, time)

    # -------------------------
    # Scan files
    # -------------------------
    def _scan_dicom_files(self):
        paths = []
        for root, _, files in os.walk(self.root_dir):
            if root.endswith("6.00mmX6.00mm_OCTA Retina"):
                if "Enface.dcm" in files:
                    paths.append(os.path.join(root, "OCT.dcm"))
        return paths

    # -------------------------
    # Parse metadata from path
    # -------------------------
    def _parse_meta(self, path):
        parts = path.replace("\\", "/").split("/")

        patient_folder = parts[-3]
        scan_folder    = parts[-2]

        # ---- Subject ID ----
        if "poor" in patient_folder.lower() or "mild" in patient_folder.lower():
            match   = re.search(r'BIOC_\d+_v\d+', patient_folder)
            subj_id = match.group(0) if match else "_".join(patient_folder.split("_")[3:])
        else:
            subj_id = "_".join(patient_folder.split("_")[3:])

        # ---- Gender ----
        gender_match = re.search(r'(Male|Female)', patient_folder, re.IGNORECASE)
        gender       = gender_match.group(0) if gender_match else None

        # ---- Age ----
        age_match = re.search(r'_(\d+)_BIOC', patient_folder)
        age       = int(age_match.group(1)) if age_match else None

        # ---- Scan info ----
        scan_parts = scan_folder.split("_")
        date = str(scan_parts[0])
        time = str(scan_parts[1]).zfill(6)
        eye  = scan_parts[2].upper()

        return {
            "Subj_ID": subj_id,
            "Gender":  gender,
            "Age":     age,
            "Eye":     eye,
            "Date":    date,
            "Time":    time,
        }

    # -------------------------
    # Load DICOM — evenly spaced frames, original spatial dimensions
    # -------------------------
    def load_dcm_volume(self, path):
        dcm = pydicom.dcmread(path)
        vol = torch.tensor(dcm.pixel_array, dtype=torch.float32)  # (D, H, W)

        # Normalize
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)

        D = vol.shape[0]

        # Evenly spaced index selection along depth — no interpolation, no resizing
        indices = torch.linspace(0, D - 1, self.target_depth).long()
        vol     = vol[indices]                                # (target_depth, H, W)

        return vol

    # -------------------------
    # Augmentation
    # -------------------------
    def apply_augmentation(self, volume):
        # volume: (D, H, W)
        volume = volume.unsqueeze(1)     # (D, 1, H, W)

        seed = torch.randint(0, 1_000_000, (1,)).item()
        augmented = []
        for slc in volume:
            torch.manual_seed(seed)
            augmented.append(self.aug(slc))

        return torch.stack(augmented).squeeze(1)  # (D, H, W)

    # -------------------------
    # Get item
    # -------------------------
    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        meta  = self.meta_rows[idx]
        image = self.load_dcm_volume(path)

        if self.augment:
            image = self.apply_augmentation(image)

        return {
            "image":   image,
            "subj_id": meta["Subj_ID"],
            "gender":  meta["Gender"],
            "age":     meta["Age"],
            "eye":     meta["Eye"],
            "date":    meta["Date"],
            "time":    meta["Time"],
            "path":    path,
        }

    def __len__(self):
        return len(self.image_paths)


# =========================================================
# QC FILTER DATASET
# =========================================================
class OCTDatasetPassQC(OCTDataset):
    """Only keep samples where QC_pass >= 1"""

    def __init__(self, root_dir, qc_csv, augment=False):
        super().__init__(root_dir=root_dir, augment=augment)

        df = pd.read_csv(qc_csv)
        df["Date"] = df["Date"].astype(str)
        df["Time"] = df["Time"].apply(lambda x: str(int(float(x))).zfill(6))
        df = df[df["QC_pass"] >= 1]

        self.qc_keys = set(zip(df["Subj_ID"], df["Date"], df["Time"]))

        filtered_paths, filtered_meta = [], []
        for path, meta in zip(self.image_paths, self.meta_rows):
            key = self.build_key(meta["Subj_ID"], meta["Date"], meta["Time"])
            if key in self.qc_keys:
                filtered_paths.append(path)
                filtered_meta.append(meta)

        self.image_paths = filtered_paths
        self.meta_rows   = filtered_meta
        print(f"[OCTDatasetPassQC] Samples: {len(self.image_paths)}")


# =========================================================
# QC + LABEL DATASET
# =========================================================
class OCTDatasetPassQCWithLabel(OCTDatasetPassQC):
    """Only keep samples that pass QC and have a label."""

    def __init__(self, root_dir, qc_csv, label_csv, label_col, augment=False):
        super().__init__(root_dir=root_dir, qc_csv=qc_csv, augment=augment)

        df = pd.read_csv(label_csv)
        if label_col not in df.columns:
            raise ValueError(f"{label_col} not found in label CSV")

        df["Date"] = df["Date"].astype(str)
        df["Time"] = df["Time"].apply(lambda x: str(int(float(x))).zfill(6))

        self.label_map = {}
        for _, r in df.iterrows():
            if pd.notna(r[label_col]):
                key = (r["Subj_ID"], r["Date"], r["Time"])
                self.label_map[key] = r[label_col]

        filtered_paths, filtered_meta = [], []
        for path, meta in zip(self.image_paths, self.meta_rows):
            key = self.build_key(meta["Subj_ID"], meta["Date"], meta["Time"])
            if key in self.label_map:
                filtered_paths.append(path)
                filtered_meta.append(meta)

        self.image_paths = filtered_paths
        self.meta_rows   = filtered_meta
        print(f"[OCTDatasetPassQCWithLabel] Samples: {len(self.image_paths)}")

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        key    = (sample["subj_id"], sample["date"], sample["time"])
        label  = self.label_map.get(key, None)
        if label is not None:
            label = torch.tensor(label, dtype=torch.float32)
        sample["label"] = label
        return sample


# =========================================================
# Export slices
# =========================================================
def export_slices_with_structure(dataset, output_dir):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        sample  = dataset[idx]
        volume  = sample["image"]   # (D, H, W)
        path    = sample["path"]

        parts          = path.replace("\\", "/").split("/")
        patient_folder = parts[-3]
        scan_folder    = parts[-2]

        save_folder = os.path.join(output_dir, patient_folder, scan_folder)
        os.makedirs(save_folder, exist_ok=True)

        for i in range(volume.shape[0]):
            save_path = os.path.join(save_folder, f"slice_{i}.png")
            plt.imsave(save_path, volume[i].cpu().numpy(), cmap="gray")

        print(f"Saved: {save_folder}")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = OCTDataset(root_dir="/Volumes/ONEtouch/OCT_images")
    export_slices_with_structure(dataset, "/Volumes/ONEtouch/only_OCT_images")

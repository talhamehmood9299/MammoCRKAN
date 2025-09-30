
import csv, os
import torch
from torch.utils.data import Dataset
from .transforms import read_image, right_orient, resize_keep_ratio, to_tensor, normalize_tensor

class DualViewPairs(Dataset):
    """
    CSV format:
    patient_id,side,cc_path,mlo_path,label
    """
    def __init__(self, csv_path, image_size=(2944,1920), channels=1, right_orient_flag=True, num_classes=3):
        super().__init__()
        self.items = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.items.append({
                    'patient_id': r['patient_id'],
                    'side': r.get('side', 'R'),
                    'cc': r['cc_path'],
                    'mlo': r['mlo_path'],
                    'label': int(r['label'])
                })
        self.image_size = tuple(image_size)
        self.channels = channels
        self.right_orient_flag = right_orient_flag
        self.num_classes = num_classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        cc = read_image(it['cc'], channels=self.channels)
        mlo = read_image(it['mlo'], channels=self.channels)

        if self.right_orient_flag:
            cc = right_orient(cc, it['side'])
            mlo = right_orient(mlo, it['side'])

        cc = resize_keep_ratio(cc, self.image_size)
        mlo = resize_keep_ratio(mlo, self.image_size)

        t_cc = to_tensor(cc, channels=self.channels)
        t_mlo = to_tensor(mlo, channels=self.channels)
        t_cc = normalize_tensor(t_cc)
        t_mlo = normalize_tensor(t_mlo)

        y = int(it['label'])
        return t_mlo, t_cc, torch.tensor(y, dtype=torch.long), it['patient_id']

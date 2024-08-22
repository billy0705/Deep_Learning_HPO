from torch.utils.data import Dataset
import json


class HPODataset(Dataset):

    def __init__(self, json_file):
        self.data = self.read_json(json_file)

    def read_json(self, json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
        return data

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        
        # pass
        return [X, I, C]
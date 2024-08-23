from torch.utils.data import Dataset
import json
import random
import numpy as np


class HPODataset(Dataset):

    def __init__(self, json_file, search_space_id="5971"):
        self.data = self.read_json(json_file)
        self.search_space_id = search_space_id
        self.datasets_idx, self.datasets_cnt = self.get_dataset_idx()
        self.datasets_num = len(self.datasets_idx)
        self.data_len = sum(self.datasets_cnt)
        self.probabilities = [x / self.data_len for x in self.datasets_cnt]
        print(f"{self.datasets_num=}")
        print(f"{self.datasets_idx=}")
        print(f"{self.datasets_cnt=}")

    def read_json(self, json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
        return data

    def get_dataset_idx(self):
        dataset_idx = []
        dataset_cnt = []
        for idx in self.data[self.search_space_id]:
            dataset_idx.append(idx)
            dataset_cnt.append(len(self.data[self.search_space_id][idx]["X"]))

        return dataset_idx, dataset_cnt

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        dataset_idx = np.random.choice(self.datasets_num,
                                       size=1,
                                       p=self.probabilities)[0]
        dataset_count = self.datasets_cnt[dataset_idx]
        dataset_id = self.datasets_idx[dataset_idx]
        X = self.data[self.search_space_id][dataset_id]["X"]
        y = self.data[self.search_space_id][dataset_id]["y"]
        # print(f"{self.probabilities=}")
        print(f"{dataset_count=}")
        print(f"{dataset_id=}")

        num_samples = random.randint(1, dataset_count)
        print(f"{num_samples=}")
        indices = random.sample(range(dataset_count), num_samples)
        C = [(X[i], y[i]) for i in indices]
        random_index = random.randint(0, dataset_count)
        random_x = X[random_index]
        random_y = y[random_index]
        if isinstance(random_y, list):
            random_y = random_y[0]

        improve = 1

        # Compare random_y with all y values in C
        for _, y_c in C:
            y_c = y_c[0]

            # Compare random_y with y_c
            if random_y < y_c:
                improve = 0
                break

        # Return the list [random_x, I, C]
        return [random_x, improve, C]


if __name__ == "__main__":

    dataset = HPODataset("./data/train.json")
    print(dataset[""])

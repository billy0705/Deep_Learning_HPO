from torch.utils.data import Dataset
import json
import random
import numpy as np
from src.noiser import NoiseAdder


class HPODataset(Dataset):

    def __init__(self, json_file, search_space_id="5971"):
        self.data = self.read_json(json_file)
        self.search_space_id = search_space_id
        self.datasets_idx, self.datasets_cnt = self.get_dataset_idx()
        self.datasets_num = len(self.datasets_idx)
        self.data_len = sum(self.datasets_cnt)
        self.probabilities = [x / self.data_len for x in self.datasets_cnt]
        # print(f"{self.datasets_num=}")
        # print(f"{self.datasets_idx=}")
        # print(f"{self.datasets_cnt=}")
        self.define_noiser()

    def read_json(self, json_path):
        f = open(json_path)
        data = json.load(f)
        f.close()
        return data

    def define_noiser(self, beta_timesteps=100, beta_start=0.001, beta_end=0.05,
                      schedule_method='cosine'):
        self.beta_timesteps = beta_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # schedule_method can be 'linear', 'cosine' or 'quadraric'
        self.schedule_method = schedule_method

        self.noiser = NoiseAdder(beta_start, beta_end,
                                 beta_timesteps, schedule_method)

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
        # print(f"{dataset_count=}")
        # print(f"{dataset_id=}")

        num_samples = random.randint(1, dataset_count)
        num_samples = 10
        # print(f"{num_samples=}")
        indices = random.sample(range(dataset_count), num_samples)
        # print(f"{indices=}")
        # print(f"{X[indices[0]]=}")
        # print(f"{y[indices[0]]=}")
        C = []
        for i in indices:
            x_temp = X[i].copy()
            x_temp.extend((y[i][0],))
            C.append(x_temp)
        # print(f"{C=}")
        random_index = random.randint(0, dataset_count-1)
        random_x = X[random_index]
        random_y = y[random_index][0]

        # if isinstance(random_y, list):
        #     random_y = random_y[0]

        improve = 1
        # Compare random_y with all y values in C
        for xy in C:
            # Compare random_y with y_c
            if random_y < xy[16]:
                improve = 0
                break

        t = random.randint(1, self.beta_timesteps)
        x_bar, noise = self.noiser.add_noise(np.array(random_x), t)

        # Return the list [random_x, I, C]
        return x_bar, noise, np.array(improve), np.array(C)


if __name__ == "__main__":

    dataset = HPODataset("./data/train.json")

    for x_bar, noise, improve, C in dataset:
        print(type(x_bar))
        print(type(noise))
        print(type(improve))
        print(type(C))
        print(x_bar.shape)
        print(noise.shape)
        print(improve)
        print(C.shape)
        break

import random
import numpy as np
from HPO_B.hpob_handler import HPOBHandler
import matplotlib.pyplot as plt
from src.HPO_inference import HPOInference
import torch


class RandomSearch:
    def __init__(self, model_save_path):
        print("Using custom random search method...")
        self.HPO_inference = HPOInference(model_save_path)

    # observe and suggest for continuous search space
    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        if X_pen is not None:
            set_num = X_obs.shape[0]
            x_dim = len(X_obs[0])
            y_dim = len(y_obs[0])
            dim = x_dim + y_dim
            combined = np.concatenate((X_obs, y_obs), axis=1)
            c = combined.reshape(1, set_num, dim)
            c = torch.from_numpy(c)
            # print(f"{c.shape=}")
            # c = torch.randn(1, 10, 17, dtype=torch.float64)
            x_hat = torch.randn(1, 16, dtype=torch.float64)
            x = self.HPO_inference.generator(x_hat, c)
            # print(f"{x.shape=}")
            return x.detach().numpy()

        else:
            # print(X_obs)
            # print(f"{X_obs.shape=}")
            # print(f"{y_obs.shape=}")
            set_num = X_obs.shape[0]
            x_dim = len(X_obs[0])
            y_dim = len(y_obs[0])
            dim = x_dim + y_dim
            combined = np.concatenate((X_obs, y_obs), axis=1)
            c = combined.reshape(1, set_num, dim)
            c = torch.from_numpy(c)
            # print(f"{c.shape=}")
            # c = torch.randn(1, 10, 17, dtype=torch.float64)
            x_hat = torch.randn(1, 16, dtype=torch.float64)
            x = self.HPO_inference.generator(x_hat, c)
            # print(f"{x.shape=}")
            return x.detach().numpy()


def load_dataset_and_handler():

    hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3",
                            surrogates_dir="saved-surrogates/")

    # using the firstdataset with search space ID 5971
    search_space_id = "5971"
    dataset_id = hpob_hdlr.get_datasets(search_space_id)[0] 

    return hpob_hdlr, search_space_id, dataset_id

# eval function


def evaluate_random_search():

    hpob_hdlr, search_space_id, dataset_id = load_dataset_and_handler()
    model_save_path = "./model/HPO-model-weight-32-new.pth"
    method = RandomSearch(model_save_path)
    # search_space_id = ""
    # dataset_id = ""

    # Evaluate_continous on the method
    acc = hpob_hdlr.evaluate_continuous(method, search_space_id=search_space_id,
                                        dataset_id=dataset_id, seed="test4", n_trials=50)

    plt.plot(acc)
    plt.xlabel("Trials")
    plt.ylabel("Accuracy")
    plt.show()



if __name__ == "__main__":
    evaluate_random_search()

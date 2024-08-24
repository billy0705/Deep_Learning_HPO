import random
import numpy as np
from HPO_B.hpob_handler import HPOBHandler
import matplotlib.pyplot as plt


class RandomSearch:
    def __init__(self):
        print("Using custom random search method...")

    #observe and suggest for continuous search space
    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        if X_pen is None:
            dim = len(X_obs[0])
            bounds = [(0, 1) for _ in range(dim)]
            x_new = [random.uniform(lower, upper) for lower, upper in bounds]
            return np.array(x_new).reshape(-1, dim)


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

    
    method = RandomSearch()

    # Evaluate_continous on the method
    acc = hpob_hdlr.evaluate_continuous(method, search_space_id=search_space_id,
                                        dataset_id=dataset_id, seed="test4", n_trials=1000)

    
    plt.plot(acc)
    plt.xlabel("Trials")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    evaluate_random_search()

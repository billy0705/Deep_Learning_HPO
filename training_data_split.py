import random
import json
from HPO_B.hpob_handler import HPOBHandler
import os

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3",
                        surrogates_dir="saved-surrogates/")

split_ratio = 0.2
search_space_id = "5971"

data_cnt = 0
train_data = {
    search_space_id: {}
}

test_data = {
    search_space_id: {}
}
dataset_train = {}
dataset_test = {}

for dataset_id in hpob_hdlr.meta_train_data[search_space_id]:

    print(f"{dataset_id=}")
    data_points_x = hpob_hdlr.meta_train_data[search_space_id][dataset_id]["X"]
    data_points_y = hpob_hdlr.meta_train_data[search_space_id][dataset_id]["y"]
    split_index = int(len(data_points_x) * split_ratio)
    combined = list(zip(data_points_x, data_points_y))
    random.shuffle(combined)
    # print(" x:", data_points_x)
    # print(" y:", data_points_y)

    # Unzip
    x_shuffled, y_shuffled = zip(*combined)

    x_shuffled = list(x_shuffled)
    y_shuffled = list(y_shuffled)

    dataset_train["X"] = x_shuffled[:split_index]
    dataset_train["y"] = y_shuffled[:split_index]
    dataset_test["X"] = x_shuffled[split_index:]
    dataset_test["y"] = y_shuffled[split_index:]
    train_data[search_space_id][dataset_id] = dataset_train
    test_data[search_space_id][dataset_id] = dataset_test

    data_cnt += len(data_points_x)
    # break

# print(train_data)

if not os.path.exists("./data/"):
    os.mkdir("./data/")

with open("./data/train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("./data/test.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

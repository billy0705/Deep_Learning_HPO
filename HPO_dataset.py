from torch.utils.data import Dataset
import json
import random

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
        
        num_samples = 5 # Number of elements in C
        indices = random.sample(range(len(self.X)), num_samples)
        C = [(self.X[i], self.y[i]) for i in indices]

        #random element from the original dataset
        random_index = random.randint(0, len(self.X) - 1)
        random_x = self.X[random_index]
        random_y = self.y[random_index]

        #Extractig the float value from random_y (originally a list)
        if isinstance(random_y, list):
            random_y = random_y[0]

        I = 1

        # Compare random_y with all y values in C
        for _, y_c in C:
            y_c = y_c[0]
            

            # Compare random_y with y_c
            if random_y < y_c:
                I = 0
                break  

        # Return the list [random_x, I, C]
        return [random_x, I, C]
        
    

if __name__ == "__main__":

    dataset = HPODataset("./data/train.json")
    print(dataset[""])
    search_space_id = "5971"
    dataset_id = "145972"
    print(len(dataset.data[search_space_id][dataset_id]))

    for key in dataset.data[search_space_id][dataset_id]:
        print(key)
        print(len(dataset.data[search_space_id][dataset_id][key]))
        # print(dataset.data[search_space_id][dataset_id][key])
        for i in range(len(dataset.data[search_space_id][dataset_id][key])):
            print(dataset.data[search_space_id][dataset_id][key][i])
            print(len(dataset.data[search_space_id][dataset_id][key][i]))
            if i == 10:
                break
    
    random_index = random.randint(0, len(dataset) - 1)

    # Retrieve the item using the random index
    item = dataset[random_index]

    # Print the results
    print(f"Random index: {random_index}")
    print(f"Returned item for index {random_index}: {item}")

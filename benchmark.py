
# import numpy as np
# import json
from HPO_B.benchmark_plot import BenchmarkPlotter
from HPOB_eval import RandomSearch
# from methods.pygpgo import RandomForest

if __name__ == "__main__":

    data_path = "./HPO_B/hpob-data/"
    results_path = "./HPO_B/results/"
    output_path = "./HPO_B/plots/"
    name = "DDPM_benchmark"
    new_method_name = "New-2.json"
    # experiments = ["Random", "FSBO", "TST", "DGP", "RGPE", "BOHAMIANN",
    #                "DNGO", "TAF", "GP"]
    experiments = ["DDPM", "RS-C", "DGP-C", "GP-C"]
    n_trials = 50

    # method = RandomSearch()
    model_save_path = "./model/HPO-model-weight-32.pth"
    method = RandomSearch(model_save_path)

    benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                         name=name,
                                         n_trials=n_trials,
                                         results_path=results_path,
                                         output_path=output_path,
                                         data_path=data_path)

    # seeds = ["test0", "test1", "test2", "test3", "test4"]
    # benchmark_plotter.generate_results(method, n_trials, new_method_name, search_spaces=["5971"], seeds=seeds)
    print("Ploting")
    benchmark_plotter.plot()
    benchmark_plotter.draw_cd_diagram(bo_iter=50, name="Rank@50")

import os

from cagsgp.benchmark.DatasetGenerator import DatasetGenerator


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/benchmark/'
    seed = 42

    d = DatasetGenerator.friedman1(seed=seed, reset=True, path=folder)
    print(d["training"][0][0])
    print(d["training"][0][1])
    print(d["training"][1][0])
    print(d["training"][1][1])
    d =DatasetGenerator.keijzer6(seed=seed, reset=True, path=folder)
    print(d["training"][0][0])
    print(d["training"][0][1])
    print(d["training"][1][0])
    print(d["training"][1][1])
    d = DatasetGenerator.korns12(seed=seed, reset=True, path=folder)
    print(d["training"][0][0])
    print(d["training"][0][1])
    print(d["training"][1][0])
    print(d["training"][1][1])
    d = DatasetGenerator.vladislavleva4(seed=seed, reset=True, path=folder)
    print(d["training"][0][0])
    print(d["training"][0][1])
    print(d["training"][1][0])
    print(d["training"][1][1])

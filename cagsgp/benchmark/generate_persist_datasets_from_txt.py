import os

from cagsgp.benchmark.DatasetGenerator import DatasetGenerator


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/'
    seed = 42
    dataset_names: list[str] = ['airfoil', 'bioav', 'keijzer6', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva14', 'yacht']

    for name in dataset_names:
        for i in range(1, 100 + 1):
            DatasetGenerator.convert_text_dataset_to_csv(input_path=folder+'datasets/' + name +'/',
                                                         output_path=folder+'datasets_csv/' + name + '/',
                                                         idx=i,
                                                         scale=False)
            DatasetGenerator.read_csv_data(path=folder+'datasets_csv/' + name + '/', idx=i)

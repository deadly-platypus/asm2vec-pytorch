import yaml
import click
import os
import subprocess
import torch
import pickle

import asm2vec


class Training:
    def __init__(self):
        self.model_path = None
        self.binary_path = None
        self.function_path = None
        self.results = list()

    def add_result(self, training_func, test_func, similarity):
        self.results.append((training_func, test_func, similarity))

    def __hash__(self):
        return hash(self.model_path, self.binary_path, self.function_path)

    def __eq__(self, other):
        return self.binary_path == other.binary_path


class TestBinary:
    def __init__(self):
        self.binary_path = None
        self.function_path = None

    def __hash__(self):
        return hash(self.binary_path, self.function_path)

    def __eq__(self, other):
        return self.binary_path == other.binary_path


def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


def generate_assembly(binary: str, output_dir: str):
    bin2asm_path = os.path.join(os.path.dirname(__file__), 'bin2asm.py')
    cmd = ['python3', bin2asm_path, '-i', binary, '-o', output_dir]
    print(f"Generating assembly for {binary} using {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)


def generate_model(function_path: str, model_path: str):
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    cmd = ['python3', train_path, '-i', function_path, '-o', model_path, '--epochs',
           str(100)]
    print(f"Generating model {model_path} using {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)


def compare_functions(func1, func2, mpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    lr = 0.02

    # load model, tokens
    model, tokens = asm2vec.utils.load_model(mpath, device=device)
    functions, tokens_new = asm2vec.utils.load_data([func1, func2])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)

    # train function embedding
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        device=device,
        mode='test',
        learning_rate=lr
    )

    # compare 2 function vectors
    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
    return cosine_similarity(v1, v2)


@click.command()
@click.option('-i', '--input', 'ipath', help='input yaml', required=True)
@click.option('-o', '--output', 'opath', help='output object', required=True)
def main(ipath, opath):
    with open(ipath, 'r') as f:
        experiment_yaml = yaml.safe_load(f)

    trainings = list()
    for yaml_training in experiment_yaml['training']:
        dest_path = yaml_training['dest_path']
        training_name = yaml_training['name']
        root_name = yaml_training['root_dir']
        for binary in yaml_training['binaries']:
            training = Training()
            training.model_path = os.path.join(dest_path, training_name, binary,
                                               "model.pt")
            training.binary_path = os.path.join(root_name, binary)
            training.function_path = os.path.join(os.path.dirname(training.model_path),
                                                  'asm')
            if not os.path.exists(training.binary_path):
                raise FileNotFoundError(training.binary_path)

            if not os.path.exists(training.function_path):
                os.makedirs(training.function_path, exist_ok=True)
            if not os.path.exists(os.path.dirname(training.model_path)):
                os.makedirs(os.path.dirname(training.model_path), exist_ok=True)

            trainings.append(training)

    tests = list()
    for yaml_test in experiment_yaml['experiments']:
        test_name = yaml_test['name']
        root_name = yaml_test['root_dir']
        dest_path = yaml_test['dest_path']
        for binary in yaml_test['binaries']:
            test_bin = TestBinary()
            test_bin.binary_path = os.path.join(root_name, binary)
            test_bin.function_path = os.path.join(dest_path, test_name, binary, 'asm')

            if not os.path.exists(test_bin.binary_path):
                raise FileNotFoundError(test_bin.binary_path)
            if not os.path.exists(test_bin.function_path):
                os.makedirs(test_bin.function_path, exist_ok=True)
            tests.append(test_bin)

    # for training in trainings:
    #     generate_assembly(training.binary_path, training.function_path)
    #     if not os.path.exists(training.model_path):
    #         generate_model(training.function_path, training.model_path)
    # for test in tests:
    #     generate_assembly(test.binary_path, test.function_path)

    for training in trainings:
        for test in tests:
            print(f"Comparing {training.binary_path} with {test.binary_path}")
            for _, _, training_funcs in os.walk(training.function_path):
                for _, _, test_funcs in os.walk(test.function_path):
                    for training_func in training_funcs:
                        for test_func in test_funcs:
                            training_func = os.path.join(training.binary_path,
                                                         training_func)
                            test_func = os.path.join(test.function_path, test_func)
                            similarity = compare_functions(training_func, test_func,
                                                           training.model_path)
                            training.add_result(training_func, test_func, similarity)
    with open(opath, 'wb') as f:
        pickle.dump(trainings, f)


if __name__ == '__main__':
    main()

import yaml
import click
import os
import subprocess
import torch
import pickle
import concurrent.futures
import multiprocessing
import re
import time
from sklearn.metrics import f1_score
import statistics

import asm2vec


class Training:
    def __init__(self):
        self.model_path = None
        self.binary_path = None
        self.function_path = None

    def __hash__(self):
        return hash((self.model_path, self.binary_path, self.function_path))

    def __eq__(self, other):
        return self.binary_path == other.binary_path


class TestBinary:
    def __init__(self):
        self.binary_path = None
        self.function_path = None
        self.results = dict()

    def add_function_similarity(self, function_file, training, trained_func, similarity):
        if function_file not in self.results:
            self.results[function_file] = dict()
        if training not in self.results[function_file]:
            self.results[function_file][training] = dict()
        self.results[function_file][training][trained_func] = similarity

    def __hash__(self):
        return hash((self.binary_path, self.function_path))

    def __eq__(self, other):
        return self.binary_path == other.binary_path


def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


def get_asm_function_name(function_path: str):
    name_regex = re.compile('\.name ([\w\.]+)')
    with open(function_path, 'r') as f:
        for line in f.readlines():
            match = name_regex.match(line.strip())
            if match:
                return match[1]
    return ""


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
@click.option('-o', '--output', 'opath', help='output object')
@click.option('-p', '--print-results', 'print_results',
              help='Run experiment or print results', default=False, is_flag=True)
@click.option('--skip-assembly', 'skip_assembly', default=False, is_flag=True)
@click.option('--compute-fscore', "compute_f_score", help='Compute F-Score of results',
              default=False, is_flag=True)
def main(ipath, opath, print_results, skip_assembly, compute_f_score):
    if not print_results and not compute_f_score:
        if opath is None:
            raise RuntimeError("Output must be specified")
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
                training.function_path = os.path.join(
                    os.path.dirname(training.model_path),
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

        for training in trainings:
            if not skip_assembly:
                generate_assembly(training.binary_path, training.function_path)
            if not os.path.exists(training.model_path):
                generate_model(training.function_path, training.model_path)
        if not skip_assembly:
            for test in tests:
                generate_assembly(test.binary_path, test.function_path)

        total_tests = 0
        start = time.time()
        for test in tests:
            test_funcs = [os.path.realpath(os.path.join(test.function_path, f)) for f
                          in os.listdir(test.function_path) if
                          os.path.isfile(os.path.join(test.function_path, f))]
            for training in trainings:
                training_funcs = [
                    os.path.realpath(os.path.join(training.function_path, f))
                    for f in os.listdir(training.function_path) if
                    os.path.isfile(os.path.join(training.function_path, f))]
                for test_func in test_funcs:
                    with concurrent.futures.ThreadPoolExecutor(
                            max_workers=multiprocessing.cpu_count()) as executor:
                        completed = {executor.submit(compare_functions, test_func,
                                                     training_func,
                                                     training.model_path): training_func
                                     for training_func in training_funcs}
                        for future in concurrent.futures.as_completed(completed):
                            training_func = completed[future]
                            try:
                                print(f"Completed {training_func}: {future.result()}")
                                total_tests += 1

                                test.add_function_similarity(test_func, training,
                                                             training_func,
                                                             future.result())
                            except Exception as e:
                                print(f"Error identifying {test_func} using "
                                      f"{training_func}")
        end = time.time()
        print(f"Total tests: {total_tests}")
        print(f"Total time: {end - start} seconds")
        if not os.path.exists(os.path.dirname(opath)):
            os.makedirs(os.path.dirname(opath), exist_ok=True)
        with open(opath, 'wb') as f:
            pickle.dump(tests, f)
    elif print_results:
        with open(ipath, 'rb') as f:
            tests = pickle.load(f)
        total_tests = 0
        for test in tests:
            print(f'{test.binary_path}:')
            for function_file, results in test.results.items():
                function_name = get_asm_function_name(function_file)
                print(f"\t{function_name}: ")
                for trained_binary, function_similarities in results.items():
                    index = 0
                    found = False
                    found_similarity = None
                    for trained_func, similarity in sorted(function_similarities.items(),
                                                           reverse=True,
                                                           key=lambda a: a[1]):
                        candidate_name = get_asm_function_name(trained_func)
                        if candidate_name == function_name:
                            found = True
                            found_similarity = similarity
                            break

                        index += 1
                    if found and index < 1:
                        print(f"\t\t{trained_binary.binary_path}: Y! {found_similarity}")
                    else:
                        print(f"\t\t{trained_binary.binary_path}: N! {index} / {len(results)}")
        print(f"Total tests: {total_tests}")
    elif compute_f_score:
        with open(ipath, 'rb') as f:
            tests = pickle.load(f);

        for test in tests:
            ground_truth = dict()
            predictions = dict()
            found_similarities = dict()
            for function_file, results in test.results.items():
                true_function_name = get_asm_function_name(function_file)
                for trained_binary, function_similarities in results.items():
                    if trained_binary not in ground_truth:
                        ground_truth[trained_binary] = list()
                    if trained_binary not in predictions:
                        predictions[trained_binary] = list()
                    if trained_binary not in found_similarities:
                        found_similarities[trained_binary] = list()

                    index = 0
                    found = False
                    found_similarity = None
                    label_name = None
                    for trained_func, similarity in sorted(function_similarities.items(),
                                                           reverse=True,
                                                           key=lambda a: a[1]):
                        candidate_name = get_asm_function_name(trained_func)
                        if index == 0:
                            label_name = candidate_name
                        if candidate_name == true_function_name:
                            found = True
                            if index < 2:
                                label_name = candidate_name
                                found_similarity = similarity
                        index += 1
                    if not found:
                        true_function_name = "!!UNKNOWN!!"
                    ground_truth[trained_binary].append(true_function_name)
                    predictions[trained_binary].append(label_name)
                    if found_similarity:
                        found_similarities[trained_binary].append(found_similarity)
            print(test.binary_path)
            for trained_binary in ground_truth.keys():
                ground_truth_labels = ground_truth[trained_binary]
                prediction_labels = predictions[trained_binary]
                print(f'\t{trained_binary.binary_path}: '
                      f'{f1_score(ground_truth_labels, prediction_labels, average=None)}')
                print(f'\tMean found similarity: '
                      f'{statistics.mean(found_similarities[trained_binary])}')
                print()


if __name__ == '__main__':
    main()

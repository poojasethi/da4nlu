import re
import argparse
import os
from typing import Dict, OrderedDict

# Expand each subdirectory and get the confidence scores.

CIVIL_COMMENTS_TEST_REGEX = "Results for: test\nAverage acc: (0.\d+)"
AMAZON_TEST_REGEX = "TODO"

"""
Prints out the results in a pretty-format for LaTeX. 
"""

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--results-path", required=True, help="Path to results, e.g. results/civilcomments/100_percent"
    )
    config = parser.parse_args()
    return config

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def find_accuracy(text: str) -> float:
    match = re.search(CIVIL_COMMENTS_TEST_REGEX, text)
    if not match:
        raise ValueError("Did not find accuracy!")
    return float(match.groups()[0])

def latex_format_results(results: Dict[str, Dict[int, float]]) -> str:
    pretty_results = ""

    for strategy, accuracies in results.items():
        pretty_results_line = f"{strategy.replace('_', ' ').title()}"
        for iteration, accuracy in accuracies.items():
            pretty_results_line += f" & {accuracy}" 
        pretty_results += f"{pretty_results_line}\n"
    
    return pretty_results

def main():
    config = get_config()
    results = os.walk(config.results_path)
    results = OrderedDict()

    iteration_dirnames = get_immediate_subdirectories(config.results_path)
   
    for iteration in iteration_dirnames: 
        strategy_dirnames = get_immediate_subdirectories(os.path.join(config.results_path, iteration))

        for strategy in strategy_dirnames:
            with open(os.path.join(config.results_path, iteration, strategy, "results.txt"), 'r') as fh:
                text = fh.read()
                if strategy not in results:
                    results[strategy] = {}
                results[strategy][int(iteration)] = find_accuracy(text)
    

    # Go back and populate the accuracy for the 0th iteration.
    zero_iteration_accuracy = float('nan')
    with open(os.path.join(config.results_path, "0", "results.txt"), 'r') as fh:
        text = fh.read()
        zero_iteration_accuracy = find_accuracy(text)
    

    for strategy in results.keys():
        iterations = results[strategy]
        iterations[0] = zero_iteration_accuracy
        results[strategy] = OrderedDict(sorted(iterations.items()))
    
    print(latex_format_results(results))
        

if __name__ == "__main__":
    main()


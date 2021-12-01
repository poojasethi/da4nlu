import re
import argparse
import os
from typing import Dict, OrderedDict

# Expand each subdirectory and get the confidence scores.

TEST_AVG_ACCURACY_REGEX = "Results for: test\nAverage acc: (0.\d+)"
WORST_GROUP_ACCURACY_REGEX = "Results for: test(.|\n)*(Worst-group acc:) (0.\d+)*" 
TENTH_PERCENTILE_ACCURACY_REGEX = "Results for: test(.|\n)*(Worst-group acc:) (0.\d+)*" 

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


def find_accuracy(text: str, regex: str) -> float:
    match = re.search(regex, text)
    if not match:
        raise ValueError("Did not find accuracy!")

    if regex == TEST_AVG_ACCURACY_REGEX:    
        return float(match.groups()[0])
    elif regex == WORST_GROUP_ACCURACY_REGEX:
        return float(match.groups()[2])
    else:
        raise ValueError("Did not recognize regex")

def latex_format_results(results: Dict[str, Dict[int, float]]) -> str:
    pretty_results = ""

    for strategy, accuracies in results.items():
        pretty_results_line = f"{strategy.replace('_', ' ').title()}"
        for iteration, accuracy in accuracies.items():
            pretty_results_line += f" & {accuracy}" 
        pretty_results += f"{pretty_results_line} \\\\\n"
    
    return pretty_results

def get_accuracy_results(results_path: str, regex: str):
    results = os.walk(results_path)
    results = OrderedDict()

    iteration_dirnames = get_immediate_subdirectories(results_path)
   
    for iteration in iteration_dirnames: 
        strategy_dirnames = get_immediate_subdirectories(os.path.join(results_path, iteration))

        for strategy in strategy_dirnames:
            with open(os.path.join(results_path, iteration, strategy, "results.txt"), 'r') as fh:
                text = fh.read()
                if strategy not in results:
                    results[strategy] = {}
                results[strategy][int(iteration)] = find_accuracy(text, regex)
    

    # Go back and populate the accuracy for the 0th iteration.
    zero_iteration_accuracy = float('nan')
    with open(os.path.join(results_path, "0", "results.txt"), 'r') as fh:
        text = fh.read()
        zero_iteration_accuracy = find_accuracy(text, regex)
    

    for strategy in results.keys():
        iterations = results[strategy]
        iterations[0] = zero_iteration_accuracy
        results[strategy] = OrderedDict(sorted(iterations.items()))
    
    return results

def main():
    config = get_config()
    average_accuracy = get_accuracy_results(config.results_path, TEST_AVG_ACCURACY_REGEX)
    worst_group_accuracy = get_accuracy_results(config.results_path, WORST_GROUP_ACCURACY_REGEX) 
    print("Test Average Accuracy")
    print(latex_format_results(average_accuracy))
    print()

    print("Worst-Group Accuracy")
    print(latex_format_results(worst_group_accuracy))
        

if __name__ == "__main__":
    main()


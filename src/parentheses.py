"""
parentheses.py

This module provides a generator function for producing balanced parentheses strings 
and includes a test function to demonstrate its usage 
and a function that checks if the next character is a closed parentheses or End Character.
"""
import random
import math
import numpy as np
from constants import *

def generate_balanced_parentheses(max_depth: int = MAX_STACK_DEPTH, max_length: int = MAX_STRING_LENGTH) -> str:
    """
    Generator function to yield balanced parentheses strings up to a specified maximum depth and length.
    
    Args:
        max_depth (int): Maximum depth of nested parentheses.
        max_length (int): Maximum length of the generated string.
    Yields:
        str: A balanced parentheses string.
    """
    return_string = "()"
    for _ in range(math.floor(random.triangular(2, max_length - 1,max_length - 1))):
        index = int(np.random.randint(0, len(return_string) - 1))
        while check_current_depth(return_string[:index]) >= max_depth:  #slow but works
            index = int(np.random.randint(0, len(return_string) - 1))
        return_string = return_string[:index + 1] + "()" + return_string[index + 1:]
    return return_string

def test_generate_balanced_parentheses():
    """
    Test function to demonstrate the usage of the generate_balanced_parentheses function.
    It generates a specified number of balanced parentheses strings and prints them.
    """
    num_samples = 10
    samples = list(generate_balanced_parentheses(max_depth=MAX_STACK_DEPTH, max_length=MAX_STRING_LENGTH) for _ in range(num_samples))
    for sample in samples:
        print(sample)

def check_current_depth(parentheses_string: str) -> int:
    """
    Function to check the current depth of nested parentheses in a given string.
    """
    depth = 0
    for char in parentheses_string:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
    return depth

def replace_random_parentheses(parentheses_string: str) -> tuple[str,str]:
    """
    Function to replace a random parentheses in the string with a '?'.
    """
    if len(parentheses_string) <= 1:
        return parentheses_string
    index = np.random.randint(0, len(parentheses_string) - 1)
    deleted_parentheses = parentheses_string[index]
    return (parentheses_string[:index] + '?' + parentheses_string[index + 1:],deleted_parentheses)

def generate_testcase_with_deletion( max_depth: int = MAX_STACK_DEPTH, max_length: int = MAX_STRING_LENGTH) -> tuple[str,str]:
    """
    Function to generate a balanced parentheses string and then delete a random parentheses.
    """
    balanced_string = generate_balanced_parentheses(max_depth, max_length)
    return replace_random_parentheses(balanced_string)

def generate_testcase_with_cut( max_depth: int = MAX_STACK_DEPTH, max_length: int = MAX_STRING_LENGTH) -> tuple[str,str]:
    """
    Function to generate a balanced parentheses string and then delete a random parentheses.
    """
    balanced_string = generate_balanced_parentheses(max_depth, max_length)
    cut_string = balanced_string[:-np.random.randint(0, min(len(balanced_string),MAX_CUT))]
    depth = check_current_depth(cut_string)
    return (cut_string  , ")" * depth)

def generate_testdata_set_with_deletion(number_of_entries: int,
                                         path: str,
                                         max_depth: int = MAX_STACK_DEPTH,
                                         max_length: int = MAX_STRING_LENGTH,)-> list[tuple[str,str]]:
    """
    Function to generate a dataset of balanced parentheses strings with random deletions.
    """
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(number_of_entries):
            test_case, deleted_parentheses = generate_testcase_with_deletion(max_depth, max_length)
            f.write(f"{test_case},{deleted_parentheses}\n")
            
def generate_testdata_set_with_cut(number_of_entries: int,
                                         path: str,
                                         max_depth: int = MAX_STACK_DEPTH,
                                         max_length: int = MAX_STRING_LENGTH,)-> list[tuple[str,str]]:
    """
    Function to generate a dataset of balanced parentheses strings with random deletions.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"feature,label")
        for _ in range(number_of_entries):
            test_case, deleted_parentheses = generate_testcase_with_cut(max_depth, max_length)
            f.write(f"\n{test_case},{deleted_parentheses}")

vocab = {'(':0, ')':1, '[':2, ']':3, '{':4, '}':5, '?':6}
inv_vocab = {v:k for k,v in vocab.items()}

def tokenize(seq):
    return [vocab[c] for c in seq]

if __name__ == "__main__":
    generate_testdata_set_with_cut(5000,"belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_?_paranthesees.csv")
    generate_testdata_set_with_cut(5000,"belief-state-geometry-in-the-residual-stream/src/data/testdata/test_?_data.csv")

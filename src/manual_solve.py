#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

# The input is a 9X9 matrix, output is the top right corner values in 3x3 matrix.
def solve_5bd6f4ac(x):
    # Slicing the input matrix
    x = x[:3,-3:]
    return x
	
def solve_3c9b0459(x):
    result = list()
    #swap columns
    for indx in range(len(x)):
        first_ele = x[indx][0]
        last_ele = x[indx][-1]
        x[indx][0] = last_ele
        x[indx][-1] = first_ele
        result.append(x[indx])
    
    first_row = result[0]
    last_row = result[-1]
    #swap rows
    result[-1] = first_row
    result[0] = last_row
    x = np.array(result)
    return x

def solve_b91ae062(x):
    # Find the number of unique colours
    unique = np.unique(x)
    # unique colours except black
    unique_colours = len(list(unique)) - 1
    result_col = list()
    for row in x:
        ele_list = list()
        for ele in row:
            for reps in range(0,unique_colours):
                ele_list.append(ele)
        result_col.append(ele_list)
    
    result_row = list()
    for r in result_col:
        for rep in range(0,unique_colours):
            result_row.append(r)
                        
    x = np.array(result_row)
        
    return x	

def solve_d9fac9be(x):
    indicator = 0
    for val in range(len(x)):
        for indx in range(len(x[val])):
            if indx <= len(x[val])-3:
                # check for the repeating pattern
                if x[val][indx] != 0 and list(x[val][indx:indx+3]) == [x[val][indx]]*3:
                    sliced_matrix = x[val:val+3,indx:indx+3]
                    indicator = 1
                    break
        if indicator == 1:
            break
            
    x = np.array(sliced_matrix[1][1]).reshape(1,1)
    return x
	
def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()


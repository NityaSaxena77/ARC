#!/usr/bin/python

#Name: Nitya Saxena
#Student ID: 21230046
#GIT Link: https://github.com/NityaSaxena77/ARC

import os, sys
import json
import numpy as np
import re
# Additional libraries for solve functions.
from itertools import permutations

"""
Summary of solve functions:

For this assignment, the matrix related problems were solved using python slicing techniques and few of the numpy library functions. Often times the multidimentional arrays require many looping statements. However, most of the solve functions were resolved using the simple for-loops and once using the itertools library.

Summary of additional libraries/functions used is as follows:
np.unique : The unique function proved useful for many of the solve functions to find the unique colour in the matrix or in any slice.

np.column_stack: This function was used an efficient way to merge many arrays of different shape in a columnwise manner. This was useful in the solve functions to combine the different matrix to find the resultant matrix.

np.concatenate: This was another function used to join the arrays of same shape along an axis.

itertools.permutations: In one of the solve function, it was required to iterate through all possible permutations and combinations of the list of elements where this itertools function came handy.

np.count_nonzero: This function was used to count the number of cells in the input grid which do not have a zero value (i.e are not represented by the colour black)

"""

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

# The input is a 9X9 matrix, output is the top right corner values in 3x3 matrix.

"""
5bd6f4ac.json: The input is a 9X9 matrix of random colours. The solution to this is the top right corner values in 3x3 matrix.

The solve function carries out the transformation by slicing the input matrix from the top right to produce a 3x3 matrix.

All the training and test cases are solved correctly.
"""
def solve_5bd6f4ac(x):
    # Slicing the input matrix
    x = x[:3,-3:]
    return x

"""
3c9b0459.json: The input is a 3x3 matrix with different colours. The solution is to rotate the matrix clockwise twice.

To achieve the clockwise rotation twice, the solve function first swaps the columns of the input matrix followed by the rows of the matrix.

All the training and test cases are solved correctly.
"""	
def solve_3c9b0459(x):
    result = list()
    #swap columns of the matrix
    for indx in range(len(x)):
        first_ele = x[indx][0]
        last_ele = x[indx][-1]
        x[indx][0] = last_ele
        x[indx][-1] = first_ele
        result.append(x[indx])
    
    first_row = result[0]
    last_row = result[-1]
    #swap rows of the matrix
    result[-1] = first_row
    result[0] = last_row
    x = np.array(result)
    return x

"""
b91ae062.json: The input is a 3x3 matrix with different colours. 
The solution is a matrix where the shape of each cell of the input matrix depends on the number of unique colours. Each cell of the input grid is repeated by the number of unique colours. The result matrix is of the shape (2*number of unique colours)x(2*number of unique colours).

After fincing all the unique colours, the solve function iterates through the all the cells and repeats each cell by the number of unique colour.

All the training and test cases are solved correctly.


"""
def solve_b91ae062(x):
    # Find the number of unique colours
    unique = np.unique(x)
    # number of unique colours except black
    unique_colours = len(list(unique)) - 1
    result_col = list()
    # Fill the colours for each column.
	for row in x:
        ele_list = list()
        for ele in row:
		# Each colour in the input row is repeated as the number of unique colours except black.
            for reps in range(0,unique_colours):
                ele_list.append(ele)
        result_col.append(ele_list)
    
    # Fill the colours for each row.
	result_row = list()
    for r in result_col:
        for rep in range(0,unique_colours):
            result_row.append(r)
                        
    x = np.array(result_row)
        
    return x	

"""
d9fac9be.json: The input is a matrix of varied shape. The matrix contains a figure of a 3x3 matrix.  
The solution is the center element of this 3x3 matrix.

The solve function will find the pattern for the 3x3 matrix inside the input matrix by using matrix slicing. Next, the center element of this matrix will be returned as the output.

All the training and test cases are solved correctly.
"""	
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
        # if the repeating pattern is found, stop the loop. This will be the result.
		if indicator == 1:
            break
    # center element of the sliced matrix        
    x = np.array(sliced_matrix[1][1]).reshape(1,1)
    return x


"""
25d487eb.json: The inputs are matrix of varried shape containing a shape with 2 colours. One of the colour occurs only once.
The solution is to fill all all the cells of the grid till the border in the opposite direction of the unique colour.

After fincing the unique colour which occurs only once in the input matrix, the solve function finds the direction in which this unique colour needs to be filled. The direction and the position upto which the unique colour must be filled is found by using matrix slicing.

All the training and test cases are solved correctly.
"""
def solve_25d487eb(x):
    # detect the anomaly colour
    anomaly_colour = 0
    unique, counts = np.unique(x, return_counts=True)
    unique = list(unique)
    counts = list(counts)
    unique_val = unique[counts.index(1)]
    col = 0
    row_x, col_x = x.shape
    
    # check the position of the unique value and the direction in which we need to fill the values.
    for idx in range(len(x)):
        if unique_val in x[idx]:
            col = list(x[idx]).index(unique_val)
            # check in which direction should the values be added.
            # Values are added in the opposite direction 
            if x[idx][col - 1] == 0:
                # Fill the values to the right of the last occurance of coloured value
				# Find the index of the last coloured cell.
                index_last_non_zero = np.max(np.nonzero(x[idx]))
                diff_indx = col_x - (index_last_non_zero+1)
                temp_list = list(x[idx][:index_last_non_zero+1]) + list(([unique_val] * diff_indx))
                x[idx] = np.array(temp_list)
                break
            elif x[idx][col + 1] == 0:
                # Fill the values to the left from the start of the row to the first occurance of coloured value.
				# Find the index of the first coloured cell.
                index_first_non_zero = np.min(np.nonzero(x[idx]))
                temp_list = list([unique_val] * (index_first_non_zero)) + list(x[idx][index_first_non_zero:])
                x[idx] = np.array(temp_list)
                break
            elif x[idx + 1][col] == 0:
                # Fill the values in the columnwise from the start of the column to the first occurance of the colour value
                left_col = x[:,:col]
                right_col = x[:,col+1:]
                col_val = x[:,col]
				#Find the index of the first coloured cell.
                index_first_non_zero = np.min(np.nonzero(col_val))
                temp_list = list([unique_val] * (index_first_non_zero)) + list(col_val[index_first_non_zero:])
                x = np.column_stack((left_col,temp_list, right_col))
                break
            elif x[idx - 1][col] == 0:
                # # Fill the values in the columnwise from the last occurance of the colour value to the end of the column
                left_col = x[:,:col]
                right_col = x[:,col+1:]
                col_val = x[:,col]
				# Find the index of the last coloured cell.
                index_last_non_zero = np.max(np.nonzero(col_val))
                diff_indx = row_x - (index_last_non_zero+1)
                temp_list = list(col_val[:index_last_non_zero+1]) + list(([unique_val] * diff_indx))
                x = np.column_stack((left_col, temp_list, right_col))
                break
        
    return x

"""
a61ba2ce.json: The input is a 13x13 matrix consisting of various figures of shape 2x2. 
The solution is to join these sub-matrix such that the center elements of the solution matrix are all black. The solution matrix is of the shape 4x4. 

The solve function finds and stores all the 2x2 matrices in the input matrix. All the permutations of these slices are verified, if any of the matrix results in the center elements being 0. If the center element is 0 for any of them, this is the output matrix.

All the training and test cases are solved correctly.
"""	
def solve_a61ba2ce(x):
    silce_matrix_list = list()
    colour_history = list()
    for row in range(len(x)):
        for ele in x[row]:
            # Find all the submatrix. Since we are itterating through all the rows, a history for the colours is maintained such that the matrix for one colour is not sliced twice.
            if ele != 0 and ele not in colour_history:
                colour_history.append(ele)
                indx_ele = list(x[row]).index(ele)
				# Slice a 2x2 matrix of the colour from the current index.
                slice_matrix = x[row:row+2, indx_ele: indx_ele+2]
                count = np.count_nonzero(slice_matrix)
                # Check if the matrix is sliced correctly. If not, slice the 2x2 matrix again, the current index being the end of the matrix.
				if count != 3:
                    slice_matrix = x[row:row+2, indx_ele-1: indx_ele+1]
                silce_matrix_list.append(slice_matrix)
    
    # Check all the permutations of the sliced matrix and see if the combination of any of them results in the middle elements being 0
    for ele in permutations(silce_matrix_list):
        ele_list = list(ele)
        left = np.column_stack(ele[:len(ele_list)//2])
        right = np.column_stack(ele[len(ele_list)//2:])
        result = np.concatenate((left, right), axis=0)
        if np.all((result[1:3,1:3] == 0)):
            x = result
            break
    
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


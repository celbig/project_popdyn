"""
A set of helper function for the kuznetsov model

credits:
    written by Célestin BIGARRÉ for the cellular population dynamics course
    (M2 Maths en action UCBL Lyon 1)
    2020

license: CC BY SA
"""
import numpy as np

def add_ss_to_list(ss, ss_list):
    """
    create a dict of list (for plotting purpose)
    INPUTS:
        ss -> a steady state in dict form
        ss_list -> a dict of list (fields are the same as for steady state)
    OUTPUTS:
        dict of list of length n+1
    """
    if not ss_list:
        for key, value in ss.items():
            ss_list[key] = [value]

    for key, value in ss.items():
        ss_list[key] = ss_list[key] + [value]

    return ss_list

def function_intersect(x, y1, y2):
    """
    intesect of two 1-D functions computed on the same grid
    INPUTS :
       x : the grid over wich the function are computed
       y1 : values of the first function over the grid x
       y2 : values of the second function over the grid x
    OUTPUTS :
       approx x values of intersections (not necessarly on the grid)
    """
    eps = max(np.amax(np.abs(np.diff(y1))), np.abs(np.amax(np.diff(y2))))/2


    y_diff = y1-y2
    diff_abs = np.abs(y1-y2)

    ind = np.arange(0,len(y1))
    ind = ind[diff_abs < eps]

    x_intersects = []

    prev = 0
    for j in ind:
        if j == 0 :
            if np.sign(y_diff[j]) != np.sign(y_diff[j+1]):
                x_intersects += [np.mean([x[j], x[j+1]])]
        elif j == len(y1) - 1 :
            if np.sign(y_diff[j-1]) != np.sign(y_diff[j]):
                x_intersects += [np.mean([x[j-1], x[j]])]
        else :
            if prev + 1 != j and np.sign(y_diff[j-1]) != np.sign(y_diff[j]):
                x_intersects += [np.mean([x[j-1], x[j]])]
            if np.sign(y_diff[j]) != np.sign(y_diff[j+1]):
                x_intersects += [np.mean([x[j], x[j+1]])]
        prev = j

    return np.array(x_intersects)

import numpy as np
import time

def solve_grid(grid):
    """
    Solve a Sudoku grid
    :param grid:
    :return:
    """
    grid = np.array(grid, copy=True)
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                for k in range(1, 10):
                    grid[i][j]=k
                    if is_ok(grid):
                        result = solve_grid(grid)
                        if result is not None:
                            return result
                        grid[i][j]=0
                return None
    return grid

def solve_grid_opt(grid):
    """
    Solve a Sudoku grid
    :param grid:
    :return:
    """
    grid = np.array(grid, copy=True)
    solve_nrec(grid)
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                poss = find_poss(grid, i, j)
                if len(poss) == 0:
                    return None
                elif len(poss) == 1:
                    grid[i][j] = poss[0]
                else:
                    for k in poss:
                        grid[i][j] = k
                        result = solve_grid(grid)
                        if result is not None:
                            return result
                        grid[i][j] = 0
                    return None
    return grid

def solve_nrec(grid):
    """
    Solve all that can be solved without recursion
    :param grid:
    :return:
    """
    for i in range(9):
        for j in range(9):
            if grid[i][j]==0:
                poss = find_poss(grid, i, j)
                if len(poss) == 1:
                    grid[i][j] = poss[0]

def find_poss(grid, i, j):
    """
    Find all possible
    :param grid:
    :return:
    """
    if grid[i][j]==0:
        poss = range(1, 10)

        cornerI = i/3
        cornerJ = j/3
        for k in range(cornerI, cornerI+3):
            for l in range(cornerJ, cornerJ+3):
                if grid[k][l]!=0 and (k!=i or l!=j):
                    poss.remove(grid[k][l])

        for k in range(9):
            if grid[i][k]!=0 and grid[i][k] in poss:
                poss.remove(grid[i][k])
            if grid[k][j]!= 0 and grid[k][j] in poss:
                poss.remove(grid[k][j])

        return poss


def is_ok(grid):
    """
    Check if the grid is OK (ie valid)
    :param grid:
    :return:
    """
    for i in range(0, 9):
        foundCol = []
        foundLi = []
        for j in range(0,9):
            col = grid[i][j]
            li = grid[j][i]
            if (col!=0 and col in foundCol) or (li!=0 and li in foundLi):
                return False
            else:
                foundCol.append(col)
                foundLi.append(li)

    for i in range(0, 10, 3):
        for j in range(0, 10, 3):
            found = []
            for k in range(3):
                for l in range(3):
                    case = grid[k][l]
                    if case!=0 and case in found:
                        return False
                    else:
                        found.append(case)

    return True

grid = np.array([[3, 2, 0, 0, 8, 0, 0, 0, 0],
                 [0, 0, 0, 9, 2, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 9, 0, 8],
                 [0, 0, 0, 0, 0, 0, 7, 0, 3],
                 [6, 0, 0, 8, 3, 1, 0, 0, 4],
                 [1, 0, 5, 0, 0, 0, 0, 0, 0],
                 [5, 0, 4, 0, 0, 6, 0, 0, 0],
                 [0, 0, 0, 0, 9, 7, 0, 0, 0],
                 [0, 0, 0, 0, 5, 0, 0, 3, 6]], np.uint8)
grid2 = np.array(grid, copy=True)

start = time.time()
res = solve_grid_opt(grid)
end = time.time()
print res
print end-start

'''
start = time.time()
res = solve_grid(grid2)
end = time.time()
print res
print is_ok(res)
print end-start
'''

import numpy as np


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
                return None
    return grid


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

print solve_grid(grid)
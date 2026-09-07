import numpy as np

from optix.simplex.tableau import to_tableau


def pivot(input_tbl, debug=False):
    def log(v):
        if debug:
            print(v)

    tbl = input_tbl.copy()
    col_pivot = tbl[-1, :-1].argmin()
    log(tbl[-1, :-1])
    if tbl[-1, :-1][col_pivot] >= 0:
        log("No more pivots possible")
        return tbl, True
    log(f"Pivoting on index {col_pivot} ({tbl[-1, :-1][col_pivot]}) as its the lowest value")

    pivot_candidates = tbl[:-1, -1, None] / tbl[:-1, col_pivot, None]
    log(pivot_candidates)
    masked_pivots = np.where(pivot_candidates > 0, pivot_candidates, np.inf)
    row_pivot = masked_pivots.argmin()
    if np.all(np.isinf(masked_pivots)):
        raise Exception("Problem is unbounded")
    log(f"row pivot on {row_pivot}")

    log(f"Pivots: Row: {row_pivot}, Col: {col_pivot}")

    # Divide row by coefficient in pivot_col to reduce that coefficient to 1
    reduction_factor = tbl[row_pivot][col_pivot]
    tbl[row_pivot] = tbl[row_pivot] / reduction_factor

    log(tbl)
    # Now zero out coefficient in non pivot_row rows
    for idx, row in enumerate(tbl):
        if idx == row_pivot:
            continue
        coeff = row[col_pivot]
        row_to_add = (-1 * coeff) * tbl[row_pivot]
        tbl[idx] = row_to_add + row
    log(tbl)
    return tbl, False


def solve(objective, constraints):
    tbl, variables = to_tableau(objective, constraints)
    final_tableau = None
    while True:
        tbl, finished = pivot(tbl)
        if finished:
            final_tableau = tbl
            break
    assert final_tableau is not None
    A = final_tableau[:-1, :-1]
    basic_cols = np.where(
        ((A == 1).sum(axis=0) == 1) &  # Sum is 1
        ((A == 0).sum(axis=0) == A.shape[0] - 1)  # All other rows are 0's
    )[0]
    basic_rows = np.argmax(A[:, basic_cols] == 1, axis=0)
    basic_values = final_tableau[basic_rows, -1]
    solution = {
        variables[col]: value
        for col, value in zip(basic_cols, basic_values)
    }
    return solution

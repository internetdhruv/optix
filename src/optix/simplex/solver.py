import numpy as np

from optix.simplex.tableau import to_tableau


def pivot(tbl, basis, debug=False):
    def log(v):
        if debug:
            print(v)

    # Pick var to make basic
    # In Danzig's - most non negative
    # In Bland's smallest index w/ negative reduced cost
    negative_cols = np.where(tbl[-1, :-1] < 0)[0]
    if len(negative_cols) == 0:
        log("No more pivots possible")
        return tbl, True
    col_pivot = negative_cols[0]
    log(f"Pivoting on index {col_pivot} ({tbl[-1, :-1][col_pivot]}) as its the lowest index negative reduced cost")

    with np.errstate(divide="ignore", invalid="ignore"):
        pivot_candidates = (
                tbl[:-1, -1] /
                tbl[:-1, col_pivot]
        )
    log(pivot_candidates)
    valid = tbl[:-1, col_pivot] > 0
    masked_pivots = np.where(valid, pivot_candidates, np.inf)
    if np.all(np.isinf(masked_pivots)):
        raise Exception("Problem is unbounded")

    # Pick var to make non basic
    # of everything that is lowest ratio, pick the row where current
    # basic var has the lowest index
    min_ratio = masked_pivots.min()
    tied_rows = np.where(masked_pivots <= min_ratio + 1e-9)[0]
    row_pivot = min(tied_rows, key=lambda row: basis[row])
    
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


def solve(objective, constraints, debug=False):
    tbl, variables = to_tableau(objective, constraints)
    n_vars, n_constraints = tbl.shape[1] - 1, tbl.shape[0] - 1

    # Basis - list containing entry for which var is currently basic
    # Seeded with index of slack var per row, as that is our initial basic vars per row
    basis = list(range(n_vars - n_constraints, n_vars))
    
    final_tableau = None
    while True:
        tbl, finished = pivot(tbl, basis, debug)
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

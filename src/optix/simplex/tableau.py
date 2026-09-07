import numpy as np

from optix.expressions import Var


def equation_to_row(equation, variables, slack=None):
    terms = equation.lhs.terms
    row = [
        terms.get(var, 0) + (var == slack)
        for var in variables
    ]
    return row + [equation.rhs]


def to_tableau(objective, constraints):
    objective = objective * -1 <= 0

    variables = []

    for equation in constraints + [objective]:
        for var in equation.lhs.terms:
            if var not in variables:
                variables.append(var)

    slacks = [
        Var(f"s{i}")
        for i in range(1, len(constraints) + 1)
    ]

    variables += slacks

    rows = [
        equation_to_row(constraint, variables, slack)
        for constraint, slack in zip(constraints, slacks)
    ]

    rows.append(equation_to_row(objective, variables))

    return np.array(rows, dtype=float), variables

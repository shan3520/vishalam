"""Task families: matrices (lists of lists of numbers)."""
from __future__ import annotations

from .base import Family

M = ["matrix", "grid", "m", "rows", "table", "board"]
M2 = ["other", "b", "second", "rhs"]
N = ["n", "size", "k", "dim"]
V = ["value", "target", "x", "needle"]


def matrix(rng, rows=(0, 4), cols=(1, 4), lo=-9, hi=9) -> list[list[int]]:
    r, c = rng.randint(*rows), rng.randint(*cols)
    return [[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)]


def square(rng, n=(1, 4), lo=-9, hi=9) -> list[list[int]]:
    k = rng.randint(*n)
    return [[rng.randint(lo, hi) for _ in range(k)] for _ in range(k)]


def same_shape(rng):
    a = matrix(rng, (1, 3), (1, 3))
    return a, [[rng.randint(-9, 9) for _ in row] for row in a]


FAMILIES = [
    Family(
        "transpose", "matrices", ["transpose", "flip_matrix", "swap_rows_cols", "transposed"], [M],
        "def {f}({p0}):\n    if not {p0}:\n        return []\n    return [list(col) for col in zip(*{p0})]\n",
        ["a function `{sig}` that returns the transpose of the matrix `{p0}` (rows become columns)",
         "a function `{f}` which swaps the rows and columns of the 2-D list `{p0}`",
         "`{sig}` returning a new list of lists where element [i][j] is `{p0}`[j][i]",
         "a function named `{f}` that transposes the rectangular grid `{p0}`"],
        lambda r, v: (matrix(r),),
        notes=[["An empty matrix gives []. Rows are lists.", "Return a list of lists; [] for an empty input."]],
        edge=[([],), ([[1, 2, 3]],)],
    ),
    Family(
        "row_sums", "matrices", ["row_sums", "sum_rows", "totals_per_row", "row_totals"], [M],
        "def {f}({p0}):\n    return [sum(row) for row in {p0}]\n",
        ["a function `{sig}` that returns a list with the sum of each row of `{p0}`",
         "a function `{f}` computing the total of every row of the matrix `{p0}`",
         "`{sig}` returning one sum per row of the 2-D list `{p0}`",
         "a function called `{f}` that adds up each row of `{p0}` separately"],
        lambda r, v: (matrix(r),),
        edge=[([],), ([[]],)],
    ),
    Family(
        "col_sums", "matrices", ["col_sums", "column_sums", "sum_columns", "column_totals"], [M],
        "def {f}({p0}):\n    if not {p0}:\n        return []\n    return [sum(col) for col in zip(*{p0})]\n",
        ["a function `{sig}` that returns a list with the sum of each column of `{p0}`",
         "a function `{f}` computing column totals for the rectangular matrix `{p0}`",
         "`{sig}` which adds up every column of the 2-D list `{p0}`",
         "a function named `{f}` returning the per-column sums of `{p0}`"],
        lambda r, v: (matrix(r),),
        notes=[["An empty matrix gives [].", "Return [] if `{p0}` has no rows."]],
        edge=[([],), ([[1, 2]],)],
    ),
    Family(
        "diagonal_sum", "matrices", ["{which}_diagonal_sum", "sum_{which}_diagonal", "{which}_diag_total"], [M],
        "def {f}({p0}):\n    n = len({p0})\n    return sum({p0}[i][{idx}] for i in range(n))\n",
        ["a function `{sig}` that returns the sum of the {which} diagonal of the square matrix `{p0}`",
         "a function `{f}` adding up the {which} diagonal of the n x n grid `{p0}`",
         "`{sig}` which sums the {which}-diagonal elements of the square 2-D list `{p0}`",
         "a function called `{f}` returning the total along the {which} diagonal of `{p0}`"],
        lambda r, v: (square(r),),
        notes=[["{diag_note}"], ["An empty matrix gives 0.", "Return 0 for []."]],
        edge=[([],), ([[5]],)],
        variants=[{"which": "main", "idx": "i", "diag_note": "The main diagonal runs from top-left to bottom-right."},
                  {"which": "anti", "idx": "n - 1 - i", "diag_note": "The anti-diagonal runs from top-right to bottom-left."}],
    ),
    Family(
        "rotate_90", "matrices", ["rotate_90", "rotate_clockwise", "turn_right", "rotate_matrix"], [M],
        "def {f}({p0}):\n    return [list(row) for row in zip(*{p0}[::-1])]\n",
        ["a function `{sig}` that returns `{p0}` rotated 90 degrees clockwise",
         "a function `{f}` turning the matrix `{p0}` a quarter turn to the right",
         "`{sig}` which rotates the 2-D list `{p0}` clockwise by 90 degrees and returns the new matrix",
         "a function named `{f}` returning a clockwise-rotated copy of the grid `{p0}`"],
        lambda r, v: (matrix(r, (1, 4)),),
        notes=[["The input may be rectangular. An empty matrix gives [].", "Works for non-square input too; [] stays []."]],
        edge=[([],), ([[1, 2], [3, 4]],)],
    ),
    Family(
        "is_symmetric", "matrices", ["is_symmetric", "symmetric", "check_symmetric", "equals_transpose"], [M],
        "def {f}({p0}):\n    n = len({p0})\n    return all({p0}[i][j] == {p0}[j][i] for i in range(n) for j in range(n))\n",
        ["a function `{sig}` that returns True if the square matrix `{p0}` equals its own transpose",
         "a function `{f}` checking whether `{p0}`[i][j] == `{p0}`[j][i] for every i and j",
         "`{sig}` which tells if the square grid `{p0}` is symmetric",
         "a function called `{f}` that reports whether the n x n matrix `{p0}` is symmetric about its main diagonal"],
        lambda r, v: ((lambda m: [[m[min(i, j)][max(i, j)] for j in range(len(m))] for i in range(len(m))] if r.random() < 0.5 else m)(square(r, lo=0, hi=3)),),
        notes=[["An empty matrix is symmetric.", "Return True for []."]],
        edge=[([],), ([[1]],), ([[1, 2], [3, 1]],)],
    ),
    Family(
        "identity_matrix", "matrices", ["identity_matrix", "identity", "eye", "unit_matrix"], [N],
        "def {f}({p0}):\n    return [[1 if i == j else 0 for j in range({p0})] for i in range({p0})]\n",
        ["a function `{sig}` that returns the `{p0}` x `{p0}` identity matrix as a list of lists",
         "a function `{f}` building an identity matrix of size `{p0}` (1 on the diagonal, 0 elsewhere)",
         "`{sig}` which creates the square identity matrix with `{p0}` rows",
         "a function named `{f}` returning a `{p0}`-by-`{p0}` grid with ones on the main diagonal and zeros elsewhere"],
        lambda r, v: (r.randint(0, 5),),
        notes=[["For 0 return [].", "Size 0 gives an empty list."]],
        edge=[(0,), (1,)],
    ),
    Family(
        "spiral_order", "matrices", ["spiral_order", "spiral", "unroll_spiral", "spiral_walk"], [M],
        "def {f}({p0}):\n    out = []\n    grid = [list(row) for row in {p0}]\n    while grid:\n        out += grid.pop(0)\n        grid = [list(row) for row in zip(*grid)][::-1]\n    return out\n",
        ["a function `{sig}` that returns the elements of `{p0}` in clockwise spiral order, starting at the top-left",
         "a function `{f}` reading the matrix `{p0}` in a spiral: along the top row, down the right side, back along the bottom, and so on",
         "`{sig}` which walks the 2-D list `{p0}` in a clockwise spiral and returns the visited values",
         "a function called `{f}` returning the clockwise spiral traversal of `{p0}`"],
        lambda r, v: (matrix(r, (1, 4), (1, 4)),),
        notes=[["An empty matrix gives [].", "Return [] for []."]],
        edge=[([],), ([[1, 2], [3, 4]],), ([[1], [2], [3]],)],
    ),
    Family(
        "row_maxes", "matrices", ["row_maxes", "max_per_row", "row_max", "largest_in_each_row"], [M],
        "def {f}({p0}):\n    return [max(row) if row else None for row in {p0}]\n",
        ["a function `{sig}` that returns the largest value of each row of `{p0}`",
         "a function `{f}` listing the maximum of every row in the matrix `{p0}`",
         "`{sig}` which finds the biggest element per row of `{p0}`",
         "a function named `{f}` returning one maximum per row of the 2-D list `{p0}`"],
        lambda r, v: (matrix(r, (0, 4), (0, 4)),),
        notes=[["An empty row contributes None.", "Use None for a row with no elements."]],
        edge=[([],), ([[], [3, 1]],)],
    ),
    Family(
        "count_in_grid", "matrices", ["count_in_grid", "count_cells", "grid_count", "occurrences_2d"], [M, V],
        "def {f}({p0}, {p1}):\n    return sum(row.count({p1}) for row in {p0})\n",
        ["a function `{sig}` that counts how many cells of the 2-D list `{p0}` are equal to `{p1}`",
         "a function `{f}` returning the number of times `{p1}` appears anywhere in the matrix `{p0}`",
         "`{sig}` which counts occurrences of `{p1}` in the grid `{p0}`",
         "a function called `{f}` that tells how many entries of `{p0}` equal `{p1}`"],
        lambda r, v: (matrix(r, lo=0, hi=3), r.randint(0, 3)),
        edge=[([], 0), ([[0, 0], [0, 1]], 0)],
    ),
    Family(
        "neighbor_sum", "matrices", ["neighbor_sum", "sum_neighbors", "neighbours_total", "adjacent_sum"], [M, ["row", "r", "i", "y"], ["col", "c", "j", "x"]],
        "def {f}({p0}, {p1}, {p2}):\n    total = 0\n    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):\n        rr, cc = {p1} + dr, {p2} + dc\n        if 0 <= rr < len({p0}) and 0 <= cc < len({p0}[rr]):\n            total += {p0}[rr][cc]\n    return total\n",
        ["a function `{sig}` that returns the sum of the up, down, left and right neighbours of cell (`{p1}`, `{p2}`) in `{p0}`",
         "a function `{f}` adding the values of the 4 orthogonal neighbours of `{p0}`[`{p1}`][`{p2}`]",
         "`{sig}` which sums the cells directly above, below, left and right of the given cell in the grid `{p0}`",
         "a function named `{f}` returning the total of the edge-adjacent neighbours of position (`{p1}`, `{p2}`)"],
        lambda r, v: (lambda m: (m, r.randrange(len(m)), r.randrange(len(m[0]))))(matrix(r, (1, 4), (1, 4))),
        notes=[["Neighbours outside the grid are skipped; the cell itself is not included.",
                "Ignore positions that fall off the edge, and don't count the centre cell."]],
        edge=[([[5]], 0, 0), ([[1, 2], [3, 4]], 0, 0)],
    ),
    Family(
        "matrix_add", "matrices", ["matrix_add", "add_matrices", "sum_matrices", "elementwise_add"], [M, M2],
        "def {f}({p0}, {p1}):\n    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip({p0}, {p1})]\n",
        ["a function `{sig}` that returns the element-wise sum of two matrices `{p0}` and `{p1}` of the same shape",
         "a function `{f}` adding the matrices `{p0}` and `{p1}` cell by cell",
         "`{sig}` which returns a new matrix where each cell is `{p0}`[i][j] + `{p1}`[i][j]",
         "a function called `{f}` that adds two equally-sized 2-D lists"],
        lambda r, v: same_shape(r),
        edge=[([], []), ([[1]], [[-1]])],
    ),
    Family(
        "scale_matrix", "matrices", ["scale_matrix", "multiply_matrix", "scalar_multiply", "times_matrix"], [M, ["factor", "k", "scalar", "c"]],
        "def {f}({p0}, {p1}):\n    return [[{p1} * x for x in row] for row in {p0}]\n",
        ["a function `{sig}` that returns `{p0}` with every element multiplied by `{p1}`",
         "a function `{f}` scaling each cell of the matrix `{p0}` by `{p1}`",
         "`{sig}` which multiplies the whole 2-D list `{p0}` by the number `{p1}`",
         "a function named `{f}` returning a new matrix equal to `{p1}` times `{p0}`"],
        lambda r, v: (matrix(r), r.randint(-3, 4)),
        notes=[["Do not modify `{p0}`.", "Return a new matrix; leave the input unchanged."]],
        edge=[([], 3), ([[1, -2]], 0)],
    ),
    Family(
        "matmul", "matrices", ["matmul", "matrix_multiply", "multiply_matrices", "mat_product"], [M, M2],
        "def {f}({p0}, {p1}):\n    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*{p1})] for row in {p0}]\n",
        ["a function `{sig}` that returns the matrix product of `{p0}` and `{p1}`",
         "a function `{f}` multiplying the matrix `{p0}` (n x m) by `{p1}` (m x p)",
         "`{sig}` which computes `{p0}` times `{p1}` using the usual row-by-column rule",
         "a function called `{f}` returning the product of two compatible 2-D lists"],
        lambda r, v: (lambda n, m, p: ([[r.randint(-3, 3) for _ in range(m)] for _ in range(n)],
                                       [[r.randint(-3, 3) for _ in range(p)] for _ in range(m)]))(r.randint(1, 3), r.randint(1, 3), r.randint(1, 3)),
        notes=[["You can assume the number of columns of `{p0}` equals the number of rows of `{p1}`.",
                "The shapes are always compatible."]],
        edge=[([[2]], [[3]]), ([[1, 2]], [[3], [4]])],
    ),
    Family(
        "find_in_grid", "matrices", ["find_in_grid", "locate", "position_of", "find_cell"], [M, V],
        "def {f}({p0}, {p1}):\n    for i, row in enumerate({p0}):\n        for j, x in enumerate(row):\n            if x == {p1}:\n                return (i, j)\n    return None\n",
        ["a function `{sig}` that returns the (row, column) of the first cell of `{p0}` equal to `{p1}`, scanning row by row",
         "a function `{f}` returning a tuple with the position of `{p1}` in the grid `{p0}`",
         "`{sig}` which searches the 2-D list `{p0}` for `{p1}` and gives its (row, col)",
         "a function named `{f}` that locates `{p1}` in the matrix `{p0}`"],
        lambda r, v: (matrix(r, lo=0, hi=5), r.randint(0, 6)),
        notes=[["Return None if it isn't there.", "If `{p1}` doesn't occur, return None."]],
        edge=[([], 1), ([[1, 2], [2, 1]], 2)],
    ),
    Family(
        "border_sum", "matrices", ["border_sum", "edge_sum", "perimeter_sum", "frame_total"], [M],
        "def {f}({p0}):\n    total = 0\n    for i, row in enumerate({p0}):\n        for j, x in enumerate(row):\n            if i == 0 or i == len({p0}) - 1 or j == 0 or j == len(row) - 1:\n                total += x\n    return total\n",
        ["a function `{sig}` that returns the sum of the cells on the outer border of the matrix `{p0}`",
         "a function `{f}` adding up every element in the first/last row and first/last column of `{p0}`",
         "`{sig}` which totals the frame (edge cells) of the grid `{p0}`, counting each cell once",
         "a function called `{f}` returning the sum along the perimeter of `{p0}`"],
        lambda r, v: (matrix(r, (0, 4), (1, 4)),),
        notes=[["An empty matrix gives 0.", "Return 0 for []."]],
        edge=[([],), ([[7]],), ([[1, 1, 1], [1, 9, 1], [1, 1, 1]],)],
    ),
    Family(
        "flatten_grid", "matrices", ["flatten_grid", "grid_to_list", "unroll_rows", "flatten_matrix"], [M],
        "def {f}({p0}):\n    return [x for row in {p0} for x in row]\n",
        ["a function `{sig}` that returns all elements of the matrix `{p0}` in one list, row by row",
         "a function `{f}` flattening the 2-D list `{p0}` in row-major order",
         "`{sig}` which turns the grid `{p0}` into a flat list, top row first",
         "a function named `{f}` that concatenates the rows of `{p0}`"],
        lambda r, v: (matrix(r, (0, 4), (0, 4)),),
        edge=[([],), ([[], [1]],)],
    ),
    Family(
        "is_upper_triangular", "matrices", ["is_upper_triangular", "upper_triangular", "check_upper", "zeros_below_diagonal"], [M],
        "def {f}({p0}):\n    return all({p0}[i][j] == 0 for i in range(len({p0})) for j in range(i))\n",
        ["a function `{sig}` that returns True if every element below the main diagonal of the square matrix `{p0}` is 0",
         "a function `{f}` checking whether the square grid `{p0}` is upper triangular",
         "`{sig}` which tells if `{p0}` has only zeros under its main diagonal",
         "a function called `{f}` returning whether the n x n matrix `{p0}` is upper-triangular"],
        lambda r, v: ((lambda m: [[x if j >= i or r.random() < 0.15 else 0 for j, x in enumerate(row)] for i, row in enumerate(m)])(square(r)),),
        notes=[["An empty matrix counts as upper triangular.", "Return True for []."]],
        edge=[([],), ([[1, 2], [0, 3]],), ([[1, 2], [4, 3]],)],
    ),
]

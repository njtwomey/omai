// The two-variable subproblem of libsvm's solvers (SMO and its nu variant), shared by `smo.ts` and `smo-nu.ts`: the
// closed-form step on a working pair, clipped to the box and the equality. Not exported from the barrel.

const TAU = 1e-12

/**
 * Optimise the pair (i, j) of `a` in place: the closed-form minimiser of the dual along the direction that keeps the
 * equality `y^T a` fixed, clipped to the boxes `[0, ci]` and `[0, cj]`.
 */
export function updatePair(
  a: Float64Array,
  i: number,
  j: number,
  signs: { readonly yi: number; readonly yj: number },
  grads: { readonly gi: number; readonly gj: number },
  curvature: { readonly qij: number; readonly qdi: number; readonly qdj: number },
  bounds: { readonly ci: number; readonly cj: number },
): void {
  const { yi, yj } = signs
  const { gi, gj } = grads
  const { qij, qdi, qdj } = curvature
  const { ci, cj } = bounds
  const oldI = a[i] ?? 0
  const oldJ = a[j] ?? 0
  if (yi !== yj) {
    let quad = qdi + qdj + 2 * qij
    if (quad <= 0) quad = TAU
    const delta = (-gi - gj) / quad
    const diff = oldI - oldJ
    a[i] = oldI + delta
    a[j] = oldJ + delta
    if (diff > 0) {
      if ((a[j] ?? 0) < 0) {
        a[j] = 0
        a[i] = diff
      }
    } else if ((a[i] ?? 0) < 0) {
      a[i] = 0
      a[j] = -diff
    }
    if (diff > ci - cj) {
      if ((a[i] ?? 0) > ci) {
        a[i] = ci
        a[j] = ci - diff
      }
    } else if ((a[j] ?? 0) > cj) {
      a[j] = cj
      a[i] = cj + diff
    }
  } else {
    let quad = qdi + qdj - 2 * qij
    if (quad <= 0) quad = TAU
    const delta = (gi - gj) / quad
    const sum = oldI + oldJ
    a[i] = oldI - delta
    a[j] = oldJ + delta
    if (sum > ci) {
      if ((a[i] ?? 0) > ci) {
        a[i] = ci
        a[j] = sum - ci
      }
    } else if ((a[j] ?? 0) < 0) {
      a[j] = 0
      a[i] = sum
    }
    if (sum > cj) {
      if ((a[j] ?? 0) > cj) {
        a[j] = cj
        a[i] = sum - cj
      }
    } else if ((a[i] ?? 0) < 0) {
      a[i] = 0
      a[j] = sum
    }
  }
}

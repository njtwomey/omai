/**
 * The six SVM duals are set up once (`optim/_smo-problems.ts`) and used by two entry points: the library's run-to-the-end
 * functions of `optim` and the steppable `Dual`s of `svm`. This table pins both on one small problem per dual, with the
 * linear and the RBF kernel, so that a change to either entry point, or to the shared set-up, shows as a drift.
 */

import { describe, expect, it } from 'vitest'
import * as O from '../optim/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as D from './_dual.ts'

const X = [
  [-2, -1],
  [-1.5, 0.5],
  [-1, -1.2],
  [-0.3, 0.2],
  [0.4, -0.1],
  [1, 1.3],
  [1.6, -0.5],
  [2, 0.8],
]
const n = X.length
const gram = (k: (a: number[], b: number[]) => number): Float64Array =>
  Float64Array.from({ length: n * n }, (_, i) => k(X[Math.floor(i / n)] as number[], X[i % n] as number[]))
const kernels = {
  linear: gram((a, b) => (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0)),
  rbf: gram((a, b) => Math.exp(-0.5 * ((a[0] ?? 0) - (b[0] ?? 0)) ** 2 - 0.5 * ((a[1] ?? 0) - (b[1] ?? 0)) ** 2)),
}
const rowsOf = (K: Float64Array): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from(K.subarray(i * n, (i + 1) * n)))
const y = Float64Array.of(-1, -1, -1, 1, -1, 1, 1, 1)
const t = Float64Array.of(-1.1, -0.4, -0.9, 0.1, 0.3, 0.8, 0.5, 1.2)
const C = new Float64Array(n).fill(2)
const tol = 1e-6
const stepped = { tol, selection: 'second-order' } as const
const fixed = { tol, selection: 'second-order', shrinking: false } as const

const near = (actual: number, expected: number): void => {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-8 * Math.max(1, Math.abs(expected)))
}

// Per dual and kernel: the number of SMO steps, rho, the expansion coefficients, `optim`'s objective and `svm`'s dual.
const rows = [
  {
    kernel: 'linear',
    dual: 'svc',
    iterations: 19,
    rho: 0.0361445448,
    coef: [0, -0.724052837, 0, 2, -2, 0.202786993, 0.521265845, 0],
    optimObjective: -5.15778778,
    svmDual: 5.15778778,
  },
  {
    kernel: 'linear',
    dual: 'svr',
    iterations: 221,
    rho: -0.0629310345,
    coef: [0, 0, -2, 1.89655172, 1.85344828, -1.75, 0, 0],
    optimObjective: -0.255603448,
    svmDual: 0.255603448,
  },
  {
    kernel: 'linear',
    dual: 'oneclass',
    iterations: 14,
    rho: -6.89911219e-9,
    coef: [0, 0.82769229, 0.987076974, 0.2, 0, 0.0129230255, 0.17230771, 1],
    optimObjective: 1.24998531e-14,
    svmDual: -1.24998531e-14,
  },
  {
    kernel: 'linear',
    dual: 'svdd',
    iterations: 1,
    rho: -3.875,
    coef: [0.5, 0, 0, 0, 0, 0, 0, 0.5],
    optimObjective: -4.81,
    svmDual: 4.81,
  },
  {
    kernel: 'linear',
    dual: 'nusvc',
    iterations: 2,
    rho: 0.0361445783,
    coef: [0, -0.454648784, 0, 0.757747973, -0.757747973, 0.260160137, 0.194488646, 0],
    optimObjective: 0.290317898,
    svmDual: 1.84415772,
  },
  {
    kernel: 'linear',
    dual: 'nusvr',
    iterations: 570,
    rho: -0.0817478287,
    coef: [-0.080429139, 0, -2, 2, 2, -1.855183, -0.0643878651, 0],
    optimObjective: -1.04580038,
    svmDual: 1.04580038,
  },
  {
    kernel: 'rbf',
    dual: 'svc',
    iterations: 28,
    rho: -0.00689726092,
    coef: [-0.304924116, -1.45509487, -0.743987612, 2, -2, 0.841430243, 1.43073868, 0.23183768],
    optimObjective: -6.61824291,
    svmDual: 6.61824291,
  },
  {
    kernel: 'rbf',
    dual: 'svr',
    iterations: 26,
    rho: -0.0234787566,
    coef: [-0.790199946, -0.0857379868, -0.376795543, 0.131977031, 0, 0.110307759, 0, 1.01044869],
    optimObjective: -1.15303347,
    svmDual: 1.15303347,
  },
  {
    kernel: 'rbf',
    dual: 'oneclass',
    iterations: 49,
    rho: 0.904261483,
    coef: [0.467605884, 0.572030167, 0.41842781, 0.165249347, 0.10148392, 0.522462318, 0.594247706, 0.358492848],
    optimObjective: 1.44681848,
    svmDual: -1.44681848,
  },
  {
    kernel: 'rbf',
    dual: 'svdd',
    iterations: 48,
    rho: -0.43483652,
    coef: [0.146126451, 0.178759721, 0.130759125, 0.0516391421, 0.0317151431, 0.163268854, 0.185701959, 0.112029604],
    optimObjective: -0.717418265,
    svmDual: 0.717418265,
  },
  {
    kernel: 'rbf',
    dual: 'nusvc',
    iterations: 22,
    rho: -0.0189688785,
    coef: [-0.0361101465, -3.16989922, -1.0583061, 7.10719244, -7.10719244, 1.1005819, 3.09396342, 0.0697701465],
    optimObjective: 9.07716317,
    svmDual: 4.58868947,
  },
  {
    kernel: 'rbf',
    dual: 'nusvr',
    iterations: 203,
    rho: -0.0226534339,
    coef: [-0.787582997, -0.303521588, -0.501048036, 0.49456741, -0.193655852, 0.145512802, 0.0633811657, 1.08234709],
    optimObjective: -1.43848561,
    svmDual: 1.43848561,
  },
] as const

type Dual = ReturnType<typeof D.cSvcDual>
type Optim = { iterations: number; rho: number; coef: { data: ArrayLike<number> }; objective: number }

function entries(dual: string, K: Float64Array): { svm: Dual; optim: Optim } {
  const R = rowsOf(K)
  switch (dual) {
    case 'svc':
      return { svm: D.cSvcDual(K, y, C, stepped), optim: O.svcDual(R, y, C, fixed) }
    case 'svr':
      return { svm: D.svrDual(K, t, C, 0.1, stepped), optim: O.svrDual(R, t, 2, 0.1, fixed) }
    case 'oneclass':
      return { svm: D.oneClassDual(K, 0.4, stepped), optim: O.oneClassDual(R, 0.4, fixed) }
    case 'svdd':
      return { svm: D.svddDual(K, 0.5, stepped), optim: O.svddDual(R, 0.5, fixed) }
    case 'nusvc':
      return { svm: D.nuSvcDual(K, y, 0.4, stepped), optim: O.nuSvcDual(R, y, 0.4, { tol }) }
    default:
      return { svm: D.nuSvrDual(K, t, 2, 0.5, stepped), optim: O.nuSvrDual(R, t, 2, 0.5, { tol }) }
  }
}

describe('the SVM duals through both entry points', () => {
  it('give the same solution, step count and objective on small problems (classification, regression, one-class, SVDD)', () => {
    for (const row of rows) {
      const { svm, optim } = entries(row.dual, kernels[row.kernel])
      const run = runAlgorithm(svm.algorithm, { maxSteps: 100_000 })
      const e = svm.expansion(run.state)
      const label = `${row.kernel} ${row.dual}`
      expect(run.state.iterations, label).toBe(row.iterations)
      expect(optim.iterations, label).toBe(row.iterations)
      near(e.rho, row.rho)
      near(optim.rho, row.rho)
      near(optim.objective, row.optimObjective)
      near(svm.dual(run.state), row.svmDual)
      row.coef.forEach((c, i) => {
        near(e.coef[i] ?? Number.NaN, c)
        near(Array.from(optim.coef.data)[i] ?? Number.NaN, c)
      })
    }
  })
})

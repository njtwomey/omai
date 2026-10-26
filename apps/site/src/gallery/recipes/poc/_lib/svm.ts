// A sklearn-iterative method on the contract: an RBF support vector classifier solved by the core's SMO, whose native
// channels (objective, violation, alpha, working pair) are renamed to contract paths with `from`, so the algorithm
// is untouched; the margin (support vectors, dual coefficients, the pair being optimised, the decision function) is
// derived from any state as the standard `margin` record, which the margin view draws.
import { decisionSurface, marginRecord } from 'mdml-engine/exposition'
import { type SmoState, smoAlgorithm } from 'mdml-engine/optim'
import { foldIn, type Key } from 'mdml-engine/random'
import { choice, dict, int, real } from 'mdml-engine/space'
import { defineMethod, derived, Log, type Method, message, scalar, tensor } from 'mdml-engine/trace'
import { DATASETS, type DatasetName, gridOver, rowsOf, twoClassData } from './data.ts'

export type SvmConfig = {
  readonly dataset: DatasetName
  readonly n: number
  readonly C: number
  readonly gamma: number
}

function built(
  c: SvmConfig,
  key: Key,
): {
  data: ReturnType<typeof twoClassData>
  y: number[]
  kernel: (a: readonly number[], b: readonly number[]) => number
} {
  const data = twoClassData(c.dataset, c.n, foldIn(key, 0))
  const y = data.y.map((v) => (v === 1 ? 1 : -1))
  const kernel = (a: readonly number[], b: readonly number[]): number => {
    const dx = (a[0] ?? 0) - (b[0] ?? 0)
    const dy = (a[1] ?? 0) - (b[1] ?? 0)
    return Math.exp(-c.gamma * (dx * dx + dy * dy))
  }
  return { data, y, kernel }
}

export const svmMethod: Method<SmoState, SvmConfig> = defineMethod<SmoState, SvmConfig>({
  name: 'poc/svm',
  config: dict({
    dataset: choice([...DATASETS], { default: 'moons' }),
    n: int(20, 400, { default: 120 }),
    C: real(0.01, 100, { log: true, default: 1 }),
    gamma: real(0.05, 20, { log: true, default: 1 }),
  }),
  make: (c, key) => {
    const { data, y, kernel } = built(c, key)
    const Q = data.x.map((xi, i) => data.x.map((xj, j) => (y[i] ?? 0) * (y[j] ?? 0) * kernel(xi, xj)))
    return smoAlgorithm({ Q, p: y.map(() => -1), y, C: c.C }, { tol: 1e-3, shrinking: false })
  },
  contract: (c, key) => {
    const { data, y, kernel } = built(c, key)
    const support = (s: SmoState): number[] => Array.from(s.alpha, (a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0)
    const decision = (s: SmoState, sv: readonly number[], b: number, x: readonly number[]): number => {
      let f = 0
      for (const i of sv) f += (s.alpha[i] ?? 0) * (y[i] ?? 0) * kernel(data.x[i] ?? [0, 0], x)
      return f + b
    }
    return {
      'smo/dual-objective': scalar({ from: 'objective', description: 'the dual objective', direction: 'lower' }),
      'smo/violation': scalar({ from: 'violation', description: 'the largest KKT violation' }),
      'smo/support': scalar({ from: 'support', description: 'variables with alpha > 0' }),
      'smo/alpha': tensor({ from: 'alpha', role: 'param', shape: ['n'], description: 'the dual variables' }),
      'smo/working-set': message({ from: 'workingSet', description: 'the pair optimised this step' }),
      'smo/support-vectors': derived({ description: 'indices of the support vectors', compute: support }),
      'smo/margin': derived({
        description: 'the margin at this step: support vectors, the pair being optimised, the decision function',
        cost: 'dear',
        compute: (s, args) => {
          const res = ((args as { readonly resolution?: number } | undefined)?.resolution ?? 40) | 0
          const sv = support(s)
          const b = -rho(s, y, c.C)
          const grid = gridOver(data.bounds, res)
          const f = grid.points.map((p) => decision(s, sv, b, p))
          let objective = 0
          for (let i = 0; i < y.length; i++) objective += 0.5 * (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) - 1)
          const rows = { points: data.x, features: ['x', 'y'] as const, labels: data.y }
          return marginRecord({
            support: sv,
            dualCoefficients: sv.map((i) => (s.alpha[i] ?? 0) * (y[i] ?? 0)),
            intercept: b,
            margins: data.x.map((x, i) => (y[i] ?? 0) * decision(s, sv, b, x)),
            ...(s.pair === undefined ? {} : { workingSets: [[s.pair[0], s.pair[1]]], dualObjective: [objective] }),
            rows,
            surface: decisionSurface({
              features: ['x', 'y'],
              x: grid.xs,
              y: grid.ys,
              decision: rowsOf(f, res),
              labels: rowsOf(
                f.map((v) => (v > 0 ? 1 : 0)),
                res,
              ),
              levels: [-1, 1],
              rows,
            }),
          })
        },
      }),
    }
  },
  defaults: { 'smo/dual-objective': Log(), 'smo/support': Log() },
})

/** libsvm's rho: the mean of y_i G_i over free variables, else over the support vectors. */
function rho(s: SmoState, y: readonly number[], C: number): number {
  let sum = 0
  let count = 0
  for (let i = 0; i < y.length; i++) {
    const a = s.alpha[i] ?? 0
    if (a > 0 && a < C) {
      sum += (y[i] ?? 0) * (s.grad[i] ?? 0)
      count++
    }
  }
  if (count === 0) {
    for (let i = 0; i < y.length; i++) {
      if ((s.alpha[i] ?? 0) > 0) {
        sum += (y[i] ?? 0) * (s.grad[i] ?? 0)
        count++
      }
    }
  }
  return count === 0 ? 0 : sum / count
}

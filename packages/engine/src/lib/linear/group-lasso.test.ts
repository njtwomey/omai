import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { designOf } from './design.ts'
import {
  groupLasso,
  groupLassoAlgorithm,
  groupLassoAlphaMax,
  groupStructure,
  multiTaskDesign,
  multiTaskLasso,
  multiTaskLassoAlgorithm,
} from './group-lasso.ts'
import { linearModel } from './model.ts'

// Six features in three groups of two; only the first group matters, the third is noise.
const n = 40
const x = Array.from({ length: n }, (_, i) => [
  Math.sin(i * 1.3),
  Math.cos(i * 0.7),
  Math.sin(i * 2.9 + 1),
  Math.cos(i * 1.9),
  Math.sin(i * 0.37 + 2),
  Math.cos(i * 3.3),
])
const y = x.map((r, i) => 2 * (r[0] ?? 0) - 1.5 * (r[1] ?? 0) + 0.6 * (r[2] ?? 0) + 0.3 * Math.sin(i * 5.1) + 0.5)

/** The optimality conditions of the group lasso, recomputed from the data. */
function kktViolation(
  coef: readonly number[],
  intercept: number,
  groups: readonly number[],
  alpha: number,
  ridge: number,
): number {
  const s = groupStructure(6, groups)
  const r = x.map((row, i) => (y[i] ?? 0) - intercept - row.reduce((a, v, j) => a + v * (coef[j] ?? 0), 0))
  let worst = 0
  s.members.forEach((m, g) => {
    const grad = m.map((j) => x.reduce((a, row, i) => a + (row[j] ?? 0) * (r[i] ?? 0), 0) / n - ridge * (coef[j] ?? 0))
    const bn = Math.sqrt(m.reduce((a, j) => a + (coef[j] ?? 0) ** 2, 0))
    const w = alpha * (s.weights[g] ?? 0)
    if (bn > 0) worst = Math.max(worst, Math.hypot(...grad.map((v, k) => v - (w * (coef[m[k] ?? 0] ?? 0)) / bn)))
    else worst = Math.max(worst, Math.max(0, Math.hypot(...grad) - w))
  })
  return worst
}

describe('group lasso', () => {
  const groups = [0, 0, 1, 1, 2, 2]

  it('satisfies the optimality conditions and zeroes whole groups', () => {
    const aMax = groupLassoAlphaMax({ x, y }, { groups })
    const f = groupLasso({ groups, alpha: 0.2 * aMax, tol: 1e-10 }).fit({ x, y })
    expect(f.params.converged).toBe(true)
    expect(kktViolation(f.params.coef, f.params.intercept, groups, 0.2 * aMax, 0)).toBeLessThan(1e-8)
    const c = f.params.coef
    for (const [a, b] of [
      [0, 1],
      [2, 3],
      [4, 5],
    ] as const)
      expect((c[a] === 0) === (c[b] === 0)).toBe(true)
    expect(f.params.activeGroups.length).toBeGreaterThan(0)
    expect(f.params.activeGroups.length).toBeLessThan(3)
    expect(f.params.activeGroups).toContain(0)
  })

  it('has everything zero above alpha max and something non-zero below', () => {
    const aMax = groupLassoAlphaMax({ x, y }, { groups })
    expect(
      groupLasso({ groups, alpha: 1.01 * aMax })
        .fit({ x, y })
        .params.coef.every((v) => v === 0),
    ).toBe(true)
    expect(groupLasso({ groups, alpha: 0.9 * aMax }).fit({ x, y }).params.activeGroups.length).toBe(1)
  })

  it('is the lasso when every feature is its own group', () => {
    const f = groupLasso({ alpha: 0.05, tol: 1e-11 }).fit({ x, y })
    const lasso = linearModel({
      alpha: 0.05,
      l1Ratio: 1,
      tol: 1e-12,
      maxIter: 100000,
      solver: 'coordinate-descent',
    }).fit({ x, y })
    f.params.coef.forEach((v, j) => {
      expect(v).toBeCloseTo(lasso.params.coef[j] ?? 0, 6)
    })
    expect(f.params.intercept).toBeCloseTo(lasso.params.intercept, 6)
  })

  it('reaches the same solution with block steps and with FISTA, with a ridge term', () => {
    const o = { groups, alpha: 0.05, ridge: 0.1, tol: 1e-10, maxIter: 20000 }
    const a = groupLasso({ ...o, solver: 'bcd' }).fit({ x, y }).params
    const b = groupLasso({ ...o, solver: 'fista' }).fit({ x, y }).params
    a.coef.forEach((v, j) => {
      expect(v).toBeCloseTo(b.coef[j] ?? 0, 6)
    })
    expect(a.objective).toBeCloseTo(b.objective, 10)
    expect(kktViolation(a.coef, a.intercept, groups, 0.05, 0.1)).toBeLessThan(1e-8)
  })

  it('treats integer sample weights as repeated rows', () => {
    const w = x.map((_, i) => (i % 3 === 0 ? 2 : 1))
    const rep = x.flatMap((r, i) => (w[i] === 2 ? [r, r] : [r]))
    const yr = y.flatMap((v, i) => (w[i] === 2 ? [v, v] : [v]))
    const a = groupLasso({ groups, alpha: 0.1, tol: 1e-11 }).fit({ x, y, weights: w })
    const b = groupLasso({ groups, alpha: 0.1, tol: 1e-11 }).fit({ x: rep, y: yr })
    a.params.coef.forEach((v, j) => {
      expect(v).toBeCloseTo(b.params.coef[j] ?? 0, 7)
    })
  })

  it('resumes from a serialised state', () => {
    const design = designOf({ x, y }, 'test')
    const alg = groupLassoAlgorithm(design, { groups, alpha: 0.05 })
    const direct = runAlgorithm(alg, { maxSteps: 12, key: key(0) })
    const half = runAlgorithm(alg, { maxSteps: 5, key: key(0) })
    const rest = runAlgorithm(alg, {
      maxSteps: 7,
      key: key(0),
      start: { state: alg.serialise?.(half.state), step: half.steps, serialised: true },
    })
    expect(rest.state.coef).toEqual(direct.state.coef)
  })

  it('predicts and scores', () => {
    const f = groupLasso({ groups, alpha: 0.01 }).fit({ x, y })
    expect(toFlat(f.predict(x))).toHaveLength(n)
    expect(f.score({ x, y })).toBeGreaterThan(0.9)
  })
})

describe('multi-task lasso', () => {
  const tasks = x.map((r, i) => [
    2 * (r[0] ?? 0) + 0.2 * Math.sin(i * 4.1) + 1,
    -1 * (r[0] ?? 0) + 3 * (r[2] ?? 0) + 0.2 * Math.cos(i * 2.3),
    0.5 * (r[2] ?? 0) + 0.1 * Math.sin(i * 7.7) - 2,
  ])

  it('shares the support across tasks and satisfies the optimality conditions', () => {
    const f = multiTaskLasso({ alpha: 0.15, tol: 1e-11 }).fit({ x, y: tasks })
    expect(f.params.converged).toBe(true)
    const { coef, support } = f.params
    for (let j = 0; j < 6; j++) {
      const zeros = coef.map((row) => row[j] === 0)
      expect(zeros.every(Boolean) || zeros.every((z) => !z)).toBe(true)
    }
    expect(support).toEqual(expect.arrayContaining([0, 2]))
    expect(support.length).toBeLessThan(6)
    expect(toFlat(f.predict(x))).toHaveLength(n * 3)
  })

  it('is the lasso for one task', () => {
    const one = tasks.map((r) => [r[1] ?? 0])
    const m = multiTaskLasso({ alpha: 0.05, tol: 1e-12 }).fit({ x, y: one }).params
    const lasso = linearModel({
      alpha: 0.05,
      l1Ratio: 1,
      tol: 1e-12,
      maxIter: 100000,
      solver: 'coordinate-descent',
    }).fit({ x, y: one.map((r) => r[0] ?? 0) })
    m.coef[0]?.forEach((v, j) => {
      expect(v).toBeCloseTo(lasso.params.coef[j] ?? 0, 6)
    })
    expect(m.intercept[0]).toBeCloseTo(lasso.params.intercept, 6)
  })

  it('is all zero at a large penalty and logs a decreasing objective', () => {
    const problem = multiTaskDesign({ x, y: tasks }, 'test')
    expect(
      runAlgorithm(multiTaskLassoAlgorithm(problem, { alpha: 100 }), { maxSteps: 5 }).state.coef.every((v) => v === 0),
    ).toBe(true)
    const objectives: number[] = []
    const alg = multiTaskLassoAlgorithm(problem, { alpha: 0.05 })
    let s = alg.init(key(0))
    for (let t = 0; t < 8; t++) {
      s = alg.step(s, { t, key: key(0) } as never)
      objectives.push(s.objective)
    }
    expect(objectives.every((v, i) => i === 0 || v <= (objectives[i - 1] ?? 0) + 1e-12)).toBe(true)
  })
})

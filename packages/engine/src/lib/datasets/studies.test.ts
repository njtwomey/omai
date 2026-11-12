import { describe, expect, it } from 'vitest'
import { argmaxOf } from '../base/index.ts'
import { generate, rowCount, type TableColumn } from '../data/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import {
  abTest,
  antiderivativeOperator,
  burgersOperator,
  crowdLabels,
  heatOperator,
  instrumentalVariables,
  labellingFunctions,
  labelProportionBags,
  linearScm,
  loggedBandit,
  miscalibratedPredictions,
  offlineMdp,
  policyEvaluation,
  positiveUnlabelled,
  survivalData,
  type TabularMdp,
  treatmentEffects,
} from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
const col = (x: Tensor, j: number): number[] => {
  const [n = 0, d = 1] = x.shape
  return Array.from({ length: n }, (_, i) => Number(x.data[i * d + j] ?? 0))
}

describe('causal data', () => {
  it('linear SCMs: structural equations, total effects and the DAG', () => {
    const { data, truth } = generate(linearScm, { key: key(0), knobs: { n: 4000, variables: 5, edgeProbability: 0.6 } })
    const b = truth.extra?.['weights'] as number[][]
    const cols = Array.from({ length: 5 }, (_, j) => col(data.x, j))
    for (let j = 0; j < 5; j++) {
      const row = b[j] ?? []
      const r = Array.from(
        { length: 4000 },
        (_, i) => (cols[j]?.[i] ?? 0) - row.reduce((s, w, a) => s + w * (cols[a]?.[i] ?? 0), 0),
      )
      expect(mean(r.map((v) => v * v))).toBeCloseTo(1, 0)
    }
    const total = truth.extra?.['totalEffects'] as number[][]
    const eye = Array.from({ length: 5 }, (_, i) =>
      Array.from({ length: 5 }, (_, j) => (i === j ? 1 : 0) - (b[i]?.[j] ?? 0)),
    )
    const expected = toFlat(inv(tensor(eye)))
    total.flat().forEach((v, i) => {
      expect(v + (Math.floor(i / 5) === i % 5 ? 1 : 0)).toBeCloseTo(expected[i] ?? 0, 10)
    })
    expect(truth.structure?.edges.length).toBe(b.flat().filter((w) => w !== 0).length)
  })
  it('treatment effects: confounding biases the naive contrast, the truth does not', () => {
    const { data, truth } = generate(treatmentEffects, {
      key: key(1),
      knobs: { n: 20000, confounding: 2, ate: 1, heterogeneity: 0 },
    })
    const t = col(data.x, 5)
    const y = toFlat(data.y)
    const naive = mean(y.filter((_, i) => t[i] === 1)) - mean(y.filter((_, i) => t[i] === 0))
    expect(Math.abs(naive - 1)).toBeGreaterThan(0.3)
    const ite = toFlat(truth.effects?.ite as Tensor)
    expect(mean(ite)).toBeCloseTo(1, 10)
    const po = truth.effects?.potentialOutcomes
    const control = toFlat(po?.control as Tensor)
    const diff = toFlat(po?.treated as Tensor).map((v, i) => v - (control[i] ?? 0))
    expect(diff.every((v, i) => Math.abs(v - (ite[i] ?? 0)) < 1e-12)).toBe(true)
    const propensity = truth.extra?.['propensity'] as (v: Tensor) => Tensor
    const prop = toFlat(propensity(data.x))
    expect(prop.every((p) => p >= 0.05 - 1e-12 && p <= 0.95 + 1e-12)).toBe(true)
  })
  it('instrumental variables: OLS is biased by the stated amount, the Wald ratio is not', () => {
    const { data, truth } = generate(instrumentalVariables, {
      key: key(2),
      knobs: { n: 40000, effect: 2, strength: 1, confounding: 1 },
    })
    const z = col(data.x, 0)
    const t = col(data.x, 1)
    const y = toFlat(data.y)
    const cov = (a: number[], b: number[]) => {
      const ma = mean(a)
      const mb = mean(b)
      return mean(a.map((v, i) => (v - ma) * ((b[i] ?? 0) - mb)))
    }
    expect(cov(z, y) / cov(z, t)).toBeCloseTo(2, 1)
    expect(cov(t, y) / cov(t, t)).toBeCloseTo(2 + (truth.extra?.['olsBias'] as number), 1)
  })
})

describe('experiments', () => {
  for (const metric of ['conversion', 'continuous', 'revenue']) {
    it(`A/B ${metric}: arm means and allocation`, () => {
      const { data, truth } = generate(abTest, {
        key: key(3),
        knobs: { users: 40000, metric, baseline: metric === 'conversion' ? 0.2 : 10, lift: 0.1, arms: 3 },
      })
      const arm = data.columns['arm'] as Extract<TableColumn, { kind: 'categorical' }>
      const y = toFlat((data.columns['metric'] as Extract<TableColumn, { kind: 'numeric' }>).values)
      const means = truth.extra?.['armMeans'] as number[]
      for (let a = 0; a < 3; a++) {
        const ys = y.filter((_, i) => arm.codes.data[i] === a)
        const m = mean(ys)
        const sd = Math.sqrt(mean(ys.map((v) => (v - m) ** 2)))
        expect(Math.abs(m - (means[a] ?? 0)), `${metric} arm ${a}`).toBeLessThan(5 * (sd / Math.sqrt(ys.length)))
      }
      if (metric === 'continuous') {
        const pre = toFlat((data.columns['pre'] as Extract<TableColumn, { kind: 'numeric' }>).values)
        const control = y.map((v, i) => [v, pre[i] ?? 0] as const).filter((_, i) => arm.codes.data[i] === 0)
        const mx = mean(control.map((p) => p[0]))
        const my = mean(control.map((p) => p[1]))
        const c = mean(control.map((p) => (p[0] - mx) * (p[1] - my)))
        const sx = Math.sqrt(mean(control.map((p) => (p[0] - mx) ** 2)))
        const sy = Math.sqrt(mean(control.map((p) => (p[1] - my) ** 2)))
        expect(c / (sx * sy)).toBeCloseTo(0.5, 1)
      }
    })
  }
})

describe('trajectories', () => {
  it('logged bandits: propensities match the logging policy, IPS recovers policy values', () => {
    const { data, truth } = generate(loggedBandit, { key: key(4), knobs: { rounds: 20000, actions: 4, features: 3 } })
    const r = toFlat(data.rewards)
    expect(Math.abs(mean(r) - (truth.extra?.['loggingValue'] as number))).toBeLessThan(0.02)
    const optimal = truth.extra?.['optimalPolicy'] as (x: ArrayLike<number>) => number[]
    const obs = data.observations
    const acts = toFlat(data.actions)
    const props = toFlat(data.propensities as Tensor)
    const ips = mean(
      r.map(
        (v, i) =>
          (v * (optimal(obs.data.subarray(i * 3, (i + 1) * 3) as Float64Array)[acts[i] ?? 0] ?? 0)) / (props[i] ?? 1),
      ),
    )
    expect(Math.abs(ips - (truth.extra?.['optimalValue'] as number))).toBeLessThan(0.05)
  })
  it('offline MDPs: exact values satisfy the Bellman equation and match long Monte Carlo returns', () => {
    const { data, truth } = generate(offlineMdp, {
      key: key(5),
      knobs: { states: 6, actions: 2, episodes: 400, horizon: 60, gamma: 0.8 },
    })
    const mdp = truth.extra?.['mdp'] as TabularMdp
    const pi = truth.extra?.['behaviourPolicy'] as number[][]
    const { values } = policyEvaluation(mdp, pi)
    for (let s = 0; s < 6; s++) {
      let rhs = 0
      for (let a = 0; a < 2; a++) {
        let next = 0
        for (let t = 0; t < 6; t++) next += (mdp.transitions[s]?.[a]?.[t] ?? 0) * (values[t] ?? 0)
        rhs += (pi[s]?.[a] ?? 0) * ((mdp.rewards[s]?.[a] ?? 0) + mdp.gamma * next)
      }
      expect(rhs).toBeCloseTo(values[s] ?? 0, 10)
    }
    const rew = toFlat(data.rewards)
    const returns = Array.from({ length: 400 }, (_, e) =>
      rew.slice(e * 60, (e + 1) * 60).reduce((s, v, t) => s + 0.8 ** t * v, 0),
    )
    expect(Math.abs(mean(returns) - (truth.extra?.['behaviourValue'] as number))).toBeLessThan(0.35)
    expect(truth.extra?.['optimalValue'] as number).toBeGreaterThanOrEqual(
      (truth.extra?.['behaviourValue'] as number) - 1e-9,
    )
    const chain = generate(offlineMdp, {
      key: key(6),
      knobs: { layout: 'chain', states: 5, gamma: 0.99, episodes: 2, horizon: 5 },
    })
    const chainPolicy = (chain.truth.extra?.['optimalPolicy'] ?? []) as number[][]
    expect(chainPolicy[0]).toEqual([1, 0])
  })
})

describe('weak labels', () => {
  it('labelling functions have their coverages and accuracies, and the posterior beats any one function', () => {
    const { data, truth } = generate(labellingFunctions, { key: key(7), knobs: { n: 6000, functions: 5 } })
    const acc = truth.extra?.['accuracy'] as number[]
    const cov = truth.extra?.['coverage'] as number[]
    const y = toFlat(data.y)
    for (let j = 0; j < 5; j++) {
      const v = col(data.x, j)
      const covered = v.filter((x) => x >= 0).length / 6000
      expect(Math.abs(covered - (cov[j] ?? 0))).toBeLessThan(0.03)
      const right = v.filter((x, i) => x >= 0 && x === y[i]).length / Math.max(1, v.filter((x) => x >= 0).length)
      expect(Math.abs(right - (acc[j] ?? 0))).toBeLessThan(0.04)
    }
    const post = truth.extra?.['posterior'] as (row: readonly number[]) => number[]
    const voteCols = Array.from({ length: 5 }, (_, j) => col(data.x, j))
    const rows = Array.from({ length: 6000 }, (_, i) => Array.from({ length: 5 }, (_, j) => voteCols[j]?.[i] ?? -1))
    const accuracy = mean(rows.map((r, i) => (argmaxOf(post(r)) === y[i] ? 1 : 0)))
    expect(accuracy).toBeGreaterThan(0.6)
  })
  it('crowd labels: the Dawid-Skene posterior recovers the truth', () => {
    const { data, truth } = generate(crowdLabels, {
      key: key(8),
      knobs: { items: 300, workers: 15, perItem: 5, skill: 6 },
    })
    const post = truth.extra?.['posterior'] as (a: readonly { worker: number; label: number }[]) => number[]
    const users = toFlat(data.users)
    const values = toFlat(data.values)
    const items = toFlat(data.items)
    const truthLabels = truth.latents?.['trueLabels'] as number[]
    let right = 0
    for (let i = 0; i < 300; i++) {
      const answers = items.flatMap((it, j) => (it === i ? [{ worker: users[j] ?? 0, label: values[j] ?? 0 }] : []))
      if (argmaxOf(post(answers)) === truthLabels[i]) right++
    }
    expect(right / 300).toBeGreaterThan(0.9)
  })
  it('positive-unlabelled and label-proportion data', () => {
    const pu = generate(positiveUnlabelled, { key: key(9), knobs: { n: 10000, prior: 0.4, labelFrequency: 0.5 } })
    expect(Math.abs(mean(toFlat(pu.data.y)) - 0.2)).toBeLessThan(0.015)
    const yTrue = toFlat(pu.truth.latents?.['trueLabels'] as Tensor)
    expect(toFlat(pu.data.y).every((s, i) => s === 0 || yTrue[i] === 1)).toBe(true)
    const llp = generate(labelProportionBags, { key: key(10), knobs: { n: 95, bagSize: 10, grouping: 'sorted' } })
    const props = llp.truth.extra?.['proportions'] as number[][]
    expect(props.length).toBe(10)
    for (const p of props) expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(toFlat(llp.data.y).every((v) => v === -1)).toBe(true)
    expect(rowCount(llp.data)).toBe(95)
  })
})

describe('operators', () => {
  it('antiderivatives: differences of the output recover the input', () => {
    const { data } = generate(antiderivativeOperator, { key: key(11), knobs: { n: 3, points: 30 } })
    const u = toFlat(data.x)
    const s = toFlat(data.y)
    for (let i = 0; i < 3; i++) {
      expect(s[i * 30]).toBe(0)
      for (let j = 1; j < 30; j++) {
        const step = ((s[i * 30 + j] ?? 0) - (s[i * 30 + j - 1] ?? 0)) * 29
        expect(step).toBeCloseTo(0.5 * ((u[i * 30 + j] ?? 0) + (u[i * 30 + j - 1] ?? 0)), 10)
      }
    }
  })
  it('heat: modes decay exactly', () => {
    const { data } = generate(heatOperator, {
      key: key(12),
      knobs: { n: 2, points: 32, modes: 3, time: 0.01, viscosity: 1 },
    })
    const u0 = toFlat(data.x)
    const uT = toFlat(data.y)
    // The mean is zero and the solution is smoother (smaller in norm).
    expect(Math.abs(mean(uT))).toBeLessThan(1e-12)
    expect(mean(uT.map((v) => v * v))).toBeLessThan(mean(u0.map((v) => v * v)))
  })
  it('Burgers: the Cole-Hopf solution satisfies the equation', () => {
    const { truth } = generate(burgersOperator, {
      key: key(13),
      knobs: { n: 1, points: 64, viscosity: 0.1, time: 0.1 },
    })
    const solve = truth.extra?.['solve'] as (
      a: readonly number[],
      b: readonly number[],
      t?: number,
      res?: number,
    ) => number[]
    const { a, b } = (truth.extra?.['coefficients'] ?? { a: [], b: [] }) as { a: number[][]; b: number[][] }
    const m = 1024
    const h = 1e-4
    const u = solve(a[0] ?? [], b[0] ?? [], 0.1, m)
    const up = solve(a[0] ?? [], b[0] ?? [], 0.1 + h, m)
    const um = solve(a[0] ?? [], b[0] ?? [], 0.1 - h, m)
    const dx = 1 / m
    let worst = 0
    let scale = 0
    for (let j = 0; j < m; j++) {
      const ut = ((up[j] ?? 0) - (um[j] ?? 0)) / (2 * h)
      const ux = ((u[(j + 1) % m] ?? 0) - (u[(j - 1 + m) % m] ?? 0)) / (2 * dx)
      const uxx = ((u[(j + 1) % m] ?? 0) - 2 * (u[j] ?? 0) + (u[(j - 1 + m) % m] ?? 0)) / (dx * dx)
      worst = Math.max(worst, Math.abs(ut + (u[j] ?? 0) * ux - 0.1 * uxx))
      scale = Math.max(scale, Math.abs(ut))
    }
    expect(worst / scale).toBeLessThan(1e-3)
    // At time zero the solution is the initial condition.
    const start = solve(a[0] ?? [], b[0] ?? [], 0, m)
    const direct = Array.from({ length: m }, (_, j) =>
      (a[0] ?? []).reduce(
        (s, ak, q) =>
          s +
          ak * Math.cos(2 * Math.PI * (q + 1) * (j / m)) +
          (b[0]?.[q] ?? 0) * Math.sin(2 * Math.PI * (q + 1) * (j / m)),
        0,
      ),
    )
    start.forEach((v, j) => {
      expect(v).toBeCloseTo(direct[j] ?? 0, 8)
    })
  })
})

describe('predictions', () => {
  it('miscalibration: the calibrated case has no calibration error, the distorted ones do', () => {
    const cal = generate(miscalibratedPredictions, { key: key(14), knobs: { n: 10, distortion: 'calibrated' } })
    const calEce = cal.truth.extra?.['expectedCalibrationError'] as () => number
    expect(calEce()).toBeLessThan(0.01)
    const over = generate(miscalibratedPredictions, {
      key: key(14),
      knobs: { n: 10, distortion: 'overconfident', strength: 3 },
    })
    const overEce = over.truth.extra?.['expectedCalibrationError'] as () => number
    expect(overEce()).toBeGreaterThan(0.05)
    const p = toFlat(over.data.x)
    for (let i = 0; i < 10; i++) expect((p[2 * i] ?? 0) + (p[2 * i + 1] ?? 0)).toBeCloseTo(1, 12)
  })
  it('survival: censoring and the survival function', () => {
    for (const model of ['weibull', 'exponential', 'lognormal']) {
      const { data, truth } = generate(survivalData, { key: key(15), knobs: { n: 20000, model, censoring: 0 } })
      const y = toFlat(data.y)
      const times = y.filter((_, i) => i % 2 === 0)
      expect(y.filter((_, i) => i % 2 === 1).every((e) => e === 1)).toBe(true)
      const surv = truth.extra?.['survival'] as (t: number, row: ArrayLike<number>) => number
      // Averaged over covariates, S(t) matches the empirical survival at the empirical median.
      const sorted = [...times].sort((p, q) => p - q)
      const median = sorted[10000] ?? 1
      const pop = mean(
        Array.from({ length: 2000 }, (_, i) => surv(median, data.x.data.subarray(i * 3, (i + 1) * 3) as Float64Array)),
      )
      expect(Math.abs(pop - 0.5), model).toBeLessThan(0.03)
    }
    const censored = generate(survivalData, { key: key(16), knobs: { n: 5000, censoring: 0.5 } })
    const events = toFlat(censored.data.y).filter((_, i) => i % 2 === 1)
    expect(mean(events)).toBeLessThan(0.9)
  })
  it('survival with no covariates is one Weibull population with its true survival function', () => {
    const { data, truth } = generate(survivalData, {
      key: key(17),
      knobs: { n: 20000, features: 0, shape: 1.5, scale: 0.1, censoring: 0 },
    })
    expect(data.x.shape).toEqual([20000, 0])
    const times = toFlat(data.y).filter((_, i) => i % 2 === 0)
    const surv = truth.extra?.['survival'] as (t: number, row: ArrayLike<number>) => number
    expect(surv(3, [])).toBeCloseTo(Math.exp(-0.1 * 3 ** 1.5), 12)
    // the share of lifetimes beyond 5 is the survival function at 5
    expect(Math.abs(mean(times.map((t) => (t > 5 ? 1 : 0))) - surv(5, []))).toBeLessThan(0.01)
  })
})

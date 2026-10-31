import { describe, expect, it } from 'vitest'
import { hedgeRegretBound, ogdRegretBound } from '../learn/theory.ts'
import { foldIn, key, normal, uniform } from '../random/index.ts'
import { square, sub, sum, toFlat, type Value } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import { linearLosses, onlineGradientDescent, runOnline } from './online.ts'
import {
  adagradRegretBound,
  bestSwitchingLoss,
  exp4Algorithm,
  exp4RegretBound,
  expertsAlgorithm,
  fixedShare,
  followTheLeader,
  followThePerturbedLeader,
  hedgeRule,
  hindsightComparator,
  onlineAdagrad,
  pNormMirrorDescent,
  randomisedMajorityBound,
  runExperts,
  variableShare,
  weightedMajority,
  weightedMajorityMistakeBound,
} from './online-experts.ts'

const alternating = (T: number): number[][] =>
  Array.from({ length: T }, (_, t) => (t === 0 ? [0.5, 0] : t % 2 === 1 ? [0, 1] : [1, 0]))

describe('follow the leader and the perturbed leader', () => {
  it('FTL has linear regret on the alternating sequence; Hedge and FTPL do not', () => {
    const T = 400
    const problem = { losses: alternating(T) }
    const ftl = runExperts(followTheLeader(), problem)
    expect(ftl.regret).toBeGreaterThan(0.45 * T)
    const eta = Math.sqrt((8 * Math.log(2)) / T)
    const hedge = runExperts(hedgeRule(eta), problem)
    expect(hedge.regret).toBeLessThanOrEqual(hedgeRegretBound(T, 2, eta))
    // FTPL is randomised: the expected regret over keys is sublinear.
    let total = 0
    const reps = 30
    for (let r = 0; r < reps; r++)
      total += runExperts(followThePerturbedLeader(Math.sqrt(Math.log(2) / T)), problem, key(r)).regret / reps
    expect(total).toBeLessThan(0.2 * T)
    expect(total).toBeLessThan(ftl.regret / 3)
  })

  it('FTL tie-breaks to the lowest index and is optimal on a stochastic sequence', () => {
    const flat = toFlat(uniform(key(1), [3000]))
    const losses = Array.from({ length: 1000 }, (_, t) => [
      0.3 * (flat[3 * t] ?? 0),
      0.5 * (flat[3 * t + 1] ?? 0) + 0.2,
      0.5 * (flat[3 * t + 2] ?? 0) + 0.2,
    ])
    const r = runExperts(followTheLeader(), { losses })
    expect(r.regret).toBeLessThan(5)
    expect(r.finalProbabilities[0]).toBe(1)
  })
})

describe('weighted majority', () => {
  const N = 8
  const T = 300
  function binaryProblem(
    seed: number,
    noise: number,
  ): { problem: { losses: number[][]; predictions: number[][]; outcomes: number[] }; best: number } {
    const u = toFlat(uniform(key(seed), [T * N + T]))
    const outcomes = Array.from({ length: T }, (_, t) => ((u[T * N + t] ?? 0) < 0.5 ? 1 : 0))
    const predictions = Array.from({ length: T }, (_, t) =>
      Array.from({ length: N }, (_, i) =>
        i === 3
          ? (u[t * N + i] ?? 0) < noise
            ? 1 - (outcomes[t] ?? 0)
            : (outcomes[t] ?? 0)
          : (u[t * N + i] ?? 0) < 0.5
            ? 1
            : 0,
      ),
    )
    const losses = predictions.map((row, t) => row.map((p) => (p === outcomes[t] ? 0 : 1)))
    const best = Math.min(...Array.from({ length: N }, (_, i) => losses.reduce((a, row) => a + (row[i] ?? 0), 0)))
    return { problem: { losses, predictions, outcomes }, best }
  }

  it('makes at most (ln N + m ln(1/beta)) / ln(2 / (1 + beta)) mistakes, over replicates and betas', () => {
    for (const beta of [0.5, 0.8]) {
      for (let rep = 0; rep < 10; rep++) {
        const { problem, best } = binaryProblem(10 + rep, 0.1)
        const r = runExperts(weightedMajority({ beta }), problem)
        expect(r.cumulativeLoss).toBeLessThanOrEqual(weightedMajorityMistakeBound(N, best, beta))
        // Non-vacuous: far fewer mistakes than the T / 2 of guessing, and within a small multiple of the best expert.
        expect(r.cumulativeLoss).toBeLessThan(0.3 * T)
      }
    }
  })

  it('the randomised version has expected mistakes under (ln N + m ln(1/beta)) / (1 - beta) and the formulas are right', () => {
    const { problem, best } = binaryProblem(31, 0.15)
    const r = runExperts(weightedMajority({ beta: 0.5, randomised: true }), problem)
    expect(r.cumulativeLoss).toBeLessThanOrEqual(randomisedMajorityBound(N, best, 0.5))
    expect(weightedMajorityMistakeBound(4, 10, 0.5)).toBeCloseTo((Math.log(4) + 10 * Math.log(2)) / Math.log(4 / 3), 12)
    expect(randomisedMajorityBound(4, 10, 0.5)).toBeCloseTo((Math.log(4) + 10 * Math.log(2)) / 0.5, 12)
  })

  it('demands predictions and outcomes and rejects a bad beta', () => {
    expect(() =>
      runExperts(weightedMajority(), {
        losses: [
          [0, 1],
          [1, 0],
        ],
      }),
    ).toThrow()
    expect(() => weightedMajority({ beta: 1 })).toThrow()
  })
})

describe('tracking: fixed share and variable share', () => {
  const T = 1000
  const N = 5
  // The best expert changes every 200 rounds (0, 1, 2, 3, 4): it suffers no loss, the others always do.
  const losses = Array.from({ length: T }, (_, t) =>
    Array.from({ length: N }, (_, i) => (i === Math.floor(t / 200) ? 0 : 1)),
  )

  it('the best-switching comparator agrees with brute force on a small instance', () => {
    const l = [
      [0.2, 0.9, 0.4],
      [0.8, 0.1, 0.5],
      [0.3, 0.7, 0.0],
      [0.6, 0.2, 0.9],
      [0.1, 0.4, 0.8],
      [0.5, 0.3, 0.2],
    ]
    for (const k of [0, 1, 2]) {
      const dp = bestSwitchingLoss(l, k)
      for (let t = 1; t <= l.length; t++) {
        let best = Number.POSITIVE_INFINITY
        const seq = new Array<number>(t).fill(0)
        const rec = (pos: number): void => {
          if (pos === t) {
            let sw = 0
            for (let i = 1; i < t; i++) if (seq[i] !== seq[i - 1]) sw++
            if (sw <= k)
              best = Math.min(
                best,
                seq.reduce((a, e, i) => a + (l[i]?.[e] ?? 0), 0),
              )
            return
          }
          for (let e = 0; e < 3; e++) {
            seq[pos] = e
            rec(pos + 1)
          }
        }
        rec(0)
        expect(dp[t - 1]).toBeCloseTo(best, 12)
      }
    }
  })

  it('Hedge has linear regret against the best 4-switch sequence; fixed share and variable share keep it small', () => {
    const switching = (rule: Parameters<typeof expertsAlgorithm>[0]): number => {
      const alg = expertsAlgorithm(rule, { losses }, { switches: 4 })
      const rec = recorder()
      const r = runAlgorithm(alg, { maxSteps: T, tracer: rec })
      const comparator = bestSwitchingLoss(losses, 4)
      return r.state.cumulativeLoss - (comparator[T - 1] ?? 0)
    }
    const hedge = switching(hedgeRule(1))
    const fixed = switching(fixedShare({ eta: 1, alpha: 0.005 }))
    const variable = switching(variableShare({ eta: 1, alpha: 0.02 }))
    expect(hedge).toBeGreaterThan(300)
    // O((k + 1) ln N + k ln T) / eta: about (5 * 1.6 + 4 * 6.9) / 1 = 36 for fixed share.
    expect(fixed).toBeLessThan(45)
    expect(variable).toBeLessThan(80)
    expect(fixed).toBeLessThan(hedge / 5)
    // Against the best single expert Hedge looks fine: the comparator is what makes the difference.
    expect(runExperts(hedgeRule(1), { losses }).regret).toBeLessThan(hedge)
  })

  it('the algorithm records the switching regret on the tape and its state serialises', () => {
    const alg = expertsAlgorithm(fixedShare({ eta: 1, alpha: 0.01 }), { losses: losses.slice(0, 50) }, { switches: 1 })
    const rec = recorder()
    const r = runAlgorithm(alg, { maxSteps: 50, tracer: rec })
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(r.state))))?.cumulativeLoss).toBe(
      r.state.cumulativeLoss,
    )
    expect(JSON.stringify(rec.run())).toContain('switchingRegret')
  })

  it('rejects bad parameters', () => {
    expect(() => fixedShare({ eta: 1, alpha: 0 })).toThrow()
    expect(() => variableShare({ eta: 0, alpha: 0.1 })).toThrow()
    expect(() => expertsAlgorithm(hedgeRule(1), { losses: [] })).toThrow()
  })
})

describe('Exp4', () => {
  it('has pseudo-regret under sqrt(2 K T ln N) on average and sublinear growth', () => {
    const K = 4
    const N = 3
    const T = 2000
    // Arm 2 has the smallest mean loss; expert 0 always recommends it, experts 1 and 2 recommend other arms.
    const means = [0.6, 0.5, 0.3, 0.55]
    const advice = [
      [0, 0, 1, 0],
      [0.5, 0.5, 0, 0],
      [0, 0, 0, 1],
    ]
    let regret = 0
    let half = 0
    const reps = 20
    for (let rep = 0; rep < reps; rep++) {
      const u = toFlat(uniform(foldIn(key(5), rep), [T * K]))
      const losses = Array.from({ length: T }, (_, t) => means.map((m, a) => ((u[t * K + a] ?? 0) < m ? 1 : 0)))
      const alg = exp4Algorithm(advice, losses)
      const mid = runAlgorithm(alg, { maxSteps: T / 2, key: key(100 + rep) })
      const end = runAlgorithm(alg, { maxSteps: T, key: key(100 + rep) })
      const best = (s: { t: number }): number => {
        let b = Number.POSITIVE_INFINITY
        for (let i = 0; i < N; i++) {
          let tot = 0
          for (let t = 0; t < s.t; t++) tot += (advice[i] ?? []).reduce((a, p, k) => a + p * (losses[t]?.[k] ?? 0), 0)
          b = Math.min(b, tot)
        }
        return b
      }
      regret += (end.state.expectedLoss - best(end.state)) / reps
      half += (mid.state.expectedLoss - best(mid.state)) / reps
    }
    expect(regret).toBeLessThanOrEqual(exp4RegretBound(T, K, N))
    expect(regret).toBeLessThan(0.25 * T * 0.3)
    expect(regret).toBeLessThan(2.4 * half)
    expect(exp4RegretBound(1000, 4, 10)).toBeCloseTo(Math.sqrt(2 * 1000 * 4 * Math.log(10)), 6)
  }, 60000)

  it('with identity advice it is Exp3: it concentrates on the best arm', () => {
    const T = 3000
    const u = toFlat(uniform(key(8), [T * 3]))
    const losses = Array.from({ length: T }, (_, t) => [0.8, 0.2, 0.7].map((m, a) => ((u[t * 3 + a] ?? 0) < m ? 1 : 0)))
    const r = runAlgorithm(
      exp4Algorithm(
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        losses,
      ),
      { maxSteps: T, key: key(9) },
    )
    expect(r.state.armProbabilities[1]).toBeGreaterThan(0.7)
  })
})

describe('online mirror descent, AdaGrad and the comparator in hindsight', () => {
  const linear = (seed: number, T: number, d: number): number[][] => {
    const z = toFlat(normal(key(seed), [T * d]))
    return Array.from({ length: T }, (_, t) =>
      Array.from({ length: d }, (_, j) => (z[t * d + j] ?? 0) * (j < 2 ? 1 : 0.05) + (j === 0 ? 0.3 : 0)),
    )
  }

  it('p-norm mirror descent with q = 2 is online gradient descent', () => {
    const grads = linear(1, 40, 3)
    const a = runOnline(pNormMirrorDescent({ q: 2, learningRate: 0.1 }), linearLosses(grads), { dim: 3 })
    const b = runOnline(onlineGradientDescent({ learningRate: 0.1 }), linearLosses(grads), { dim: 3 })
    a.plays.forEach((x, t) => {
      x.forEach((v, j) => {
        expect(v).toBeCloseTo(b.plays[t]?.[j] ?? 0, 10)
      })
    })
    // Another norm moves differently but keeps the dual structure: theta is minus eta times the gradient sum.
    const c = runOnline(pNormMirrorDescent({ q: 4, learningRate: 0.1 }), linearLosses(grads), { dim: 3 })
    expect(c.plays[10]).not.toEqual(b.plays[10])
    expect(() => pNormMirrorDescent({ q: 1 })).toThrow()
  })

  it('AdaGrad stays under its data-dependent bound on a sparse sequence, which is far below the worst-case bound', () => {
    const T = 600
    const d = 6
    const radius = 1
    const z = toFlat(normal(key(2), [T * d]))
    const u = toFlat(uniform(key(3), [T * d]))
    const grads = Array.from({ length: T }, (_, t) =>
      Array.from(
        { length: d },
        (_, j) => ((u[t * d + j] ?? 0) < 0.8 ** (j * 2.5) ? 1 : 0) * (z[t * d + j] ?? 0) + (j % 2 === 0 ? 0.2 : -0.1),
      ),
    )
    const sum = grads.reduce((a, g) => a.map((v, j) => v + (g[j] ?? 0)), new Array<number>(d).fill(0))
    const comparator = sum.map((v) => -radius * Math.sign(v))
    const r = runOnline(onlineAdagrad({ radius }), linearLosses(grads), { dim: d, comparator })
    const regret = r.regret[T - 1] ?? 0
    const bound = adagradRegretBound(grads, 2 * radius)
    expect(regret).toBeLessThanOrEqual(bound)
    const G = Math.max(...grads.map((g) => Math.hypot(...g)))
    expect(bound).toBeLessThan(ogdRegretBound(T, 2 * radius * Math.sqrt(d), G))
    expect(regret).toBeLessThan(0.15 * T)
    expect(
      adagradRegretBound(
        [
          [3, 0],
          [4, 0],
        ],
        2,
      ),
    ).toBeCloseTo(Math.SQRT2 * 2 * 5, 12)
  })

  it('finds the best fixed point in hindsight (quadratics: the mean; with a box: clipped)', () => {
    const f = (c: number) => (x: Value) => sum(square(sub(x, c)))
    expect(hindsightComparator([f(1), f(3), f(8)], 1).point[0]).toBeCloseTo(4, 5)
    expect(hindsightComparator([f(1), f(3), f(8)], 1, { box: 2 }).point[0]).toBeCloseTo(2, 5)
  })
})

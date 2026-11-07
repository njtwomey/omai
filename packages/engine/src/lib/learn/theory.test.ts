import { describe, expect, it } from 'vitest'
import { klDivergence, normal as normalDist } from '../dists/index.ts'
import { hedge, linearLosses, onlineGradientDescent, runOnline } from '../optim/index.ts'
import { foldIn, key, normal, randint, uniform } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import {
  biasVariance,
  biasVarianceZeroOne,
  catoniBound,
  conceptClasses,
  covariancePenalty,
  effectiveDegreesOfFreedom,
  empiricalRademacher,
  empiricalStability,
  gaussianGibbsRisk,
  gaussianKl,
  growthFunction,
  hedgeRegretBound,
  isShattered,
  klBernoulli,
  klInverse,
  klInverseLower,
  knnDegreesOfFreedom,
  lassoDegreesOfFreedom,
  marginBound,
  mcAllesterBound,
  monteCarloBiasVariance,
  mutualInformationBound,
  ogdRegretBound,
  pacFiniteBound,
  pacSampleComplexity,
  rademacherBoundFinite,
  rademacherBoundLinear,
  rademacherGeneralisationBound,
  ridgeDegreesOfFreedom,
  ridgeUniformStability,
  sauerBound,
  seegerBound,
  shatterCount,
  stabilityBound,
  unionBound,
  vcBound,
  vcDimensionByShattering,
  vcSampleComplexity,
} from './theory.ts'

const rows = (flat: ArrayLike<number>, n: number, d: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))

describe('PAC bounds for finite classes', () => {
  it('match the textbook formulas', () => {
    const lnH = Math.log(1000)
    expect(pacSampleComplexity(lnH, 0.1, 0.05)).toBe(Math.ceil((lnH + Math.log(20)) / 0.1))
    expect(pacSampleComplexity(lnH, 0.1, 0.05, 'agnostic')).toBe(Math.ceil((2 * (lnH + Math.log(40))) / 0.01))
    expect(pacFiniteBound(100, lnH, 0.05)).toBeCloseTo((lnH + Math.log(20)) / 100, 12)
    expect(pacFiniteBound(100, lnH, 0.05, 'agnostic')).toBeCloseTo(Math.sqrt((lnH + Math.log(20)) / 200), 12)
    expect(unionBound(1000, Math.log(100), 0.1)).toBeCloseTo(200 * Math.exp(-20), 15)
    expect(unionBound(5, Math.log(100), 0.01)).toBe(1)
  })

  it('invert one another: the sample size makes the bound equal epsilon', () => {
    const lnH = 12
    const n = pacSampleComplexity(lnH, 0.05, 0.01)
    expect(pacFiniteBound(n, lnH, 0.01)).toBeLessThanOrEqual(0.05)
    expect(pacFiniteBound(n - 1, lnH, 0.01)).toBeGreaterThan(0.05)
    const m = pacSampleComplexity(lnH, 0.05, 0.01, 'agnostic')
    expect(2 * pacFiniteBound(m, lnH, 0.01, 'agnostic') - 0.05).toBeLessThan(1e-3)
  })

  it('rejects out-of-range arguments', () => {
    expect(() => pacFiniteBound(0, 1, 0.1)).toThrow()
    expect(() => pacFiniteBound(10, 1, 1.5)).toThrow()
  })

  // A planted problem: 51 thresholds on [0, 1], target threshold 0.5 on uniform inputs, labels flipped with probability q.
  const thresholds = Array.from({ length: 51 }, (_, i) => i / 50)
  const riskOf = (t: number, q: number): number => (1 - 2 * q) * Math.abs(t - 0.5) + q

  it('the agnostic bound holds in every simulated draw and is not vacuous', () => {
    const n = 400
    const delta = 0.05
    const bound = pacFiniteBound(n, Math.log(thresholds.length), delta, 'agnostic')
    let violations = 0
    let worstGap = 0
    for (let rep = 0; rep < 300; rep++) {
      const u = toFlat(uniform(key(rep), [n]))
      const flip = toFlat(uniform(key(10000 + rep), [n]))
      const q = 0.2
      const labels = u.map((v, i) => ((v >= 0.5 ? 1 : 0) ^ ((flip[i] ?? 1) < q ? 1 : 0)) as number)
      let gap = 0
      for (const t of thresholds) {
        const emp = u.reduce((a, v, i) => a + ((v >= t ? 1 : 0) !== labels[i] ? 1 : 0), 0) / n
        gap = Math.max(gap, riskOf(t, q) - emp)
      }
      worstGap = Math.max(worstGap, gap)
      if (gap > bound) violations++
    }
    expect(violations / 300).toBeLessThanOrEqual(delta)
    expect(bound).toBeLessThan(0.15)
    // Not vacuous: the worst gap seen is within a factor of five of the bound.
    expect(worstGap).toBeGreaterThan(bound / 5)
  })

  it('the realisable bound holds for the worst consistent threshold', () => {
    const n = 100
    const delta = 0.05
    const bound = pacFiniteBound(n, Math.log(thresholds.length), delta)
    let violations = 0
    for (let rep = 0; rep < 500; rep++) {
      const u = toFlat(uniform(key(50000 + rep), [n]))
      const consistent = thresholds.filter((t) => u.every((v) => v >= t === v >= 0.5))
      const worst = Math.max(...consistent.map((t) => riskOf(t, 0)))
      if (worst > bound) violations++
    }
    expect(violations / 500).toBeLessThanOrEqual(delta)
  })
})

describe('VC dimension', () => {
  it('Sauer-Shelah is the sum of binomials and 2^n below d', () => {
    expect(sauerBound(5, 2)).toBe(16)
    expect(sauerBound(3, 5)).toBe(8)
    expect(sauerBound(10, 3)).toBe(1 + 10 + 45 + 120)
  })

  it('Vapnik bound has the shape of 1/sqrt(n) and sample complexity is increasing in d', () => {
    expect(vcBound(10000, 10, 0.05)).toBeCloseTo(Math.sqrt((10 * (Math.log(2000) + 1) + Math.log(80)) / 10000), 12)
    expect(vcBound(40000, 10, 0.05)).toBeLessThan(vcBound(10000, 10, 0.05) * 0.55)
    expect(vcSampleComplexity(5, 0.1, 0.05)).toBeGreaterThan(vcSampleComplexity(3, 0.1, 0.05))
    expect(() => vcBound(5, 10, 0.1)).toThrow()
  })

  it('counts the labellings of simple classes by shattering enumeration', () => {
    expect(shatterCount([[0], [1], [2]], conceptClasses.halfLine)).toBe(4)
    expect(shatterCount([[0], [1], [2], [3]], conceptClasses.interval)).toBe(11)
    expect(isShattered([[0], [1]], conceptClasses.interval)).toBe(true)
    expect(isShattered([[0], [1], [2]], conceptClasses.interval)).toBe(false)
    const diamond = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]
    expect(isShattered(diamond, conceptClasses.boxes(2))).toBe(true)
    expect(isShattered([...diamond, [0, 0]], conceptClasses.boxes(2))).toBe(false)
    const xor = [
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0],
    ]
    expect(shatterCount(xor, conceptClasses.halfspaces(2))).toBe(14)
    expect(isShattered(xor.slice(0, 3), conceptClasses.halfspaces(2))).toBe(true)
  })

  it('finds the growth function of half-spaces in the plane, k^2 - k + 2, below the Sauer-Shelah bound', () => {
    for (const k of [2, 3, 4, 5, 6]) {
      const pi = growthFunction(conceptClasses.halfspaces(2), k)
      expect(pi).toBe(k * k - k + 2)
      expect(pi).toBeLessThanOrEqual(sauerBound(k, 3))
    }
    expect(growthFunction(conceptClasses.interval, 5)).toBe(16)
    expect(growthFunction(conceptClasses.halfLine, 5)).toBe(6)
  })

  it('recovers the VC dimension of the classes by enumeration', () => {
    const classes = [
      conceptClasses.halfLine,
      conceptClasses.interval,
      conceptClasses.boxes(2),
      conceptClasses.halfspaces(2),
      conceptClasses.halfspaces(3),
    ]
    for (const cls of classes) {
      const found = vcDimensionByShattering(cls, 6)
      expect(found.vcDimension).toBe(cls.vcDimension)
      expect(found.nextFailed).toBe(true)
      expect(isShattered(found.witness, cls)).toBe(true)
    }
  })
})

describe('Rademacher complexity', () => {
  it('is zero for a single predictor and E|mean of signs| for a predictor and its negation', () => {
    const one = empiricalRademacher([new Array(100).fill(1)], { draws: 3000, key: key(1) })
    expect(Math.abs(one.estimate)).toBeLessThan(4 * one.standardError + 1e-12)
    const two = empiricalRademacher([new Array(100).fill(1), new Array(100).fill(-1)], { draws: 3000, key: key(2) })
    // sup(m, -m) = |m| with m the mean of 100 signs; E|m| -> sqrt(2 / (pi n)) = 0.0798.
    expect(two.estimate).toBeCloseTo(Math.sqrt(2 / (Math.PI * 100)), 2)
  })

  it('stays below Massart for a random finite class and approaches it as the class grows', () => {
    const n = 60
    const h = 40
    const preds = rows(
      Array.from(toFlat(randint(key(3), [h * n], 0, 2)), (v) => 2 * v - 1),
      h,
      n,
    )
    const est = empiricalRademacher(preds, { draws: 1000, key: key(4) })
    const bound = rademacherBoundFinite(n, Math.log(h))
    expect(est.estimate).toBeLessThan(bound)
    expect(est.estimate).toBeGreaterThan(0.5 * bound)
  })

  it('is exactly B / sqrt(n) for a linear class on orthonormal points and below the bound on random ones', () => {
    const n = 25
    const identity = rows(
      Array.from({ length: n * n }, (_, i) => (Math.floor(i / n) === i % n ? 1 : 0)),
      n,
      n,
    )
    const est = empiricalRademacher(identity, { norm: 2, draws: 50, key: key(5) })
    expect(est.estimate).toBeCloseTo(2 / Math.sqrt(n), 12)
    expect(rademacherBoundLinear(2, n, 1)).toBeCloseTo(2 / Math.sqrt(n), 12)
    const x = rows(toFlat(normal(key(6), [80 * 4])), 80, 4)
    const random = empiricalRademacher(x, { norm: 1.5, draws: 500, key: key(7) })
    expect(random.estimate).toBeLessThanOrEqual(rademacherBoundLinear(1.5, 80, undefined, x) + 1e-12)
    expect(random.estimate).toBeGreaterThan(0.8 * rademacherBoundLinear(1.5, 80, undefined, x))
  })

  it('the empirical-complexity generalisation bound holds for ERM over a finite class of directions, across draws', () => {
    // 72 half-plane classifiers sign(w . x) with w at multiples of 5 degrees, points on the circle, labels from the
    // half-plane x_1 >= 0 flipped with probability q. The true error of direction theta is q + (1 - 2q) |theta| / pi.
    const n = 300
    const delta = 0.05
    const q = 0.1
    const directions = Array.from({ length: 72 }, (_, a) => (a * Math.PI) / 36)
    const trueError = (theta: number): number => {
      const t = Math.abs(((theta + Math.PI) % (2 * Math.PI)) - Math.PI)
      return q + (1 - 2 * q) * (t / Math.PI)
    }
    let violations = 0
    let bound = 0
    let worstGap = 0
    for (let rep = 0; rep < 100; rep++) {
      const ang = toFlat(uniform(key(900 + rep), [n], { low: 0, high: 2 * Math.PI }))
      const flip = toFlat(uniform(key(1900 + rep), [n]))
      const y = Array.from(ang, (a, i) => (Math.cos(a) >= 0 ? 1 : -1) * ((flip[i] ?? 1) < q ? -1 : 1))
      const preds = directions.map((th) => Array.from(ang, (a) => (Math.cos(a - th) >= 0 ? 1 : -1)))
      const errs = preds.map((row) => row.reduce((acc, v, i) => acc + (v !== y[i] ? 1 : 0), 0) / n)
      // The loss class 1[h != y] = (1 - y h) / 2 has half the complexity of the +-1 class on the sample.
      const complexity =
        empiricalRademacher(
          preds.map((row) => row.map((v, i) => v * (y[i] ?? 0))),
          { draws: 100, key: key(rep) },
        ).estimate / 2
      bound = rademacherGeneralisationBound(complexity, n, delta)
      // The uniform deviation over the class is what the bound controls (it holds for every h at once).
      const gap = Math.max(...directions.map((th, a) => trueError(th) - (errs[a] ?? 0)))
      worstGap = Math.max(worstGap, gap)
      if (gap > bound) violations++
    }
    expect(violations).toBe(0)
    expect(bound).toBeLessThan(0.45)
    expect(worstGap).toBeGreaterThan(0.02)
  })
})

describe('margin bound', () => {
  it('counts the fraction below the margin and adds the complexity term', () => {
    const r = marginBound([0.5, 2, 3, 4], 2, 1.5, 0.05, { gamma: 1 })
    expect(r.fraction).toBe(0.25)
    expect(r.complexity).toBeCloseTo((2 * 2 * 1.5) / (1 * 2) + 3 * Math.sqrt(Math.log(40) / 8), 12)
    expect(r.bound).toBe(1)
  })

  it('holds across draws for a large-margin separator and is informative at moderate n', () => {
    // Two clusters at +-3 on the first axis, standard deviation 1/2, classifier w = (1, 0) (norm 1); features have norm
    // at most R = hypot(5, 2) (about ten standard deviations of slack). The test error is P(N(3, 1/2) < 0) = 1e-9.
    const delta = 0.05
    const n = 4000
    let violations = 0
    let bound = 0
    for (let rep = 0; rep < 60; rep++) {
      const z = toFlat(normal(key(300 + rep), [n * 2]))
      const lab = toFlat(randint(key(400 + rep), [n], 0, 2))
      const x0 = Array.from({ length: n }, (_, i) => (lab[i] === 1 ? 3 : -3) + 0.5 * (z[2 * i] ?? 0))
      const margins = x0.map((v, i) => (lab[i] === 1 ? v : -v))
      const r = marginBound(margins, 1, Math.hypot(5, 2), delta, { gamma: 2 })
      bound = r.bound
      if (1e-9 > r.bound) violations++
    }
    expect(violations).toBe(0)
    expect(bound).toBeLessThan(0.3)
  })
})

describe('algorithmic stability', () => {
  // One-feature ridge for (1/n) sum (w x - y)^2 + lambda w^2 has the closed form w = sum(xy) / (sum(x^2) + n lambda).
  const ridge1d =
    (lambda: number) =>
    (x: Float64Array[], y: Float64Array): ((x: Float64Array) => number) => {
      let sxy = 0
      let sxx = 0
      x.forEach((p, i) => {
        sxy += (p[0] ?? 0) * (y[i] ?? 0)
        sxx += (p[0] ?? 0) ** 2
      })
      const w = sxy / (sxx + x.length * lambda)
      return (p) => w * (p[0] ?? 0)
    }
  const draw = (seed: number, n: number): { x: Float64Array[]; y: Float64Array } => {
    const u = toFlat(uniform(key(seed), [n * 2], { low: -1, high: 1 }))
    return {
      x: Array.from({ length: n }, (_, i) => Float64Array.from([u[2 * i] ?? 0])),
      y: Float64Array.from({ length: n }, (_, i) =>
        Math.max(-1, Math.min(1, 0.8 * (u[2 * i] ?? 0) + 0.2 * (u[2 * i + 1] ?? 0))),
      ),
    }
  }

  it('the closed-form stability of ridge dominates the measured replacement change and scales as 1 / (lambda n)', () => {
    for (const lambda of [0.05, 0.2, 1]) {
      for (const n of [50, 200]) {
        const train = draw(11, n)
        const replacements = Array.from({ length: n }, (_, j) => ({
          x: Float64Array.from([j % 2 === 0 ? 1 : -1]),
          y: j % 2 === 0 ? -1 : 1,
        }))
        const test = draw(12, 100)
        const measured = empiricalStability(ridge1d(lambda), train, replacements, test)
        const beta = ridgeUniformStability(n, lambda, { radius: 1, labelBound: 1 })
        expect(measured).toBeLessThanOrEqual(beta)
      }
    }
    expect(ridgeUniformStability(100, 1, { radius: 1, labelBound: 1 })).toBeCloseTo(16 / 100, 12)
    expect(
      ridgeUniformStability(200, 0.5, { radius: 1, labelBound: 1 }) /
        ridgeUniformStability(100, 0.5, { radius: 1, labelBound: 1 }),
    ).toBeCloseTo(0.5, 12)
  })

  it('measured stability falls with n at the 1 / n rate of the bound', () => {
    const stab = (lambda: number, n: number): number => {
      const train = draw(21, n)
      const replacements = Array.from({ length: n }, (_, j) => ({
        x: Float64Array.from([j % 2 === 0 ? 1 : -1]),
        y: j % 2 === 0 ? -1 : 1,
      }))
      return empiricalStability(ridge1d(lambda), train, replacements, draw(22, 100))
    }
    expect(stab(0.1, 400) / stab(0.1, 50)).toBeLessThan(0.3)
  })

  it('turns stability into a generalisation bound that is informative when beta decays fast', () => {
    expect(stabilityBound(0.001, 1000, 0.05)).toBeCloseTo(0.002 + (4 + 1) * Math.sqrt(Math.log(20) / 2000), 12)
    expect(stabilityBound(1e-6, 1000, 0.05)).toBeLessThan(0.04)
    expect(stabilityBound(0.01, 1000, 0.05)).toBeGreaterThan(1)
  })
})

describe('PAC-Bayes', () => {
  it('inverts the binary KL', () => {
    for (const [q, c] of [
      [0.1, 0.05],
      [0.3, 0.2],
      [0, 0.1],
      [0.5, 1],
    ] as const) {
      const p = klInverse(q, c)
      expect(klBernoulli(q, p)).toBeCloseTo(c, 9)
      expect(p).toBeLessThanOrEqual(q + Math.sqrt(c / 2) + 1e-12)
      const lo = klInverseLower(q, c)
      if (q > 0) expect(klBernoulli(q, lo)).toBeCloseTo(c, 9)
      expect(lo).toBeLessThanOrEqual(q)
    }
    expect(klInverse(0.2, 0)).toBe(0.2)
    expect(klInverse(1, 0.5)).toBe(1)
    expect(klInverse(0.2, 100)).toBe(1)
  })

  it('computes the Gaussian KL in closed form, as the library divergence does', () => {
    const kl = gaussianKl([1, -0.5], [0.7, 1.2], [0.2, 0.1], [1, 2])
    const ref =
      item(klDivergence(normalDist(1, 0.7), normalDist(0.2, 1))) +
      item(klDivergence(normalDist(-0.5, 1.2), normalDist(0.1, 2)))
    expect(kl).toBeCloseTo(ref, 12)
    expect(gaussianKl([1, 0], 1, [0, 0], 1)).toBeCloseTo(0.5, 12)
    expect(gaussianKl([0.3], 0.5, [0.3], 0.5)).toBeCloseTo(0, 12)
    expect(() => gaussianKl([0], 0, [0], 1)).toThrow()
  })

  it('the Gibbs risk closed form matches sampling weights from the posterior', () => {
    const x = [
      [1, 0.5],
      [-0.3, 1],
      [0.8, -0.8],
      [2, 1],
    ]
    const y = [1, -1, 1, 1]
    const mean = [0.7, -0.2]
    const sd = 0.9
    const closed = gaussianGibbsRisk(mean, sd, x, y)
    const w = toFlat(normal(key(8), [20000 * 2]))
    let err = 0
    for (let s = 0; s < 20000; s++) {
      const a = (mean[0] ?? 0) + sd * (w[2 * s] ?? 0)
      const b = (mean[1] ?? 0) + sd * (w[2 * s + 1] ?? 0)
      x.forEach((p, i) => {
        if ((a * (p[0] ?? 0) + b * (p[1] ?? 0) >= 0 ? 1 : -1) !== y[i]) err += 1 / (x.length * 20000)
      })
    }
    expect(err).toBeCloseTo(closed, 2)
  })

  it('orders the bounds, and each holds for a Gaussian posterior on a planted linear problem', () => {
    // Labels y = sign(w* . x + noise) with w* = (1.5, 1.5), x ~ N(0, I_2); posterior N(w_hat, 0.5^2 I), prior N(0, I).
    const n = 2000
    const delta = 0.05
    const sd = 0.5
    let violations = 0
    let last = { mc: 0, seeger: 0, catoni: 0, gibbs: 0 }
    for (let rep = 0; rep < 40; rep++) {
      const gen = (k: number, m: number): { x: number[][]; y: number[] } => {
        const z = toFlat(normal(foldIn(key(rep), k), [m * 3]))
        const x = Array.from({ length: m }, (_, i) => [z[3 * i] ?? 0, z[3 * i + 1] ?? 0])
        return { x, y: x.map((p, i) => (1.5 * (p[0] ?? 0) + 1.5 * (p[1] ?? 0) + (z[3 * i + 2] ?? 0) >= 0 ? 1 : -1)) }
      }
      const train = gen(0, n)
      const fresh = gen(1, 20000)
      // The mean-difference direction scaled to norm 1.
      const sum = train.x.reduce<number[]>(
        (a, p, i) => [(a[0] ?? 0) + (train.y[i] ?? 0) * (p[0] ?? 0), (a[1] ?? 0) + (train.y[i] ?? 0) * (p[1] ?? 0)],
        [0, 0],
      )
      const norm = Math.hypot(sum[0] ?? 0, sum[1] ?? 0)
      const w = [(2.5 * (sum[0] ?? 0)) / norm, (2.5 * (sum[1] ?? 0)) / norm]
      const emp = gaussianGibbsRisk(w, sd, train.x, train.y)
      const truth = gaussianGibbsRisk(w, sd, fresh.x, fresh.y)
      const kl = gaussianKl(w, sd, [0, 0], 1)
      const mc = mcAllesterBound(emp, kl, n, delta)
      const se = seegerBound(emp, kl, n, delta)
      const ca = catoniBound(emp, kl, n, delta)
      expect(se).toBeLessThanOrEqual(mc + 1e-12)
      if (truth > mc || truth > se || truth > ca) violations++
      last = { mc, seeger: se, catoni: ca, gibbs: truth }
    }
    expect(violations).toBe(0)
    // Not vacuous: within 0.1 of the truth, and the Seeger form beats McAllester's.
    expect(last.seeger - last.gibbs).toBeLessThan(0.1)
    expect(last.seeger).toBeLessThan(last.mc)
  })

  it('Catoni with a fixed c matches the formula', () => {
    const r = 0.1
    const kl = 5
    const n = 1000
    const expected = (1 - Math.exp(-1 * r - (kl + Math.log(20)) / n)) / (1 - Math.exp(-1))
    expect(catoniBound(r, kl, n, 0.05, 1)).toBeCloseTo(expected, 12)
    expect(catoniBound(r, kl, n, 0.05)).toBeLessThan(0.25)
  })
})

describe('regret bounds', () => {
  it('Hedge stays under ln N / eta + eta T / 8 on adversarial and random loss sequences', () => {
    const T = 400
    const N = 8
    const eta = Math.sqrt((8 * Math.log(N)) / T)
    for (const mode of ['random', 'alternating', 'best-expert']) {
      const flat = toFlat(uniform(key(77), [T * N]))
      const losses = Array.from({ length: T }, (_, t) =>
        Array.from({ length: N }, (_, i) =>
          mode === 'random'
            ? (flat[t * N + i] ?? 0)
            : mode === 'alternating'
              ? (t + i) % 2 === 0
                ? 1
                : 0
              : i === 3
                ? 0.2 * (flat[t * N + i] ?? 0)
                : (flat[t * N + i] ?? 0),
        ),
      )
      const r = runOnline(hedge({ learningRate: eta }), linearLosses(losses), { dim: N, experts: true })
      const regret = toFlat(r.regret).at(-1) ?? 0
      expect(regret).toBeLessThanOrEqual(hedgeRegretBound(T, N, eta))
    }
    // The tuned rate gives sqrt(T ln N / 2).
    expect(hedgeRegretBound(T, N)).toBeCloseTo(Math.sqrt((T * Math.log(N)) / 2), 12)
  })

  it('online gradient descent stays under D G sqrt(T) against the best fixed point in the ball', () => {
    const T = 500
    const D = 2
    const radius = 1
    const flat = toFlat(normal(key(78), [T * 3]))
    const grads = Array.from({ length: T }, (_, t) => {
      const g = [(flat[3 * t] ?? 0) + 0.5, flat[3 * t + 1] ?? 0, flat[3 * t + 2] ?? 0]
      const nrm = Math.hypot(...g)
      return g.map((v) => v / Math.max(1, nrm))
    })
    const sum = grads.reduce((a, g) => a.map((v, i) => v + (g[i] ?? 0)), [0, 0, 0])
    const nrm = Math.hypot(...sum)
    const comparator = sum.map((v) => (-radius * v) / nrm)
    const r = runOnline(onlineGradientDescent({ learningRate: D / Math.sqrt(T), radius }), linearLosses(grads), {
      dim: 3,
      comparator,
    })
    const regret = toFlat(r.regret).at(-1) ?? 0
    expect(regret).toBeLessThanOrEqual(ogdRegretBound(T, D, 1))
    expect(regret).toBeGreaterThan(0)
    expect(ogdRegretBound(10000, 2, 1)).toBeCloseTo(200, 12)
  })
})

describe('bias-variance', () => {
  it('decomposes squared loss exactly: bias^2 + variance is the mean squared error of the fits against the truth', () => {
    const preds = [
      [1, 2, 0],
      [3, 2, 1],
      [2, 5, 2],
      [0, 1, 1],
    ]
    const truth = [1.5, 2, 0.5]
    const bv = biasVariance(preds, truth, 0.25)
    const direct = preds.reduce((a, row) => a + row.reduce((b, v, j) => b + (v - (truth[j] ?? 0)) ** 2, 0) / 3, 0) / 4
    expect(bv.bias2 + bv.variance).toBeCloseTo(direct, 12)
    expect(bv.expected).toBeCloseTo(direct + 0.25, 12)
    expect(
      biasVariance(
        [
          [1, 2],
          [3, 2],
        ],
        [1, 1],
      ).bias2,
    ).toBeCloseTo(1, 12)
    expect(
      biasVariance(
        [
          [1, 2],
          [3, 2],
        ],
        [1, 1],
      ).variance,
    ).toBeCloseTo(0.5, 12)
  })

  it('Monte Carlo over training sets separates bias from variance: a constant fit is biased, a 1-NN fit is not', () => {
    const f = (x: number): number => Math.sin(2 * Math.PI * x)
    const testX = Array.from({ length: 20 }, (_, i) => Float64Array.from([(i + 0.5) / 20]))
    const truth = testX.map((p) => f(p[0] ?? 0))
    const sigma = 0.3
    const sample = (k: ReturnType<typeof key>): { x: Float64Array[]; y: Float64Array } => {
      const u = toFlat(uniform(foldIn(k, 0), [30]))
      const e = toFlat(normal(foldIn(k, 1), [30]))
      return {
        x: Array.from(u, (v) => Float64Array.from([v])),
        y: Float64Array.from(u, (v, i) => f(v) + sigma * (e[i] ?? 0)),
      }
    }
    const constant = monteCarloBiasVariance({
      sample,
      fit: (_x, y) => {
        const m = y.reduce((a, b) => a + b, 0) / y.length
        return () => m
      },
      testX,
      truth,
      repeats: 200,
      noiseVariance: sigma * sigma,
    })
    const nn = monteCarloBiasVariance({
      sample,
      fit: (x, y) => (p) => {
        let best = 0
        for (let i = 1; i < x.length; i++)
          if (Math.abs((x[i]?.[0] ?? 0) - (p[0] ?? 0)) < Math.abs((x[best]?.[0] ?? 0) - (p[0] ?? 0))) best = i
        return y[best] ?? 0
      },
      testX,
      truth,
      repeats: 200,
      noiseVariance: sigma * sigma,
    })
    expect(constant.bias2).toBeGreaterThan(0.4)
    expect(constant.variance).toBeLessThan(0.05)
    expect(nn.bias2).toBeLessThan(0.02)
    expect(nn.variance).toBeGreaterThan(0.06)
    // The variance of a 1-NN fit is the noise variance plus the interpolation error: about sigma^2 or more.
    expect(nn.variance).toBeGreaterThan(0.9 * sigma * sigma)
  })

  it('the 0-1 decomposition reproduces the expected error computed directly (Domingos 2000)', () => {
    // Four test points, 10 fits each; noise N(x) = P(target != optimal label).
    const optimal = [1, 1, 0, 0]
    const fits = [
      [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
      [1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
      [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    ]
    const preds = Array.from({ length: 10 }, (_, r) => fits.map((col) => col[r] ?? 0))
    const noise = [0.1, 0.2, 0.05, 0.3]
    const d = biasVarianceZeroOne(preds, optimal, noise)
    let direct = 0
    for (let j = 0; j < 4; j++) {
      const pStar = preds.filter((row) => row[j] === optimal[j]).length / 10
      direct += (pStar * (noise[j] ?? 0) + (1 - pStar) * (1 - (noise[j] ?? 0))) / 4
    }
    expect(d.expected).toBeCloseTo(direct, 12)
    expect(d.netVariance).toBeCloseTo(d.unbiasedVariance - d.biasedVariance, 12)
    expect(d.bias).toBeGreaterThan(0)
    expect(d.variance).toBeGreaterThan(0)
    expect(() => biasVarianceZeroOne([[0, 1, 2]], [1, 1, 1])).not.toThrow()
    expect(() => biasVarianceZeroOne([[0], [1], [2]], [1])).toThrow()
  })
})

describe('degrees of freedom', () => {
  it('closed forms', () => {
    expect(ridgeDegreesOfFreedom([2, 1], 1, true)).toBeCloseTo(4 / 5 + 1 / 2, 12)
    expect(
      ridgeDegreesOfFreedom(
        [
          [1, 0],
          [0, 2],
        ],
        0,
      ),
    ).toBeCloseTo(2, 12)
    expect(
      ridgeDegreesOfFreedom(
        [
          [1, 0],
          [0, 2],
        ],
        1e6,
      ),
    ).toBeLessThan(1e-4)
    expect(lassoDegreesOfFreedom([0, 1.5, 0, -2])).toBe(2)
    expect(lassoDegreesOfFreedom([0, 1.5], true)).toBe(2)
    expect(knnDegreesOfFreedom(100, 5)).toBe(20)
    expect(covariancePenalty(1, 5, 50)).toBeCloseTo(0.2, 12)
    expect(mutualInformationBound(10, 1000)).toBeCloseTo(Math.sqrt((2 * 0.25 * 10) / 1000), 12)
  })

  it('Efron covariance estimate agrees with the trace for a ridge smoother', () => {
    // X is 12 x 3, ridge hat matrix S = X (X'X + lambda I)^-1 X'; df = trace S.
    const n = 12
    const p = 3
    const x = rows(toFlat(normal(key(31), [n * p])), n, p)
    const lambda = 2
    // Hat matrix by eigen-free route: solve (X'X + lambda I) B = X' with Gauss-Jordan on a 3 x 3 system.
    const xtx = Array.from({ length: p }, (_, a) =>
      Array.from(
        { length: p },
        (_, b) => x.reduce((s, r) => s + (r[a] ?? 0) * (r[b] ?? 0), 0) + (a === b ? lambda : 0),
      ),
    )
    const inv = invert(xtx)
    const hat = (y: Float64Array): number[] => {
      const xty = Array.from({ length: p }, (_, a) => x.reduce((s, r, i) => s + (r[a] ?? 0) * (y[i] ?? 0), 0))
      const beta = inv.map((row) => row.reduce((s, v, b) => s + v * (xty[b] ?? 0), 0))
      return x.map((r) => r.reduce((s, v, a) => s + v * (beta[a] ?? 0), 0))
    }
    const mu = new Array<number>(n).fill(0)
    const trace = ridgeDegreesOfFreedom(x, lambda)
    const est = effectiveDegreesOfFreedom((y) => hat(y), mu, 1, { draws: 4000, key: key(32) })
    expect(Math.abs(est.estimate - trace)).toBeLessThan(4 * est.standardError)
    expect(est.standardError).toBeLessThan(0.1)
  })

  it('Efron estimate for the lasso on an orthonormal design equals the expected number of nonzeros', () => {
    // With X = I the lasso is soft thresholding at lambda; df = E[#nonzero] (Zou, Hastie and Tibshirani 2007).
    const n = 10
    const mu = [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
    const lambda = 1
    const soft = (y: Float64Array): number[] => Array.from(y, (v) => Math.sign(v) * Math.max(Math.abs(v) - lambda, 0))
    const est = effectiveDegreesOfFreedom(soft, mu, 1, { draws: 6000, key: key(33) })
    let nonzeros = 0
    const draws = 6000
    for (let d = 0; d < draws; d++) {
      const e = toFlat(normal(foldIn(key(34), d), [n]))
      nonzeros += lassoDegreesOfFreedom(soft(Float64Array.from(e, (v, i) => (mu[i] ?? 0) + v))) / draws
    }
    expect(Math.abs(est.estimate - nonzeros)).toBeLessThan(4 * est.standardError + 0.05)
  })

  it('Efron estimate for k-NN smoothing is n / k', () => {
    const n = 20
    const k = 4
    // Neighbours by position on a line: the k nearest including the point itself.
    const knn = (y: Float64Array): number[] =>
      Array.from(y, (_, i) => {
        const idx = Array.from({ length: n }, (_, j) => j)
          .sort((a, b) => Math.abs(a - i) - Math.abs(b - i) || a - b)
          .slice(0, k)
        return idx.reduce((s, j) => s + (y[j] ?? 0), 0) / k
      })
    const est = effectiveDegreesOfFreedom(knn, new Array<number>(n).fill(0), 1, { draws: 3000, key: key(35) })
    expect(Math.abs(est.estimate - knnDegreesOfFreedom(n, k))).toBeLessThan(4 * est.standardError)
  })
})

/** Inverse of a small matrix by Gauss-Jordan (a test helper). */
function invert(a: number[][]): number[][] {
  const n = a.length
  const m = a.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < n; c++) {
    let piv = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]?.[c] ?? 0) > Math.abs(m[piv]?.[c] ?? 0)) piv = r
    ;[m[c], m[piv]] = [m[piv] as number[], m[c] as number[]]
    const p = m[c]?.[c] ?? 1
    m[c] = (m[c] as number[]).map((v) => v / p)
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = m[r]?.[c] ?? 0
      m[r] = (m[r] as number[]).map((v, j) => v - f * (m[c]?.[j] ?? 0))
    }
  }
  return m.map((r) => r.slice(n))
}

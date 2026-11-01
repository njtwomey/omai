import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { logisticRegression } from '../../testing/learn.ts'
import { key, normal, uniform } from '../random/index.ts'
import { svc } from '../svm/index.ts'
import { item, toArray, toFlat } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import { gaussianNaiveBayes } from './naive-bayes.ts'
import {
  ABSTAIN,
  bagData,
  confidentJoint,
  dawidSkene,
  dawidSkeneAlgorithm,
  elkanNoto,
  estimateNoiseTransition,
  findLabelIssues,
  labellingMatrixSummary,
  labelModel,
  labelModelAlgorithm,
  labelProportionClassifier,
  labelQuality,
  majorityVote,
  multipleInstanceSvm,
  puClassifier,
  puRisk,
} from './weak-supervision.ts'

// ---------------------------------------------------------------------------------------------------------------------
// Planted data, with the truth the generator knows

const draws = (seed: number, n: number): Float64Array => Float64Array.from(toFlat(uniform(key(seed), [n])))
const gauss = (seed: number, n: number): Float64Array => Float64Array.from(toFlat(normal(key(seed), [n])))

/** Sample an index from a probability row by inverse CDF. */
function sampleRow(row: readonly number[], u: number): number {
  let acc = 0
  for (let i = 0; i < row.length; i++) {
    acc += row[i] ?? 0
    if (u < acc) return i
  }
  return row.length - 1
}

/** Votes from sources with known confusion matrices `[source][true][vote]`; a source abstains with probability `1 - coverage`. */
function plantedVotes(
  seed: number,
  n: number,
  priors: readonly number[],
  confusions: readonly (readonly (readonly number[])[])[],
  coverage: number,
): { votes: number[][]; truth: number[] } {
  const m = confusions.length
  const u = draws(seed, n * (m + 1) * 2)
  const truth = Array.from({ length: n }, (_, i) => sampleRow(priors, u[i] ?? 0))
  const votes = truth.map((t, i) =>
    confusions.map((conf, j) => {
      const cover = u[n + i * m + j] ?? 0
      if (cover >= coverage) return ABSTAIN
      return sampleRow(conf[t] ?? [], u[n + n * m + i * m + j] ?? 0)
    }),
  )
  return { votes, truth }
}

/** A confusion matrix: accuracy `a` on the diagonal, the rest spread over the other classes by `skew` weights. */
function confusion(k: number, a: number, skew: readonly number[] = []): number[][] {
  return Array.from({ length: k }, (_, c) => {
    const w = Array.from({ length: k }, (_, l) => (l === c ? 0 : (skew[(c + l) % Math.max(skew.length, 1)] ?? 1)))
    const z = w.reduce((s, x) => s + x, 0)
    return w.map((x, l) => (l === c ? a : ((1 - a) * x) / z))
  })
}

/** The labels of the Bayes posterior under the planted priors and confusions: the ceiling for any label model. */
function bayesLabels(
  votes: readonly (readonly number[])[],
  priors: readonly number[],
  confusions: readonly (readonly (readonly number[])[])[],
): number[] {
  return votes.map((row) => {
    const score = priors.map((p, c) =>
      row.reduce((a, v, j) => (v < 0 ? a : a + Math.log(confusions[j]?.[c]?.[v] ?? 1e-12)), Math.log(p)),
    )
    return score.indexOf(Math.max(...score))
  })
}

const accuracyOfLabels = (got: readonly number[], truth: readonly number[]): number =>
  got.reduce((a, g, i) => a + (g === truth[i] ? 1 : 0), 0) / got.length

// ---------------------------------------------------------------------------------------------------------------------

describe('the labelling matrix summary', () => {
  it('reports coverage, conflicts, polarity and agreement', () => {
    const s = labellingMatrixSummary(
      [
        [0, 0],
        [1, -1],
        [0, 1],
        [-1, -1],
      ],
      { truth: [0, 1, 1, 0] },
    )
    expect(s.coverage).toEqual([0.75, 0.5])
    expect(s.overlap).toEqual([2 / 4, 2 / 4])
    expect(s.conflict).toEqual([1 / 4, 1 / 4])
    expect(s.polarity).toEqual([
      [0, 1],
      [0, 1],
    ])
    expect(s.totalCoverage).toBe(0.75)
    expect(s.agreement[0]?.[1]).toBe(0.5)
    expect(s.accuracy).toEqual([2 / 3, 1])
  })
})

describe('majority vote', () => {
  it('counts votes, abstains on ties and on no votes, and weights sources', () => {
    const r = majorityVote([
      [0, 0, 1],
      [1, 0, -1],
      [-1, -1, -1],
      [0, 1, -1],
    ])
    expect(r.labels).toEqual([0, ABSTAIN, ABSTAIN, ABSTAIN])
    expect(r.proba[0]).toEqual([2 / 3, 1 / 3])
    expect(r.proba[2]).toEqual([0.5, 0.5])
    expect(r.covered).toEqual([true, true, false, true])
    expect(majorityVote([[1, 0]], { ties: 'lowest' }).labels).toEqual([0])
    expect(majorityVote([[0, 1, 1]], { weights: [3, 1, 1] }).labels).toEqual([0])
    expect(majorityVote([[1, 0]], { ties: 'random', key: key(1) }).labels[0]).toBeGreaterThanOrEqual(0)
  })

  it('is exact on planted accurate sources: the vote of three 80 percent sources is right about 90 percent of the time', () => {
    const acc = confusion(2, 0.8)
    const { votes, truth } = plantedVotes(3, 4000, [0.5, 0.5], [acc, acc, acc], 1)
    const r = majorityVote(votes)
    // P(correct) = 0.8^3 + 3 * 0.8^2 * 0.2 = 0.896
    expect(accuracyOfLabels(r.labels, truth)).toBeGreaterThan(0.88)
    expect(accuracyOfLabels(r.labels, truth)).toBeLessThan(0.92)
  })
})

describe('Dawid-Skene', () => {
  const k = 3
  const priors = [0.5, 0.3, 0.2]
  const planted = [confusion(k, 0.85, [1, 3]), confusion(k, 0.8, [3, 1]), confusion(k, 0.75), confusion(k, 0.7, [2, 1])]
  const { votes, truth } = plantedVotes(11, 1200, priors, planted, 0.85)

  it('recovers planted confusion matrices and priors, and labels better than the majority', () => {
    const f = dawidSkene({ mode: 'full' }).fit({ x: votes })
    for (let j = 0; j < planted.length; j++)
      for (let c = 0; c < k; c++)
        for (let l = 0; l < k; l++)
          expect(Math.abs((f.confusions[j]?.[c]?.[l] ?? 0) - (planted[j]?.[c]?.[l] ?? 0))).toBeLessThan(0.1)
    for (const [c, p] of priors.entries()) expect(Math.abs((f.priors[c] ?? 0) - p)).toBeLessThan(0.05)
    const mv = majorityVote(votes).labels
    const ds = accuracyOfLabels(f.labels, truth)
    expect(ds).toBeGreaterThan(0.9)
    expect(ds).toBeGreaterThanOrEqual(
      accuracyOfLabels(
        mv.map((v) => Math.max(v, 0)),
        truth,
      ),
    )
  })

  it('finds an adversarial source that majority vote trusts', () => {
    // Four sources: three right 75 percent of the time, one that is wrong 80 percent of the time (a reversed labeller).
    const adversary = confusion(2, 0.2)
    const good = confusion(2, 0.75)
    const data = plantedVotes(5, 1500, [0.5, 0.5], [good, good, good, adversary], 1)
    const f = dawidSkene({ mode: 'full' }).fit({ x: data.votes })
    expect(f.confusions[3]?.[0]?.[0] ?? 1).toBeLessThan(0.35)
    // The best any method could do is the Bayes posterior under the planted confusions; EM should come close to it.
    const best = bayesLabels(data.votes, [0.5, 0.5], [good, good, good, adversary])
    expect(accuracyOfLabels(best, data.truth)).toBeGreaterThan(
      accuracyOfLabels(majorityVote(data.votes, { ties: 'lowest' }).labels, data.truth),
    )
    expect(accuracyOfLabels(f.labels, data.truth)).toBeGreaterThan(accuracyOfLabels(best, data.truth) - 0.03)
  })

  it('one-coin recovers planted accuracies; GLAD orders the sources by ability', () => {
    const accs = [0.95, 0.85, 0.75, 0.62]
    const d = plantedVotes(
      7,
      1500,
      [0.5, 0.5],
      accs.map((a) => confusion(2, a)),
      1,
    )
    const oc = dawidSkene({ mode: 'one-coin' }).fit({ x: d.votes })
    for (const [j, a] of accs.entries()) expect(Math.abs((oc.confusions[j]?.[0]?.[0] ?? 0) - a)).toBeLessThan(0.05)
    const glad = dawidSkene({ mode: 'glad' }).fit({ x: d.votes })
    const order = glad.abilities
      .map((a, j) => [a, j] as const)
      .sort((p, q) => q[0] - p[0])
      .map(([, j]) => j)
    expect(order).toEqual([0, 1, 2, 3])
    expect(accuracyOfLabels(glad.labels, d.truth)).toBeGreaterThan(0.9)
  }, 60_000)

  it('never lowers the log-likelihood from one step to the next, in any mode', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1000 }), fc.constantFrom('full', 'one-coin', 'glad'), (seed, mode) => {
        const accs = [0.9, 0.75, 0.6, 0.8]
        const d = plantedVotes(
          seed,
          150,
          [0.6, 0.4],
          accs.map((a) => confusion(2, a)),
          0.8,
        )
        const rec = recorder()
        runAlgorithm(
          dawidSkeneAlgorithm(d.votes, { mode, classes: null, smoothing: 0, maxIter: 30, tol: 0, mSteps: 3 }),
          {
            maxSteps: 30,
            tracer: rec,
          },
        )
        const ll = metricSeries(rec.run(), 'logLikelihood').values
        for (let i = 1; i < ll.length; i++) expect(ll[i] ?? 0).toBeGreaterThanOrEqual((ll[i - 1] ?? 0) - 1e-9)
      }),
      { numRuns: 12 },
    )
  })

  it('labels new votes with the learned model, round-trips its parameters and rejects a wrong source count', () => {
    const est = dawidSkene({ mode: 'full' })
    const f = est.fit({ x: votes })
    const again = est.revive(JSON.parse(JSON.stringify(f.params)))
    const q = [
      [0, 0, 0, 0],
      [1, 1, 1, 1],
      [2, 2, 2, 2],
    ]
    expect(toFlat(f.predict(q))).toEqual([0, 1, 2])
    expect(toFlat(again.predict(q))).toEqual([0, 1, 2])
    const proba = toArray(f.predictProba(q)) as number[][]
    for (const row of proba) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(() => f.predict([[0, 0]])).toThrow(/sources/)
    expect(f.score({ x: votes, y: truth })).toBeGreaterThan(0.9)
  })
})

describe('the agreement label model', () => {
  it('recovers planted accuracies from agreements alone, for two and three classes', () => {
    for (const k of [2, 3]) {
      const accs = [0.92, 0.8, 0.7, 0.62, 0.85]
      const priors = k === 2 ? [0.5, 0.5] : [0.4, 0.35, 0.25]
      const d = plantedVotes(
        21 + k,
        6000,
        priors,
        accs.map((a) => confusion(k, a)),
        0.7,
      )
      const f = labelModel().fit({ x: d.votes })
      for (const [j, a] of accs.entries()) expect(Math.abs((f.accuracies[j] ?? 0) - a)).toBeLessThan(0.05)
      const post = toArray(f.predictProba(d.votes)) as number[][]
      const labels = post.map((r) => r.indexOf(Math.max(...r)))
      const covered = d.votes.map((v) => v.some((x) => x >= 0))
      const hit = labels.filter((l, i) => covered[i] === true && l === d.truth[i]).length
      expect(hit / covered.filter(Boolean).length).toBeGreaterThan(0.85)
    }
  })

  it('lowers its objective monotonically, and flags a duplicated source by its residual', () => {
    const accs = [0.9, 0.8, 0.7, 0.75]
    const d = plantedVotes(
      33,
      4000,
      [0.5, 0.5],
      accs.map((a) => confusion(2, a)),
      0.8,
    )
    // Source 4 copies source 0 exactly: the conditional-independence assumption fails for that pair.
    const votes = d.votes.map((v) => [...v, v[0] ?? ABSTAIN])
    const rec = recorder()
    runAlgorithm(
      labelModelAlgorithm(votes, { classes: null, ignore: [], classBalance: null, maxIter: 100, tol: 1e-14 }),
      {
        maxSteps: 100,
        tracer: rec,
      },
    )
    const obj = metricSeries(rec.run(), 'objective').values
    for (let i = 1; i < obj.length; i++) expect(obj[i] ?? 0).toBeLessThanOrEqual((obj[i - 1] ?? 0) + 1e-12)
    const f = labelModel().fit({ x: votes })
    const r = f.residuals
    // The largest excess agreement over all pairs is the planted dependency (source 4 copies source 0).
    const pairs = r.flatMap((row, i) => row.map((v, j) => ({ i, j, v })).filter((e) => e.i < e.j))
    const top = pairs.reduce((best, e) => (Math.abs(e.v) > Math.abs(best.v) ? e : best))
    expect([top.i, top.j]).toEqual([0, 4])
    const others = Math.max(...pairs.filter((e) => !(e.i === 0 && e.j === 4)).map((e) => Math.abs(e.v)))
    expect(top.v).toBeGreaterThan(2 * others)
    // Ignoring the known pair recovers the accuracy of the copied source.
    const g = labelModel({ ignore: [[0, 4]] }).fit({ x: votes })
    expect(Math.abs((g.accuracies[0] ?? 0) - 0.9)).toBeLessThan(0.05)
  })
})

// ---------------------------------------------------------------------------------------------------------------------
// Noisy labels: a planted flip matrix and calibrated probabilities of the true class

/** Two or three well-separated Gaussian classes in the plane, with the true posterior and labels flipped by `flip`. */
function plantedNoise(
  seed: number,
  n: number,
  flip: readonly (readonly number[])[],
): {
  x: number[][]
  truth: number[]
  noisy: number[]
  posterior: number[][]
} {
  const k = flip.length
  const centres = Array.from({ length: k }, (_, c) => [
    3 * Math.cos((2 * Math.PI * c) / k),
    3 * Math.sin((2 * Math.PI * c) / k),
  ])
  const u = draws(seed, 2 * n)
  const g = gauss(seed + 1, 2 * n)
  const truth = Array.from({ length: n }, (_, i) => Math.min(k - 1, Math.floor((u[i] ?? 0) * k)))
  const x = truth.map((t, i) => [
    (centres[t]?.[0] ?? 0) + (g[2 * i] ?? 0),
    (centres[t]?.[1] ?? 0) + (g[2 * i + 1] ?? 0),
  ])
  const noisy = truth.map((t, i) => sampleRow(flip[t] ?? [], u[n + i] ?? 0))
  const posterior = x.map((p) => {
    const w = centres.map((c) =>
      Math.exp(-0.5 * ((p[0] ?? 0) - (c[0] ?? 0)) ** 2 - 0.5 * ((p[1] ?? 0) - (c[1] ?? 0)) ** 2),
    )
    const z = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / z)
  })
  return { x, truth, noisy, posterior }
}

describe('the noise transition and confident learning', () => {
  const flip = [
    [0.85, 0.1, 0.05],
    [0.05, 0.85, 0.1],
    [0.1, 0.0, 0.9],
  ]
  const d = plantedNoise(41, 3000, flip)
  // The probabilities a classifier trained on the noisy labels would give: the true posterior pushed through the flips.
  const noisyProba = d.posterior.map((p) =>
    (flip[0] ?? []).map((_, j) => p.reduce((a, pi, i) => a + pi * (flip[i]?.[j] ?? 0), 0)),
  )

  it('estimates the flip matrix from anchors, and from the confident joint', () => {
    const anchor = estimateNoiseTransition(noisyProba, d.noisy, { percentile: 100 })
    const joint = estimateNoiseTransition(noisyProba, d.noisy, { method: 'confident' })
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        expect(Math.abs((anchor[i]?.[j] ?? 0) - (flip[i]?.[j] ?? 0))).toBeLessThan(0.03)
        expect(Math.abs((joint[i]?.[j] ?? 0) - (flip[i]?.[j] ?? 0))).toBeLessThan(0.06)
      }
    for (const row of anchor) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('computes the confident joint by hand on a small example', () => {
    // Labels 0,0,0,1,1 with probabilities for class 0: thresholds are the mean self-confidences.
    const proba = [
      [0.9, 0.1],
      [0.8, 0.2],
      [0.3, 0.7],
      [0.2, 0.8],
      [0.1, 0.9],
    ]
    const r = confidentJoint(proba, [0, 0, 0, 1, 1])
    expect(r.thresholds[0]).toBeCloseTo((0.9 + 0.8 + 0.3) / 3, 12)
    expect(r.thresholds[1]).toBeCloseTo(0.85, 12)
    // Item 2 (given 0, p1 = 0.7 < 0.85 and p0 = 0.3 < 0.667) clears no threshold; item 3 (p1 = 0.8 < 0.85) neither.
    expect(r.guess).toEqual([0, 0, -1, -1, 1])
    expect(r.counts).toEqual([
      [2, 0],
      [0, 1],
    ])
    expect(r.joint[0]?.[0]).toBeCloseTo(0.6, 12)
    expect(r.joint.flat().reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('ranks the planted flips first: precision at k, and each filter flags mostly real errors', () => {
    const wrong = d.noisy.map((l, i) => l !== d.truth[i])
    const quality = labelQuality(d.posterior, d.noisy)
    const ranked = quality
      .map((q, i) => [q, i] as const)
      .sort((a, b) => a[0] - b[0])
      .map(([, i]) => i)
    const top = ranked.slice(0, 200)
    expect(top.filter((i) => wrong[i]).length / top.length).toBeGreaterThan(0.95)
    for (const filter of [
      'prune_by_noise_rate',
      'prune_by_class',
      'both',
      'confident_joint',
      'low_self_confidence',
      'low_normalized_margin',
    ] as const) {
      const issues = findLabelIssues(d.posterior, d.noisy, filter)
      const precision = issues.filter((i) => wrong[i]).length / issues.length
      const recall = issues.filter((i) => wrong[i]).length / wrong.filter(Boolean).length
      expect(precision, filter).toBeGreaterThan(0.85)
      expect(recall, filter).toBeGreaterThan(filter === 'both' ? 0.5 : 0.75)
      expect(new Set(issues).size).toBe(issues.length)
      const q = issues.map((i) => d.posterior[i]?.[d.noisy[i] ?? 0] ?? 0)
      for (let i = 1; i < q.length; i++) expect(q[i] ?? 0).toBeGreaterThanOrEqual((q[i - 1] ?? 0) - 1e-15)
    }
  })

  it('finds nothing in clean labels', () => {
    const clean = plantedNoise(43, 1500, [
      [1, 0],
      [0, 1],
    ])
    expect(findLabelIssues(clean.posterior, clean.truth, 'prune_by_noise_rate').length).toBeLessThan(5)
  })
})

// ---------------------------------------------------------------------------------------------------------------------
// Positive-unlabelled

describe('positive-unlabelled risks', () => {
  it('computes the unbiased and non-negative risks by hand', () => {
    // Squared loss l(m) = (1 - m)^2 / 4. Scores 2, 2, -2, -2; only the first row is a labelled positive; prior 0.5.
    const u = puRisk([2, 2, -2, -2], [1, 0, 0, 0], 0.5, { loss: 'squared' })
    // pi E_P l(f,+1) = 0.5 * 0.25; E_U l(f,-1) = (2.25 + 0.25 + 0.25) / 3; pi E_P l(f,-1) = 0.5 * 2.25.
    expect(item(u.positiveRisk)).toBeCloseTo(0.125, 12)
    expect(item(u.negativeRisk)).toBeCloseTo(2.75 / 3 - 1.125, 12)
    expect(item(u.risk)).toBeCloseTo(0.125 + 2.75 / 3 - 1.125, 12)
    expect(item(puRisk([2, 2, -2, -2], [1, 0, 0, 0], 0.5, { loss: 'squared', nonNegative: true }).risk)).toBeCloseTo(
      0.125,
      12,
    )
  })

  it('the non-negative risk is never below the unbiased one, and equals it when the negatives risk is positive', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -4, max: 4, noNaN: true }), { minLength: 6, maxLength: 20 }),
        fc.double({ min: 0.05, max: 0.95, noNaN: true }),
        fc.constantFrom('logistic', 'sigmoid', 'squared'),
        (scores, prior, loss) => {
          const lab = scores.map((_, i) => (i % 3 === 0 ? 1 : 0))
          const a = puRisk(scores, lab, prior, { loss })
          const b = puRisk(scores, lab, prior, { loss, nonNegative: true })
          expect(item(b.risk)).toBeGreaterThanOrEqual(item(a.risk) - 1e-12)
          if (item(a.negativeRisk) >= 0) expect(item(b.risk)).toBeCloseTo(item(a.risk), 12)
        },
      ),
      { numRuns: 60 },
    )
  })

  it('trains a linear classifier from positives and unlabelled rows, with the prior known', () => {
    // Positives N(2, 1), negatives N(-2, 1); a third of the unlabelled rows are positive; 150 positives labelled.
    const n = 600
    const g = gauss(51, n)
    const u = draws(52, n)
    const isPos = Array.from({ length: n }, (_, i) => (u[i] ?? 0) < 0.4)
    const x = isPos.map((p, i) => [(p ? 2 : -2) + (g[i] ?? 0)])
    let labelled = 0
    const y = isPos.map((p) => (p && labelled++ < 100 ? 1 : 0))
    // The unlabelled rows are everything but the 100 labelled positives, so their positive fraction is:
    const prior = (isPos.filter(Boolean).length - 100) / (n - 100)
    const f = puClassifier({ prior, nonNegative: true }).fit({ x, y })
    const got = toFlat(f.predict(x))
    expect(accuracyOfLabels(Array.from(got), isPos.map(Number))).toBeGreaterThan(0.93)
    expect(f.params.negativeRisk).toBeGreaterThan(-0.05)
  })

  it('Elkan-Noto estimates the label frequency and the class prior under selection completely at random', () => {
    const n = 1500
    const g = gauss(61, n)
    const u = draws(62, 2 * n)
    const pos = Array.from({ length: n }, (_, i) => (u[i] ?? 0) < 0.4)
    const x = pos.map((p, i) => [(p ? 3 : -3) + (g[i] ?? 0)])
    // Each positive is labelled with probability c = 0.3.
    const y = pos.map((p, i) => (p && (u[n + i] ?? 0) < 0.3 ? 1 : 0))
    const f = elkanNoto(logisticRegression({ C: 10 }), { holdout: 0.3 }).fit({ x, y })
    expect(Math.abs(f.labelFrequency - 0.3)).toBeLessThan(0.06)
    expect(Math.abs(f.prior - 0.4)).toBeLessThan(0.08)
    expect(accuracyOfLabels(Array.from(toFlat(f.predict(x))), pos.map(Number))).toBeGreaterThan(0.85)
  })
})

// ---------------------------------------------------------------------------------------------------------------------
// Bags

/** Two Gaussian classes, grouped into bags of a given size with the bag proportions varied; the per-row labels are the truth. */
function plantedBags(
  seed: number,
  bags: number,
  size: number,
): { x: number[][]; label: number[]; bag: number[]; proportions: number[][] } {
  const u = draws(seed, bags * size + bags)
  const g = gauss(seed + 1, bags * size)
  const x: number[][] = []
  const label: number[] = []
  const bag: number[] = []
  const proportions: number[][] = []
  for (let b = 0; b < bags; b++) {
    const p = 0.1 + 0.8 * (u[bags * size + b] ?? 0)
    let positives = 0
    for (let i = 0; i < size; i++) {
      const pos = (u[b * size + i] ?? 0) < p
      positives += pos ? 1 : 0
      label.push(pos ? 1 : 0)
      bag.push(b)
      x.push([(pos ? 1.8 : -1.8) + (g[b * size + i] ?? 0)])
    }
    proportions.push([1 - positives / size, positives / size])
  }
  return { x, label, bag, proportions }
}

describe('learning from label proportions', () => {
  it('recovers the labels of rows from the proportions of small bags, and its loss falls', () => {
    const d = plantedBags(71, 120, 6)
    const est = labelProportionClassifier({ l2: 1e-3 })
    const f = est.fit(bagData(d.x, d.bag, { proportions: d.proportions }))
    expect(accuracyOfLabels(Array.from(toFlat(f.predict(d.x))), d.label)).toBeGreaterThan(0.9)
    // The proportion loss at the solution is below its value at the zero model (the entropy-like constant of uniform predictions).
    const s = est.session(bagData(d.x, d.bag, { proportions: d.proportions }), { mode: 'eager' })
    const loss = metricSeries(s.run(), 'solver/loss').values
    expect(loss.at(-1)).toBeLessThan(loss[0] ?? 0)
    expect(toArray(f.predictProba([[-3], [3]])) as number[][]).toHaveLength(2)
  })

  it('learns less from large bags than from small ones', () => {
    const small = plantedBags(73, 200, 4)
    const large = plantedBags(73, 12, 66)
    const acc = (d: typeof small): number => {
      const f = labelProportionClassifier({ l2: 1e-3 }).fit(bagData(d.x, d.bag, { proportions: d.proportions }))
      return accuracyOfLabels(Array.from(toFlat(f.predict(d.x))), d.label)
    }
    expect(acc(small)).toBeGreaterThan(acc(large) - 0.02)
  })
})

describe('multiple-instance learning', () => {
  /** Negative bags hold only negatives; positive bags hold one or two positives among negatives. */
  function plantedMil(
    seed: number,
    bags: number,
  ): { x: number[][]; instance: number[]; bag: number[]; bagLabel: number[] } {
    const u = draws(seed, bags * 6)
    const g = gauss(seed + 1, bags * 4)
    const x: number[][] = []
    const instance: number[] = []
    const bag: number[] = []
    const bagLabel: number[] = []
    for (let b = 0; b < bags; b++) {
      const positive = b % 2 === 1
      for (let i = 0; i < 4; i++) {
        const pos = positive && i < 1 + Math.floor((u[b] ?? 0) * 2)
        x.push([(pos ? 3 : -2) + (g[b * 4 + i] ?? 0) * 0.7])
        instance.push(pos ? 1 : 0)
        bag.push(b)
        bagLabel.push(positive ? 1 : 0)
      }
    }
    return { x, instance, bag, bagLabel }
  }

  for (const mode of ['instance', 'bag'] as const) {
    it(`finds the positive instances of positive bags (${mode} mode), with a naive Bayes and with a linear SVM`, () => {
      const d = plantedMil(81, 40)
      for (const base of [gaussianNaiveBayes(), svc.with({ kernel: 'linear', C: 10 })]) {
        const f = multipleInstanceSvm(base, { mode }).fit(bagData(d.x, d.bag, { y: d.bagLabel }))
        const got = Array.from(toFlat(f.predict(d.x)))
        const inPositive = d.bag.map((b) => d.bagLabel[d.bag.indexOf(b)] === 1)
        const hit = got.filter((g, i) => inPositive[i] === true && g === d.instance[i]).length
        expect(hit / inPositive.filter(Boolean).length).toBeGreaterThan(0.9)
        const bagPred = Array.from(toFlat(f.predictBags(d.x, d.bag)))
        expect(bagPred).toEqual(Array.from({ length: 40 }, (_, b) => b % 2))
        expect(f.params.iter).toBeGreaterThanOrEqual(1)
      }
    })
  }

  it('rejects a bag whose instances carry different bag labels, and data without bag ids', () => {
    expect(() =>
      multipleInstanceSvm(gaussianNaiveBayes()).fit(bagData([[0], [1], [2], [3]], [0, 0, 1, 1], { y: [0, 1, 1, 1] })),
    ).toThrow(/bag 0/)
    expect(() => multipleInstanceSvm(gaussianNaiveBayes()).fit({ x: [[0], [1]], y: [0, 1] })).toThrow(/bag ids/)
  })
})

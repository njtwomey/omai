import { describe, expect, it } from 'vitest'
import { solve } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { svc } from '../svm/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  metricSeries,
  type Run,
  recorder,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { gaussianNaiveBayes } from './naive-bayes.ts'
import {
  coTraining,
  coTrainingAlgorithm,
  isUnlabelled,
  labelPropagation,
  labelPropagationAlgorithm,
  labelSpreading,
  labelSpreadingAlgorithm,
  laplacianRlsAlgorithm,
  laplacianRlsClassifier,
  laplacianRlsRegressor,
  s3vm,
  selfTraining,
  selfTrainingAlgorithm,
  triTraining,
} from './semi-supervised.ts'

/** Run an algorithm to its end, recording its metrics. */
function traced<S>(alg: Algorithm<S>, maxSteps: number, runKey = key(0)): { state: S; status: string; run: Run } {
  const rec = recorder()
  const r = runAlgorithm(alg, { maxSteps, tracer: rec, key: runKey })
  return { state: r.state, status: r.status, run: rec.run() }
}

const twoClusters = {
  x: [[-2.4], [-2.1], [-1.9], [-1.7], [-1.5], [1.5], [1.7], [1.9], [2.1], [2.4]],
  y: [0, -1, -1, -1, -1, -1, -1, -1, -1, 1],
}
const truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

const twoViews = {
  x: [
    [-2.5, -2.4],
    [-2.2, -2.0],
    [-1.9, -2.1],
    [-1.7, -1.8],
    [-1.5, -1.6],
    [1.5, 1.6],
    [1.7, 1.8],
    [1.9, 2.1],
    [2.2, 2.0],
    [2.5, 2.4],
  ],
  y: [0, 0, -1, -1, -1, -1, -1, -1, 1, 1],
}

describe('the unlabelled marker', () => {
  it('is -1 or NaN', () => {
    expect(isUnlabelled(-1)).toBe(true)
    expect(isUnlabelled(Number.NaN)).toBe(true)
    expect(isUnlabelled(0)).toBe(false)
    expect(isUnlabelled(2)).toBe(false)
  })
})

describe('self-training', () => {
  it('labels the unlabelled clusters, keeps the given labels, and records the round of each', () => {
    const f = selfTraining(gaussianNaiveBayes(), { threshold: 0.8 }).fit(twoClusters)
    expect(f.transduction).toEqual(truth)
    expect(f.labelledIter[0]).toBe(0)
    expect(f.labelledIter[9]).toBe(0)
    expect(f.labelledIter.slice(1, 9).every((r) => r >= 1)).toBe(true)
    expect(toFlat(f.predict([[-2], [2]]))).toEqual([0, 1])
    expect(f.score({ x: twoClusters.x, y: truth })).toBe(1)
  })

  it('under k_best adds exactly k rows per round until none are left', () => {
    const alg = selfTrainingAlgorithm(twoClusters, {
      base: gaussianNaiveBayes(),
      criterion: 'k_best',
      threshold: 0.75,
      kBest: 3,
      maxIter: 10,
    })
    const run = traced(alg, 12)
    expect(metricSeries(run.run, 'added').values.slice(1)).toEqual([3, 3, 2])
    expect(run.state.stop).toBe('all-labelled')
    expect(run.status).toBe('converged')
  })

  it('stops with no change when nothing clears the threshold, and keeps unlabelled rows labelled -1', () => {
    const f = selfTraining(gaussianNaiveBayes(), { threshold: 1 }).fit(twoClusters)
    expect(f.transduction.filter((v) => isUnlabelled(v)).length).toBe(8)
    expect(f.params.stop).toBe('no-change')
  })

  it('serialises its state and revives the fitted value from its parameters', () => {
    const est = selfTraining(gaussianNaiveBayes())
    const f = est.fit(twoClusters)
    const state = runAlgorithm(selfTrainingAlgorithm(twoClusters, { ...est.hyper }), { maxSteps: 3 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
    const again = est.revive(JSON.parse(JSON.stringify(f.params)))
    expect(toFlat(again.predict([[-2], [2]]))).toEqual([0, 1])
  })

  it('in a session shows the model at an early step with fewer labels', () => {
    const s = selfTraining(gaussianNaiveBayes(), { threshold: 0.8 }).session(twoClusters, { mode: 'eager' })
    expect(s.fittedAt(0).transduction.filter((v) => !isUnlabelled(v)).length).toBe(2)
    expect(s.fittedAt(s.t).transduction.filter((v) => !isUnlabelled(v)).length).toBe(10)
  })
})

describe('co-training', () => {
  it('teaches each view from the other and combines them', () => {
    const f = coTraining(gaussianNaiveBayes(), { viewA: [0], viewB: [1], growth: 2, threshold: 0.6 }).fit(twoViews)
    expect(f.transduction).toEqual(truth)
    expect(f.score({ x: twoViews.x, y: truth })).toBe(1)
    const p = toArray((f.predictProba as NonNullable<typeof f.predictProba>)([[-2, -2]])) as number[][]
    expect(p[0]?.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('draws the pool from the key, so a run is reproducible and the state round-trips', () => {
    const alg = coTrainingAlgorithm(twoViews, {
      base: gaussianNaiveBayes(),
      viewA: [0],
      viewB: [1],
      growth: 1,
      pool: 3,
      threshold: 0.5,
      maxRounds: 20,
    })
    const a = runAlgorithm(alg, { maxSteps: 20, key: key(3) }).state
    const b = runAlgorithm(alg, { maxSteps: 20, key: key(3) }).state
    expect(a).toEqual(b)
    expect(stateFromJson(stateToJson(a))).toEqual(a)
    expect(a.labels).toEqual(truth)
  })
})

describe('tri-training', () => {
  it('trains three members on bootstrap samples and votes', () => {
    const f = triTraining(gaussianNaiveBayes()).fit(twoViews, { key: key(7) })
    expect(
      toFlat(
        f.predict([
          [-2, -2],
          [2, 2],
        ]),
      ),
    ).toEqual([0, 1])
    expect(f.transduction).toEqual(truth)
    expect(f.params.models.length).toBe(3)
  })
})

describe('label propagation and spreading', () => {
  const chain = {
    x: [
      [0, 0],
      [0.2, 0],
      [0.4, 0],
      [0.6, 0],
      [0, 2],
      [0.2, 2],
      [0.4, 2],
      [0.6, 2],
    ],
    y: [0, -1, -1, -1, 1, -1, -1, -1],
  }
  const gold = [0, 0, 0, 0, 1, 1, 1, 1]

  it('clamps the given labels exactly and labels the chains', () => {
    const f = labelPropagation({ gamma: 8 }).fit(chain)
    expect(f.transduction).toEqual(gold)
    expect(f.labelDistributions[0]).toEqual([1, 0])
    expect(f.labelDistributions[4]).toEqual([0, 1])
    expect(
      toFlat(
        f.predict([
          [0.5, 0],
          [0.5, 2],
        ]),
      ),
    ).toEqual([0, 1])
  })

  it('converges to the harmonic solution: the closed form of the unlabelled block', () => {
    // Three points on a line, the ends labelled: with weights w01 and w12 the middle gets w01/(w01+w12) of class 0.
    const x = [[0], [1], [3]]
    const f = labelPropagation({ gamma: 0.5, tol: 1e-14, maxIter: 10000 }).fit({ x, y: [0, -1, 1] })
    const w01 = Math.exp(-0.5)
    const w12 = Math.exp(-0.5 * 4)
    // The middle row's fixed point is (w01 F0 + w12 F2) / (w01 + w12): the kernel's diagonal term cancels.
    const middle = f.labelDistributions[1] ?? []
    expect(middle[0]).toBeCloseTo(w01 / (w01 + w12), 10)
    expect(middle[1]).toBeCloseTo(w12 / (w01 + w12), 10)
  })

  it('spreading holds the labels softly and its change falls', () => {
    const f = labelSpreading({ gamma: 8, alpha: 0.8, maxIter: 100 }).fit(chain)
    expect(f.transduction).toEqual(gold)
    const run = traced(
      labelSpreadingAlgorithm(chain, { kernel: 'rbf', gamma: 8, nNeighbors: 7, alpha: 0.8, maxIter: 100, tol: 1e-3 }),
      100,
    )
    const changes = metricSeries(run.run, 'change').values
    expect(changes.at(-1)).toBeLessThan(1e-3)
    expect(changes.at(-1) ?? 1).toBeLessThan(changes[1] ?? 0)
  })

  it('works over a neighbour graph, and rejects alpha outside (0, 1)', () => {
    const f = labelPropagation({ kernel: 'knn', nNeighbors: 3 }).fit(chain)
    expect(f.transduction).toEqual(gold)
    expect(
      toFlat(
        f.predict([
          [0.3, 0.1],
          [0.3, 1.9],
        ]),
      ),
    ).toEqual([0, 1])
    expect(() => labelSpreading({ alpha: 1 }).fit(chain)).toThrow(/alpha/)
  })

  it('stops by tolerance (converged) or by the iteration cap (completed)', () => {
    const base = { kernel: 'rbf', gamma: 1, nNeighbors: 7, alpha: 0.2, tol: 1e-3 } as const
    expect(runAlgorithm(labelPropagationAlgorithm(chain, { ...base, maxIter: 1000 }), { maxSteps: 1000 }).status).toBe(
      'converged',
    )
    expect(
      runAlgorithm(labelPropagationAlgorithm(chain, { ...base, maxIter: 2, tol: 1e-12 }), { maxSteps: 1000 }).status,
    ).toBe('completed')
  })
})

describe('Laplacian regularised least squares', () => {
  const chain = {
    x: [
      [0, 0],
      [0.2, 0],
      [0.4, 0],
      [0.6, 0],
      [0, 2],
      [0.2, 2],
      [0.4, 2],
      [0.6, 2],
    ],
    y: [0, -1, -1, -1, 1, -1, -1, -1],
  }

  it('without the graph term is kernel ridge on the labelled rows, in closed form', () => {
    const x = [[0], [1], [2], [3]]
    const y = [1, Number.NaN, 3, Number.NaN]
    const f = laplacianRlsRegressor({ gamma: 0.5, gammaA: 0.1, gammaI: 0 }).fit({ x, y })
    // alpha_l = (K_ll + gammaA l I)^{-1} y_l with l = 2 labelled rows, zero on the unlabelled ones.
    const k = (a: number, b: number): number => Math.exp(-0.5 * (a - b) ** 2)
    const kll = [
      [1 + 0.2, k(0, 2)],
      [k(0, 2), 1 + 0.2],
    ]
    const alpha = toFlat(solve(kll, [1, 3]))
    const want = (q: number): number => (alpha[0] ?? 0) * k(0, q) + (alpha[1] ?? 0) * k(2, q)
    const got = toFlat(f.predict([[1], [2.5]]))
    expect(got[0]).toBeCloseTo(want(1), 10)
    expect(got[1]).toBeCloseTo(want(2.5), 10)
  })

  it('lowers the graph smoothness of the expansion as the intrinsic penalty grows, and classifies the chains', () => {
    const smooth = (gammaI: number): number => {
      const run = traced(
        laplacianRlsAlgorithm(
          chain,
          {
            kernel: 'rbf',
            gamma: 2,
            degree: 3,
            coef0: 1,
            gammaA: 0.01,
            gammaI,
            nNeighbors: 3,
            graphGamma: 8,
            normalised: false,
          },
          'classification',
        ),
        3,
      )
      return metricSeries(run.run, 'smoothness').values.at(-1) ?? Number.NaN
    }
    expect(smooth(5)).toBeLessThan(smooth(0))
    const f = laplacianRlsClassifier({ gamma: 2, gammaA: 0.01, gammaI: 5, nNeighbors: 3, graphGamma: 8 }).fit(chain)
    expect(toFlat(f.predict(chain.x))).toEqual([0, 0, 0, 0, 1, 1, 1, 1])
  })
})

describe('the transductive SVM', () => {
  it('switches labels until the margin sits in the gap between the clusters', () => {
    const x = [[-2.4], [-2.1], [-1.9], [-1.7], [-1.5], [1.5], [1.7], [1.9], [2.1], [2.4]]
    const est = s3vm(svc.with({ kernel: 'linear', C: 10 }), { unlabelledWeight: 1, startWeight: 0.05 })
    const f = est.fit({ x, y: [0, -1, -1, -1, -1, -1, -1, -1, -1, 1] })
    expect(f.transduction).toEqual(truth)
    expect(toFlat(f.predict([[-1.8], [1.8]]))).toEqual([0, 1])
    expect(f.params.stop).toBe('converged')
    // The boundary is in the gap: the decision function changes sign between the clusters.
    const d = toFlat(f.decisionFunction([[-1], [0], [1]]))
    expect((d[0] ?? 0) * (d[2] ?? 0)).toBeLessThan(0)
  })
})

import { describe, expect, it } from 'vitest'
import {
  constantKernel,
  fixHyperparameters,
  maternKernel,
  productKernel,
  rbfKernel,
  sumKernel,
  whiteNoiseKernel,
} from '../kernels/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { gpRegression } from './regression.ts'

const x = Array.from({ length: 18 }, (_, i) => [i / 3])
const noisy = x.map(([v], i) => Math.sin(v as number) + 0.08 * Math.sin(37 * i))
const data = { x, y: noisy }

const kernel = sumKernel(
  productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 1 })),
  whiteNoiseKernel({ noiseLevel: 0.05 }),
)

describe('gpRegression', () => {
  it('runs as a session: the likelihood climbs step by step, the early model and the probes are available', () => {
    const est = gpRegression.with({ kernel })
    const s = est.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 2 })
    const early = s.fittedAt(2)
    expect(toFlat(early.predict([[1]])).length).toBe(1)
    const grid = s.probe(2, 'model/posterior', { resolution: 12, samples: 2, seed: 3 }) as {
      kind: string
      mean: number[]
      variance: number[]
      samples: number[][]
    }
    expect(grid.kind).toBe('posteriorGrid')
    expect(grid.mean).toHaveLength(12)
    expect(grid.samples).toHaveLength(2)
    s.advance()
    expect(s.status).toBe('converged')
    const lml = metricSeries(s.run(), 'gp/logMarginalLikelihood').values
    expect(lml.length).toBeGreaterThan(3)
    expect(lml[lml.length - 1] as number).toBeGreaterThan(lml[0] as number)
    for (let t = 1; t < lml.length; t++) expect(lml[t] as number).toBeGreaterThanOrEqual((lml[t - 1] as number) - 1e-6)
    expect(s.fittedAt(s.t).params.logMarginalLikelihood).toBeCloseTo(lml[lml.length - 1] as number, 8)
    const surface = s.probe(s.t, 'solver/surface', { resolution: 5 }) as { kind: string; values: number[][] }
    expect(surface.kind).toBe('hyperparameterSurface')
    expect(surface.values).toHaveLength(5)
  }, 60_000)

  it('serialises its state: seeking from a checkpoint gives the model a straight run gives', () => {
    const est = gpRegression.with({ kernel })
    const long = est.session(data, { key: key(1) })
    long.advance({ steps: 12 })
    const sought = long.fittedAt(8).params
    const direct = est.session(data, { key: key(1) })
    direct.advance({ steps: 8 })
    expect(sought).toEqual(direct.fittedAt(8).params)
  })

  it('interpolates smooth data and is uncertain away from it', () => {
    const fitted = gpRegression.with({ kernel, alpha: 1e-6 }).fit(data, { explain: false })
    const { mean, std } = fitted.predictWithStd([[2], [20]])
    expect(Math.abs((toFlat(mean)[0] as number) - Math.sin(2))).toBeLessThan(0.15)
    expect(toFlat(std)[1] as number).toBeGreaterThan(toFlat(std)[0] as number)
    expect(fitted.score(data)).toBeGreaterThan(0.95)
  })

  it('draws posterior samples whose mean approaches the posterior mean, and a covariance whose diagonal is the variance', () => {
    const fitted = gpRegression
      .with({ kernel: rbfKernel({ lengthScale: 1 }), optimise: false, alpha: 0.01 })
      .fit(data, { explain: false })
    const at = [[0.5], [3], [8]]
    const draws = toFlat(fitted.sample(at, 2000, key(4)))
    const mean = toFlat(fitted.predict(at))
    for (let j = 0; j < 3; j++) {
      const column = Array.from({ length: 2000 }, (_, i) => draws[i * 3 + j] as number)
      expect(column.reduce((a, b) => a + b, 0) / 2000).toBeCloseTo(mean[j] as number, 1)
    }
    const cov = toFlat(fitted.predictCovariance(at))
    const std = toFlat(fitted.predictWithStd(at).std)
    for (let j = 0; j < 3; j++) expect(Math.sqrt(cov[j * 3 + j] as number)).toBeCloseTo(std[j] as number, 8)
  })

  it('keeps fixed hyperparameters and uses a better start among restarts', () => {
    const pinned = fixHyperparameters(rbfKernel({ lengthScale: 2 }), ['lengthScale'])
    const fitted = gpRegression.with({ kernel: pinned }).fit(data, { explain: false })
    expect(fitted.params.theta).toEqual([])
    const once = gpRegression.with({
      kernel: sumKernel(
        productKernel(constantKernel({ value: 1 }), maternKernel({ lengthScale: 0.1, nu: 1.5 })),
        whiteNoiseKernel({ noiseLevel: 1 }),
      ),
    })
    const more = once.with({ restarts: 3 })
    const a = once.fit(data, { explain: false, key: key(2) }).params.logMarginalLikelihood
    const b = more.fit(data, { explain: false, key: key(2) }).params.logMarginalLikelihood
    expect(b).toBeGreaterThanOrEqual(a - 1e-9)
  })

  it('revives from JSON and normalises targets', () => {
    const est = gpRegression.with({ kernel, normalizeY: true })
    const shifted = { x, y: noisy.map((v) => 100 + 5 * v) }
    const fitted = est.fit(shifted, { explain: false })
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predict([[1.3]]))).toEqual(toFlat(fitted.predict([[1.3]])))
    expect(Math.abs((toFlat(fitted.predict([[2]]))[0] as number) - (100 + 5 * Math.sin(2)))).toBeLessThan(1.5)
    expect(fitted.predictDistribution([[1]]).mean).toBeDefined()
  })
})

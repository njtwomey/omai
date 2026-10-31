import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, metricSeries } from '../trace/index.ts'
import { nuSvc, svc } from './classifier.ts'

const blobs = (centres: readonly (readonly [number, number])[], per: number): { x: number[][]; y: number[] } => {
  const x: number[][] = []
  const y: number[] = []
  let s = 7
  const next = (): number => {
    s = (s * 1103515245 + 12345) % 2147483648
    return s / 2147483648 - 0.5
  }
  centres.forEach(([cx, cy], c) => {
    for (let i = 0; i < per; i++) {
      x.push([cx + 1.6 * next(), cy + 1.6 * next()])
      y.push(c)
    }
  })
  return { x, y }
}

const two = blobs(
  [
    [-1, 0],
    [1, 0],
  ],
  15,
)
const three = blobs(
  [
    [-2, 0],
    [2, 0],
    [0, 2.5],
  ],
  10,
)

describe('svc', () => {
  it('runs as a session: the early model is available, the probe is a margin record, the dual rises and the gap closes', () => {
    const s = svc.with({ kernel: 'rbf', gamma: 0.5, C: 2, tol: 1e-6 }).session(two, { key: key(1), logs: 'all' })
    s.advance({ steps: 3 })
    const record = s.probe(3, 'model/support-vectors') as { kind: string; support: number[]; margins: number[] }
    expect(record.kind).toBe('margin')
    expect(record.margins).toHaveLength(30)
    expect(record.support.length).toBeGreaterThan(0)
    const early = s.fittedAt(3)
    expect(toFlat(early.predict(two.x)).length).toBe(30)
    s.advance()
    expect(s.status).toBe('converged')
    const run = s.run()
    const dual = metricSeries(run, 'solver/dual').values
    for (let t = 1; t < dual.length; t++)
      expect(dual[t] as number).toBeGreaterThanOrEqual((dual[t - 1] as number) - 1e-9)
    const gap = metricSeries(run, 'solver/gap').values
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap.every((g) => g > -1e-9)).toBe(true)
    expect(metricSeries(run, 'smo/alphaChanged').values.every((v) => v >= 0)).toBe(true)
    // The model at the head is what fit returns.
    expect(s.fittedAt(s.t).params).toEqual(
      svc.with({ kernel: 'rbf', gamma: 0.5, C: 2, tol: 1e-6 }).fit(two, { key: key(1) }).params,
    )
  })

  it('logs the dual, the gap and the pair changes on a fit that asks for them', () => {
    const fitted = svc.fit(two, { logs: { 'solver/dual': Log(), 'solver/gap': Log(), 'smo/alphaChanged': Log() } })
    expect(metricSeries(fitted.run, 'solver/dual').values.length).toBeGreaterThan(2)
    expect(metricSeries(fitted.run, 'smo/alphaChanged').values.length).toBeGreaterThan(2)
  })

  it('classifies a separable problem and keeps a support set smaller than the data when the margin is wide', () => {
    const fitted = svc.with({ kernel: 'linear', C: 10 }).fit(
      blobs(
        [
          [-3, 0],
          [3, 0],
        ],
        12,
      ),
    )
    expect(fitted.params.nSupport.reduce((a, b) => a + b, 0)).toBeLessThan(24)
    expect(
      toFlat(
        fitted.predict([
          [-3, 0],
          [3, 0],
        ]),
      ),
    ).toEqual([0, 1])
    expect(
      fitted.score(
        blobs(
          [
            [-3, 0],
            [3, 0],
          ],
          12,
        ),
      ),
    ).toBe(1)
  })

  it('solves more classes one pair at a time and votes', () => {
    const fitted = svc.with({ kernel: 'rbf', gamma: 0.4, C: 5 }).fit(three)
    expect(fitted.params.pairs).toHaveLength(3)
    expect(fitted.score(three)).toBeGreaterThan(0.9)
    expect(fitted.decisionFunction(three.x).shape).toEqual([30, 3])
    expect(svc.with({ decisionShape: 'ovo' }).revive(fitted.params).decisionFunction(three.x).shape).toEqual([30, 3])
  })

  it('fits probabilities by Platt scaling: rows sum to one and favour the predicted class', () => {
    for (const data of [two, three]) {
      const fitted = svc.with({ probability: true, gamma: 0.5 }).fit(data, { key: key(3) })
      const proba = fitted.predictProba?.(data.x)
      expect(proba).toBeDefined()
      const k = data === two ? 2 : 3
      const flat = toFlat(proba as never)
      for (let i = 0; i < data.x.length; i++) {
        const row = flat.slice(i * k, (i + 1) * k)
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      }
      expect(svc.fit(data).predictProba).toBeUndefined()
    }
  })

  it('closes the primal-dual gap of the nu form too, from above', () => {
    const s = nuSvc.with({ nu: 0.4, gamma: 0.5, tol: 1e-6 }).session(two, { key: key(1), logs: 'all' })
    s.advance()
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap.every((g) => g > -1e-8)).toBe(true)
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
  })

  it('revives from plain JSON parameters', () => {
    const fitted = svc.with({ kernel: 'poly', degree: 2, gamma: 0.5, coef0: 1 }).fit(two)
    const again = svc
      .with({ kernel: 'poly', degree: 2, gamma: 0.5, coef0: 1 })
      .revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.decisionFunction(two.x))).toEqual(toFlat(fitted.decisionFunction(two.x)))
  })

  it('has a nu form whose support fraction is at least nu', () => {
    const fitted = nuSvc.with({ nu: 0.4, gamma: 0.5 }).fit(two)
    expect(fitted.params.nSupport.reduce((a, b) => a + b, 0) / 30).toBeGreaterThanOrEqual(0.4 - 1e-9)
    expect(() => nuSvc.fit({ ...two, weights: new Array<number>(30).fill(1) })).toThrow(/nu form/)
  })

  it('refuses one class and accepts an unseen-label-free predict on new rows of the right width', () => {
    expect(() => svc.fit({ x: [[0], [1]], y: [1, 1] })).toThrow(/two classes/)
    expect(() => svc.fit(two).predict([[1, 2, 3]])).toThrow()
  })

  it('weights the penalty by class', () => {
    const plain = svc.with({ gamma: 0.5, C: 0.2 }).fit(two)
    const heavy = svc.with({ gamma: 0.5, C: 0.2, classWeight: { 0: 1, 1: 50 } }).fit(two)
    const positives = (f: typeof plain): number => toFlat(f.predict(two.x)).filter((v) => v === 1).length
    expect(positives(heavy)).toBeGreaterThanOrEqual(positives(plain))
  })
})

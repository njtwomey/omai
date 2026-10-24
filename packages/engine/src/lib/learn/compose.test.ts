import { describe, expect, it } from 'vitest'
import { gaussianNB, ridge, rows } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runTree, tensorSeries } from '../trace/index.ts'
import { columnTransformer, fitCache, makePipeline, pipeline } from './compose.ts'
import { requireVerb, revive, serialiseFitted } from './estimator.ts'
import { polynomialFeatures, randomOverSampler, standardScaler } from './preprocessing/index.ts'

const x = Array.from({ length: 12 }, (_, i) => [i / 3, Math.sin(i)])
const y = x.map(([a = 0, b = 0]) => 2 * a - b)

describe('pipelines', () => {
  it('route hyperparameters by step and keep the original', () => {
    const p = pipeline([
      ['scale', standardScaler()],
      ['model', ridge({ alpha: 1 })],
    ])
    const q = p.with({ 'model.alpha': 5 }).with({ scale: { withStd: false } })
    expect((p.hyper.steps[1] as readonly [string, ReturnType<typeof ridge>])[1].hyper.alpha).toBe(1)
    expect((q.hyper.steps[1] as readonly [string, ReturnType<typeof ridge>])[1].hyper.alpha).toBe(5)
    expect((q.hyper.steps[0] as readonly [string, ReturnType<typeof standardScaler>])[1].hyper.withStd).toBe(false)
    expect(() => p.with({ 'nope.alpha': 1 })).toThrow(/no step/)
  })

  it('serialise and revive', () => {
    const p = makePipeline(polynomialFeatures(), standardScaler(), ridge())
    const f = p.fit({ x, y })
    const again = revive(p, JSON.parse(JSON.stringify(serialiseFitted(f))))
    expect(toFlat(requireVerb(again, 'predict', 'test')(x))).toEqual(toFlat(requireVerb(f, 'predict', 'test')(x)))
  })

  it('trace each step as a child run and record intermediate outputs by policy', () => {
    const rec = recorder()
    pipeline([
      ['poly', polynomialFeatures()],
      ['scale', standardScaler()],
      ['model', ridge()],
    ]).fit({ x, y }, { tracer: rec, policy: { ...policies.debug, tensors: true, roles: ['activation'] } })
    const run = rec.run()
    expect(tensorSeries(run, 'output').length).toBeGreaterThanOrEqual(2)
    expect(runTree(run).length).toBe(1)
  })

  it('reuse cached transformers', () => {
    const cache = fitCache()
    const p = pipeline(
      [
        ['poly', polynomialFeatures()],
        ['model', ridge({ alpha: 1 })],
      ],
      { cache },
    )
    p.fit({ x, y })
    p.with({ 'model.alpha': 2 }).fit({ x, y })
    expect(cache.hits()).toBe(1)
    expect(cache.size()).toBe(1)
  })

  it('apply resamplers only while fitting', () => {
    const labels = x.map((_, i) => (i < 9 ? 0 : 1))
    const p = pipeline([
      ['over', randomOverSampler()],
      ['nb', gaussianNB()],
    ])
    const f = p.fit({ x, y: labels }, { key: key(1) })
    expect(toFlat(requireVerb(f, 'predict', 'test')(x)).length).toBe(12)
    expect(() => pipeline([['over', randomOverSampler()]]).fit({ x, y: labels })).toThrow(/last step/)
  })

  it('column transformers check their columns', () => {
    expect(() => columnTransformer([['a', standardScaler(), [5]]]).fit({ x })).toThrow(/out of range/)
    expect(
      rows(
        columnTransformer([['a', 'passthrough', [1]]])
          .fit({ x })
          .transform(x),
      )[0],
    ).toEqual([0])
  })
})

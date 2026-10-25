import { describe, expect, it } from 'vitest'
import { gaussianNB, logisticRegression, ridge, rows } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runTree, tensorSeries } from '../trace/index.ts'
import { parametersOf } from './_stages.ts'
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

  it('record what each stage did', () => {
    const f = pipeline([
      ['scale', standardScaler()],
      ['model', ridge({ alpha: 1 })],
    ]).fit({ x, y })
    const stages = f.exposition[0]
    expect(stages?.kind).toBe('pipelineStages')
    if (stages?.kind !== 'pipelineStages') return
    expect(stages.stages.map((s) => [s.name, s.kind, s.outputShape])).toEqual([
      ['scale', 'transformer', [12, 2]],
      ['model', 'estimator', [12, 1]],
    ])
    expect(stages.stages[0]?.outputSample).toHaveLength(4)
  })

  it('record each stage’s data, parameters and, for the final classifier, the surface and loss over the raw inputs', () => {
    const n = 40
    const xc = Array.from({ length: n }, (_, i) => [Math.cos(i), Math.sin(2 * i) + (i % 2)])
    const yc = xc.map((_, i) => i % 2)
    const f = pipeline(
      [
        ['poly', polynomialFeatures({ degree: 2, includeBias: false })],
        ['scale', standardScaler()],
        ['model', logisticRegression()],
      ],
      { explainRows: 30 },
    ).fit({ x: xc, y: yc, classes: ['a', 'b'] } as Parameters<ReturnType<typeof pipeline>['fit']>[0])
    const record = f.exposition[0]
    if (record?.kind !== 'pipelineStages') throw new Error('no pipelineStages')
    expect(record.classes).toEqual(['a', 'b'])
    expect(record.input?.columns).toEqual(['x0', 'x1'])
    expect(record.input?.rows).toHaveLength(30)
    expect(record.input?.n).toBe(n)
    const [poly, scale, model] = record.stages
    expect(poly?.output?.columns).toEqual(['x0', 'x1', 'x0^2', 'x0 x1', 'x1^2'])
    expect(poly?.output?.labels).toHaveLength(30)
    expect(poly?.parameters?.find((p) => p.name === 'powers')?.rows).toHaveLength(5)
    // Standardised columns have mean 0 and standard deviation 1 over all rows.
    for (const st of scale?.output?.stats ?? []) {
      expect(st.mean).toBeCloseTo(0, 10)
      expect(st.std).toBeCloseTo(1, 10)
    }
    expect(scale?.parameters?.find((p) => p.name === 'scale')?.columns).toHaveLength(5)
    expect(scale?.parameters?.some((p) => p.name === 'nSamplesSeen')).toBe(false)
    expect(model?.surface?.rows?.points).toHaveLength(30)
    expect(model?.surface?.probabilities).toHaveLength(2)
    expect(model?.loss?.name).toBe('log loss')
    expect(model?.loss?.values).toHaveLength(30)
    expect(model?.loss?.expected?.length).toBe(model?.surface?.y.length)
    expect(model?.loss?.mean).toBeGreaterThan(0)
    // Stages without surfaces carry none; a fit asked for no rows carries no data.
    expect(scale?.surface).toBeUndefined()
    const bare = pipeline(
      [
        ['scale', standardScaler()],
        ['model', ridge()],
      ],
      { explainRows: 0 },
    ).fit({ x, y })
    const r = bare.exposition[0]
    if (r?.kind !== 'pipelineStages') throw new Error('no pipelineStages')
    expect(r.input).toBeUndefined()
    expect(r.stages[0]?.output).toBeUndefined()
  })

  it('name parameters by the stage’s columns and pick out a projection’s explained variance', () => {
    const fitted = {
      estimator: 'p',
      kind: 'transformer' as const,
      params: {
        components: [
          [1, 0, 0],
          [0, 1, 0],
        ],
        explainedVarianceRatio: [0.7, 0.2],
        seen: [5, 5, 5],
        tag: 'x',
      },
    }
    const { parameters, explainedVarianceRatio } = parametersOf(fitted, ['a', 'b', 'c'], ['p0', 'p1'])
    expect(explainedVarianceRatio).toEqual([0.7, 0.2])
    expect(parameters.map((p) => p.name)).toEqual(['components', 'explainedVarianceRatio', 'tag'])
    expect(parameters[0]?.columns).toEqual(['a', 'b', 'c'])
    expect(parameters[0]?.rows).toEqual(['p0', 'p1'])
    expect(parameters[1]?.columns).toEqual(['p0', 'p1'])
  })

  it('carry a regressor’s squared error and targets', () => {
    const f = pipeline([
      ['scale', standardScaler()],
      ['model', ridge()],
    ]).fit({ x, y })
    const r = f.exposition[0]
    if (r?.kind !== 'pipelineStages') throw new Error('no pipelineStages')
    expect(r.stages[1]?.loss?.name).toBe('squared error')
    expect(r.stages[1]?.output?.targets).toHaveLength(12)
    expect(r.classes).toBeUndefined()
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

// The benchmark models as values: each one builds, binds the data its generator makes, compiles to a program whose log joint
// is finite, draws as plate notation, survives a JSON round trip, and is reproducible from its key; the suite is listed,
// filtered and handed to a worker's registry by name; and each model's truth is a set of well-formed oracles.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import { logJoint } from '../density.ts'
import { boundFromJSON, boundToJSON, fromJSON, toProgram, toStructure } from '../model/index.ts'
import { benchmarkEntries, benchmarkMethods, benchmarkRegistry, methodVerdicts } from './benchmark.ts'
import { benchmarkModel, benchmarkModels, whereBenchmarks } from './suite.ts'

const models = benchmarkModels()
const specs = models.filter((m) => m.definition !== undefined)

describe('the suite', () => {
  it('lists uniquely named models, each with a title, description, tags and references', () => {
    expect(new Set(models.map((m) => m.name)).size).toBe(models.length)
    expect(models.length).toBeGreaterThanOrEqual(16)
    for (const m of models) {
      expect(m.title.length, m.name).toBeGreaterThan(3)
      expect(m.description.length, m.name).toBeGreaterThan(20)
      expect(m.tags.length, m.name).toBeGreaterThan(1)
      expect(m.references.length, m.name).toBeGreaterThan(0)
      expect(m.plans(1).length, m.name).toBeGreaterThan(0)
    }
  })

  it('is filtered by tags and text, and found by name', () => {
    expect(whereBenchmarks(models, { tags: ['funnel'] }).map((m) => m.name)).toEqual([
      'eight-schools-centred',
      'funnel-centred',
      'funnel-non-centred',
    ])
    expect(whereBenchmarks(models, { tags: ['conjugate', 'counts'] }).map((m) => m.name)).toEqual(['gamma-poisson'])
    expect(whereBenchmarks(models, { tags: ['label-switching'] }).map((m) => m.name)).toEqual(['gaussian-mixture'])
    expect(benchmarkModel('banana').target).toBe('density')
    expect(() => benchmarkModel('nope')).toThrow(/no benchmark model called 'nope'/)
  })

  it('hands the spec models to a worker registry, and every model to the method list, by name', () => {
    const registry = benchmarkRegistry(models)
    expect(registry.names()).toEqual(specs.map((m) => m.name))
    expect(Object.keys(benchmarkEntries(models))).toEqual(registry.names())
    // Rebinding to JSON data, as a worker does.
    const json = JSON.parse(JSON.stringify({ y: [1, 0, 1, 1] }))
    expect(registry.bind('beta-bernoulli', json).sizes).toEqual({ flip: 4 })
    const offered = benchmarkMethods(models, ['nuts', 'conjugate'])
    expect(offered.map((m) => m.name)).toEqual(models.flatMap((m) => [`nuts/${m.name}`, `conjugate/${m.name}`]))
  })
})

describe.each(models)('$name', (model) => {
  it('makes data from its key, the same for the same key and different for another', () => {
    const a = model.generate(key(3))
    expect(JSON.stringify(model.generate(key(3)))).toBe(JSON.stringify(a))
    if (model.target === 'posterior' && !model.name.startsWith('eight-schools'))
      expect(JSON.stringify(model.generate(key(4)))).not.toBe(JSON.stringify(a))
    // The canonical data are the generator at seed 0, except where a dataset is embedded (eight schools).
    if (!model.name.startsWith('eight-schools'))
      expect(JSON.stringify(model.data)).toBe(JSON.stringify(model.generate(key(0))))
    // The data are JSON: numbers and nested arrays only.
    expect(JSON.parse(JSON.stringify(a.data))).toEqual(a.data)
  })

  it('compiles to a program with a finite log joint at a draw from its prior', () => {
    const lj = logJoint(model.program(), undefined, {}, { key: key(1) })
    expect(Number.isFinite(item(lj.logJoint))).toBe(true)
  })

  it('has a verdict with a reason from every method, and every planned method applies', () => {
    const verdicts = methodVerdicts(model)
    for (const v of verdicts) expect(v.verdict.reason.length, `${model.name} ${v.kind}`).toBeGreaterThan(3)
    for (const plan of model.plans(1))
      expect(verdicts.find((v) => v.kind === plan.kind)?.verdict.applies, `${model.name}: ${plan.kind}`).toBe(true)
  })

  it('has plans with notes, valid kinds and budgets that scale down', () => {
    for (const plan of model.plans(1)) {
      expect(plan.note.length).toBeGreaterThan(10)
      expect(['agree', 'differ', 'report']).toContain(plan.expect)
    }
    const small = model.plans(0.1)
    expect(small.map((p) => p.kind)).toEqual(model.plans(1).map((p) => p.kind))
  })
})

describe.each(specs)('$name as a spec', (model) => {
  const def = model.definition as NonNullable<typeof model.definition>

  it('binds its data, draws as plate notation and as a factor graph, and keeps the ids of its latent sites', () => {
    const bound = model.bound()
    for (const [dim, size] of Object.entries(bound.sizes)) expect(size, `${model.name} ${dim}`).toBeGreaterThan(0)
    const plate = toStructure(bound)
    const factor = toStructure(bound, { view: 'factor' })
    expect(plate.nodes.length).toBeGreaterThan(0)
    expect(factor.nodes.length).toBeGreaterThan(plate.nodes.length - 1)
    for (const n of def.spec.nodes.filter((x) => x.kind === 'latent'))
      expect(
        plate.nodes.some((s) => s.id === n.name),
        `${model.name}: ${n.name}`,
      ).toBe(true)
  })

  it('round-trips through JSON, as a spec and as a bound model', () => {
    const again = fromJSON(JSON.parse(JSON.stringify(def)), def.customs)
    expect(again.spec).toEqual(def.spec)
    const bound = model.bound()
    const back = boundFromJSON(JSON.parse(JSON.stringify(boundToJSON(bound))), def.customs)
    expect(back.sizes).toEqual(bound.sizes)
    const a = logJoint(model.program(), undefined, {}, { key: key(2) }).logJoint
    const b = logJoint(toProgram(back), undefined, {}, { key: key(2) }).logJoint
    expect(item(b)).toBe(item(a))
  })
})

describe.each(models)('the truth of $name', (model) => {
  it('is a set of well-formed marginal oracles (at the fast quality)', () => {
    const truth = model.truth(undefined, 'fast')
    const names = Object.keys(truth.marginals)
    expect(names.length).toBeGreaterThan(0)
    expect(truth.source.length).toBeGreaterThan(20)
    for (const name of names) {
      const o = truth.marginals[name]
      expect(o?.name).toBe(name)
      expect(Number.isFinite(o?.mean), name).toBe(true)
      expect(o?.sd, name).toBeGreaterThan(0)
      const lo = o?.quantile(0.1) ?? Number.NaN
      const mid = o?.quantile(0.5) ?? Number.NaN
      const hi = o?.quantile(0.9) ?? Number.NaN
      // The quantiles do not decrease, and are distinct for a continuous quantity (a quantity on a few values, a hidden state,
      // has the same quantile at several levels).
      expect(lo, name).toBeLessThanOrEqual(mid)
      expect(mid, name).toBeLessThanOrEqual(hi)
      const atom = (o?.cdf(mid) ?? 0) - (o?.cdf(mid - 1e-6) ?? 0) > 0.02
      if (!atom) {
        expect(lo, name).toBeLessThan(hi)
        // The CDF inverts the quantile function.
        expect(Math.abs((o?.cdf(mid) ?? 0) - 0.5), name).toBeLessThan(0.02)
        expect(o?.cdf(lo) ?? 1, name).toBeLessThan(o?.cdf(hi) ?? 0)
      } else expect(o?.cdf(mid) ?? 0, name).toBeGreaterThanOrEqual(0.5 - 0.02)
    }
  })
})

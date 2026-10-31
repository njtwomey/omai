import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { plantedRelations } from '../datasets/index.ts'
import { flatObjective } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  type KgModel,
  type KgOptions,
  kgProblem,
  kgScore,
  kgScoreBatch,
  kgSolvers,
  rankKg,
  trainKg,
} from './knowledge-graph.ts'

const models: KgModel[] = ['transe', 'distmult', 'complex', 'rotate']

describe('scoring functions', () => {
  const problemFor = (model: KgModel) =>
    kgProblem({ heads: [0, 1, 2], relations: [0, 1, 0], tails: [1, 2, 0] }, 4, 2, { model, dimension: 3 }, key(0))

  for (const model of models) {
    it(`${model}: the tensor path used by the loss agrees with the plain scorer`, () => {
      const p = problemFor(model)
      const x = Float64Array.from({ length: (p.x0 as Float64Array).length }, (_, i) => Math.sin(i + 1))
      const triples: [number, number, number][] = [
        [0, 0, 1],
        [3, 1, 2],
        [2, 1, 2],
      ]
      const batch = kgScoreBatch(p, x, triples)
      for (const [i, [h, r, t]] of triples.entries()) {
        const row = (off: number, w: number, e: number): Float64Array => x.slice(off + e * w, off + (e + 1) * w)
        const rOff = p.entities * p.entityWidth
        const plain = kgScore(
          model,
          row(0, p.entityWidth, h),
          row(rOff, p.relationWidth, r),
          row(0, p.entityWidth, t),
          p.options.norm,
        )
        expect(batch[i]).toBeCloseTo(plain, 10)
      }
    })
  }

  it('TransE scores a translation exactly, and DistMult is symmetric in head and tail', () => {
    expect(kgScore('transe', [1, 0], [0, 1], [1, 1])).toBeCloseTo(0, 12)
    expect(kgScore('transe', [1, 0], [0, 1], [1, 2])).toBeCloseTo(-1, 12)
    expect(kgScore('distmult', [1, 2], [3, 1], [2, 5])).toBe(kgScore('distmult', [2, 5], [3, 1], [1, 2]))
  })

  it('ComplEx separates the two directions of a relation with an imaginary part; RotatE rotates', () => {
    const h = [1, 0.5, 0.2, -0.3]
    const t = [0.4, 0.9, -0.1, 0.7]
    const r = [0.3, -0.2, 0.8, 0.5]
    expect(kgScore('complex', h, r, t)).not.toBeCloseTo(kgScore('complex', t, r, h), 3)
    // A rotation by theta maps h onto t exactly when t = h e^{i theta}: the score is 0.
    const theta = 0.7
    const rotated = [
      1 * Math.cos(theta) - 0.5 * Math.sin(theta),
      2 * Math.cos(theta) + 1 * Math.sin(theta),
      1 * Math.sin(theta) + 0.5 * Math.cos(theta),
      2 * Math.sin(theta) - 1 * Math.cos(theta),
    ]
    expect(kgScore('rotate', [1, 2, 0.5, -1], [theta, theta], rotated)).toBeCloseTo(0, 12)
  })
})

describe('the loss as a problem', () => {
  it('falls under training from its start, and the margin loss is nonnegative', () => {
    const train = { heads: [0, 1, 2, 3], relations: [0, 0, 0, 0], tails: [1, 2, 3, 0] }
    const p = kgProblem(train, 6, 1, { model: 'complex', dimension: 4, negatives: 4 }, key(1))
    const start = flatObjective(p.objective, p.x0).value(p.x0 as Float64Array)
    expect(start).toBeGreaterThan(0)
    const m = trainKg(train, 6, 1, { model: 'complex', dimension: 4, negatives: 4, maxSteps: 300 }, key(1))
    expect(m.loss).toBeGreaterThanOrEqual(0)
    // Some corruptions land on true triples, which keeps a floor under the loss.
    expect(m.loss).toBeLessThan(0.5 * start)
  })

  it('has the gradient of finite differences', () => {
    for (const loss of ['margin', 'negative-sampling'] as const) {
      const p = kgProblem(
        { heads: [0, 1, 2], relations: [0, 0, 1], tails: [1, 2, 3] },
        4,
        2,
        { model: 'complex', dimension: 2, loss, l2: 0.01 },
        key(2),
      )
      const f = flatObjective(p.objective, p.x0)
      const x = p.x0 as Float64Array
      const g = f.valueAndGrad(x).grad
      for (const i of [0, 5, 9, 17]) {
        const h = 1e-6
        const up = Float64Array.from(x)
        const dn = Float64Array.from(x)
        up[i] = (up[i] ?? 0) + h
        dn[i] = (dn[i] ?? 0) - h
        expect((f.value(up) - f.value(dn)) / (2 * h)).toBeCloseTo(g[i] ?? 0, 5)
      }
    }
  })

  it('offers the generic solvers with the stochastic gradient mode the default', () => {
    expect(kgSolvers.names).toContain('lbfgs')
    expect(kgSolvers.modeOf(undefined).name).toBe('gradient')
  })
})

describe('ranking', () => {
  it('filters the other true tails and averages tied ranks', () => {
    const scores = [5, 3, 5, 1]
    const emb = {
      entities: [new Float64Array(1), new Float64Array(1), new Float64Array(1), new Float64Array(1)],
      score: (_h: number, _r: number, t: number) => scores[t] as number,
    }
    // The true tail 0 ties with entity 2: rank 1.5; with (0, 0, 2) known as true it is filtered out: rank 1.
    expect(rankKg(emb, [[1, 0, 0]], [[1, 0, 0]], { heads: false }).ranks).toEqual([1.5])
    expect(
      rankKg(
        emb,
        [[1, 0, 0]],
        [
          [1, 0, 0],
          [1, 0, 2],
        ],
        { heads: false },
      ).ranks,
    ).toEqual([1])
    const r = rankKg(emb, [[1, 0, 3]], [[1, 0, 3]], { heads: false, ks: [1, 3] })
    expect(r.ranks).toEqual([4])
    expect(r.mrr).toBe(0.25)
    expect(r.hits).toEqual({ 1: 0, 3: 0 })
  })
})

describe('what each scorer can express (planted relations)', () => {
  const { data, truth } = generate(plantedRelations, { key: key(2), knobs: { groups: 6, perGroup: 10, heldOut: 0.4 } })
  const all = truth.extra?.['allTriples'] as [number, number, number][]
  const test = truth.extra?.['heldOut'] as [number, number, number][]
  const train = { heads: toFlat(data.heads), relations: toFlat(data.relations), tails: toFlat(data.tails) }
  const nE = data.entityNames.length
  const fit = (model: KgModel, options: KgOptions = {}) =>
    trainKg(train, nE, 5, { model, dimension: 16, margin: 1, negatives: 8, maxSteps: 800, ...options }, key(1))
  const trained = Object.fromEntries(models.map((m) => [m, fit(m)])) as Record<KgModel, ReturnType<typeof fit>>

  it('every scorer learns something: held-out facts rank far above chance', () => {
    const chance = Math.log(nE) / nE
    for (const model of models) {
      const r = rankKg(trained[model], test, all, { heads: false })
      expect(r.mrr, model).toBeGreaterThan(4 * chance)
    }
  })

  it('DistMult ranks held-out symmetric facts much better than TransE', () => {
    const sym = test.filter((f) => f[1] === 0)
    expect(sym.length).toBeGreaterThan(20)
    const distmult = rankKg(trained.distmult, sym, all, { heads: false }).mrr
    const transe = rankKg(trained.transe, sym, all, { heads: false }).mrr
    expect(distmult).toBeGreaterThan(0.5)
    expect(distmult).toBeGreaterThan(transe + 0.2)
  })

  it('DistMult cannot tell the two directions of the cycle; ComplEx and RotatE can', () => {
    const cycle = all.filter((f) => f[1] === 3)
    const direction = (m: (typeof trained)[KgModel]): number =>
      cycle.filter(([h, r, t]) => m.score(h, r, t) > m.score(t, r, h) + 1e-9).length / cycle.length
    // For any DistMult parameters score(h, r, t) = score(t, r, h), so no fact is ranked above its reverse.
    expect(direction(trained.distmult)).toBe(0)
    expect(direction(trained.complex)).toBeGreaterThan(0.9)
    expect(direction(trained.rotate)).toBeGreaterThan(0.9)
  })
})

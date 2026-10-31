import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { minimise } from '../../../testing/optimise.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { logsumexp, mean, mul, sub, sum, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import {
  CrossNetwork,
  DeepFm,
  Dlrm,
  dotInteraction,
  embeddingTower,
  FactorisationMachine,
  FieldAwareFm,
  fmScore,
  logQCorrection,
  NeuralCF,
  TwoTower,
  WideAndDeep,
} from './recommender.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

const ids = [
  [0, 4, 8],
  [1, 5, 9],
  [2, 4, 10],
  [0, 6, 9],
]
const values = [
  [1, 0.5, 2],
  [1, 1, 1],
  [0.3, 1.2, 0.7],
  [2, 1, 0.1],
]

describe('factorisation machine', () => {
  it('the linear-time form equals the explicit sum over pairs of features', () => {
    const fm = new FactorisationMachine(12, 3, { std: 1, key: key(0) })
    fm.loadStateDict({ ...fm.stateDict(), bias: tensor(0.7), linear: n(1, [12, 1]) })
    const factors = toArray(fm.params['factors'] as Value) as number[][]
    const linear = toFlat(fm.params['linear'] as Value)
    const got = toFlat(fm.forward({ ids, values }))
    const want = ids.map((row, b) => {
      let s = 0.7
      for (const [p, i] of row.entries()) {
        s += (linear[i] as number) * ((values[b] as number[])[p] as number)
        for (const [q, j] of row.entries())
          if (q > p)
            s +=
              (factors[i] as number[]).reduce((a, v, k) => a + v * ((factors[j] as number[])[k] as number), 0) *
              ((values[b] as number[])[p] as number) *
              ((values[b] as number[])[q] as number)
      }
      return s
    })
    expectClose(got, want, 1e-12)
    expect(
      fmScore(
        0,
        [[0, 0]],
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        [[1, 1]],
      ).shape,
    ).toEqual([1])
  })

  it('exposes the field-by-field interactions, which sum to the interaction part of the score', () => {
    const fm = new FactorisationMachine(12, 3, { std: 0.5, key: key(2) })
    const pairs = fm.interactions(fm.params, { ids, values })
    expect(pairs.shape).toEqual([4, 3, 3])
    const zeroBias = toFlat(fm.forward({ ids, values }))
    expectClose(sum(pairs, [1, 2]), zeroBias, 1e-12)
    // Strictly upper triangular.
    const flat = toFlat(pairs)
    for (let b = 0; b < 4; b++)
      for (let i = 0; i < 3; i++)
        for (let j = 0; j <= i; j++) expect(Math.abs(flat[b * 9 + i * 3 + j] as number)).toBe(0)
  })

  it('with a user field and an item field it is biased matrix factorisation, and it fits a rank-1 planted table', () => {
    const users = 6
    const items = 5
    const a = toFlat(n(3, [users]))
    const b = toFlat(n(4, [items]))
    const rows: number[][] = []
    const target: number[] = []
    for (let u = 0; u < users; u++)
      for (let i = 0; i < items; i++) {
        rows.push([u, users + i])
        target.push((a[u] as number) * (b[i] as number))
      }
    const fm = new FactorisationMachine(users + items, 2, { std: 0.3, key: key(5) })
    const y = tensor(target)
    const fit = minimise(
      fm.params,
      (p) => mean(mul(sub(fm.apply(p, { ids: rows }), y), sub(fm.apply(p, { ids: rows }), y))),
      400,
      0.05,
    )
    expect(fit.losses[0] as number).toBeGreaterThan(0.1)
    expect(fit.losses.at(-1) as number).toBeLessThan(1e-3)
    // Without the factors (k = 0 is not allowed to learn the product) the linear part cannot.
    const linearOnly = minimise(
      { ...fm.params, factors: tensor(Array.from({ length: users + items }, () => [0, 0])) },
      (p) =>
        mean(
          mul(
            sub(
              fm.apply({ ...p, factors: tensor(Array.from({ length: users + items }, () => [0, 0])) }, { ids: rows }),
              y,
            ),
            sub(
              fm.apply({ ...p, factors: tensor(Array.from({ length: users + items }, () => [0, 0])) }, { ids: rows }),
              y,
            ),
          ),
        ),
      400,
      0.05,
    )
    expect(linearOnly.losses.at(-1) as number).toBeGreaterThan(0.1)
  })

  it('is differentiable in the factors, the linear weights and the bias', () => {
    const fm = new FactorisationMachine(12, 3, { std: 0.5, key: key(6) })
    for (const name of ['factors', 'linear', 'bias']) {
      const f = (p: Value): Value => fm.apply({ ...fm.params, [name]: p }, { ids, values })
      expect(checkPrimitive(f, [fm.params[name] as Value], { rtol: 1e-5 }).failures, name).toEqual([])
    }
  })
})

describe('field-aware factorisation machine', () => {
  it('equals the plain machine when every feature uses one vector for all fields', () => {
    const fm = new FactorisationMachine(12, 3, { std: 1, key: key(10) })
    const shared = toArray(fm.params['factors'] as Value) as number[][]
    const ffm = new FieldAwareFm(12, 3, 3, { key: key(11) })
    ffm.loadStateDict({
      bias: fm.params['bias'] as Value,
      linear: fm.params['linear'] as Value,
      factors: tensor(shared.map((v) => [v, v, v])),
    })
    expectClose(ffm.forward({ ids, values }), fm.forward({ ids, values }), 1e-12)
  })

  it('uses the vector of a feature for the field it meets, and is differentiable', () => {
    const ffm = new FieldAwareFm(12, 3, 2, { std: 0.5, key: key(12) })
    expect(() => ffm.forward({ ids: [[0, 1]] })).toThrow(/fields/)
    const f = (p: Value): Value => ffm.apply({ ...ffm.params, factors: p }, { ids, values })
    expect(checkPrimitive(f, [ffm.params['factors'] as Value], { rtol: 1e-5 }).failures).toEqual([])
  })
})

describe('deep and wide models', () => {
  it('DeepFM is the FM score plus the deep logit, sharing the factors', () => {
    const model = new DeepFm(12, 3, 4, { hidden: [6], std: 0.5, key: key(20) })
    const fm = new FactorisationMachine(12, 4, { key: key(21) })
    fm.loadStateDict({
      bias: model.params['bias'] as Value,
      linear: model.params['linear'] as Value,
      factors: model.params['factors'] as Value,
    })
    const withDeep = toFlat(model.forward({ ids, values }))
    const silent = Object.fromEntries(
      Object.entries(model.stateDict()).map(([k, v]) => [k, k.startsWith('deep.') ? mul(v, 0) : v]),
    )
    model.loadStateDict(silent)
    expectClose(model.forward({ ids, values }), fm.forward({ ids, values }), 1e-12)
    expect(withDeep).not.toEqual(toFlat(fm.forward({ ids, values })))
    const f = (p: Value): Value => model.apply({ ...model.params, factors: p }, { ids, values })
    expect(checkPrimitive(f, [model.params['factors'] as Value], { rtol: 1e-5 }).failures).toEqual([])
  })

  it('wide and deep adds a linear path over the features to the embedding network', () => {
    const model = new WideAndDeep(12, 3, { embeddingDim: 4, hidden: [5], key: key(22) })
    const base = toFlat(model.forward({ ids }))
    expect(base.length).toBe(4)
    const wide = tensor(Array.from({ length: 12 }, (_, i) => [i === 0 ? 2 : 0]))
    const shifted = toFlat(model.apply({ ...model.params, wide }, { ids }))
    expectClose(
      shifted,
      base.map((v, b) => v + ((ids[b] as number[])[0] === 0 ? 2 : 0)),
      1e-12,
    )
    expect(Object.keys(model.stateDict())).toEqual(
      expect.arrayContaining(['wide', 'embedding.weight', 'deep.0.weight']),
    )
  })
})

describe('neural collaborative filtering', () => {
  const user = [0, 1, 2, 1]
  const item = [3, 4, 0, 4]

  it('the GMF mode is a linear map of the elementwise product, and the modes have the parameters they need', () => {
    const gmf = new NeuralCF(4, 5, { mode: 'gmf', gmfDim: 3, key: key(30) })
    const p = gmf.params as Record<string, Record<string, Value>>
    const pu = toArray((p['userGmf'] as Record<string, Value>)['weight'] as Value) as number[][]
    const qi = toArray((p['itemGmf'] as Record<string, Value>)['weight'] as Value) as number[][]
    const w = toFlat((p['output'] as Record<string, Value>)['weight'] as Value)
    const b = toFlat((p['output'] as Record<string, Value>)['bias'] as Value)[0] as number
    const want = user.map(
      (u, r) =>
        b +
        w.reduce(
          (a, wk, k) =>
            a + wk * ((pu[u] as number[])[k] as number) * ((qi[item[r] as number] as number[])[k] as number),
          0,
        ),
    )
    expectClose(gmf.forward({ user, item }), want, 1e-12)
    expect(Object.keys(gmf.stateDict()).some((k) => k.startsWith('tower'))).toBe(false)
    const deep = new NeuralCF(4, 5, { mode: 'mlp', key: key(31) })
    expect(Object.keys(deep.stateDict()).some((k) => k.includes('Gmf'))).toBe(false)
    const both = new NeuralCF(4, 5, { key: key(32) })
    expect(both.parameterCount()).toBeGreaterThan(gmf.parameterCount())
    expect(both.userEmbeddings(both.params).shape).toEqual([4, 8])
    expect(both.itemEmbeddings(both.params).shape).toEqual([5, 8])
  })

  it('is differentiable in every parameter group', () => {
    const model = new NeuralCF(4, 5, { gmfDim: 2, mlpDim: 2, hidden: [3], key: key(33) })
    for (const path of [
      ['userGmf', 'weight'],
      ['itemMlp', 'weight'],
      ['tower', '0', 'weight'],
      ['output', 'bias'],
    ]) {
      const get = (t: Params, p: readonly string[]): Value => p.reduce<unknown>((a, k) => (a as Params)[k], t) as Value
      const set = (t: Params, p: readonly string[], v: Value): Params => {
        const [h, ...r] = p
        return h === undefined ? t : { ...t, [h]: r.length === 0 ? v : set(t[h] as Params, r, v) }
      }
      const f = (v: Value): Value => model.apply(set(model.params, path, v), { user, item })
      expect(checkPrimitive(f, [get(model.params, path)], { rtol: 1e-5 }).failures, path.join('.')).toEqual([])
    }
  })
})

describe('two-tower model', () => {
  it('scores every user against every item: the logits are the dot products of the unit embeddings over the temperature', () => {
    const model = new TwoTower(embeddingTower(6, 4, [8], key(40)), embeddingTower(7, 4, [8], key(41)), {
      temperature: 0.5,
    })
    const out = model.forward({ user: [0, 1, 2], item: [3, 4, 5] })
    expect(out.logits.shape).toEqual([3, 3])
    const u = toArray(out.user) as number[][]
    const v = toArray(out.item) as number[][]
    for (let i = 0; i < 3; i++) {
      expect(Math.hypot(...(u[i] as number[]))).toBeCloseTo(1, 9)
      for (let j = 0; j < 3; j++)
        expect(toFlat(out.logits)[i * 3 + j]).toBeCloseTo(
          2 * (u[i] as number[]).reduce((a, x, k) => a + x * ((v[j] as number[])[k] as number), 0),
          12,
        )
    }
    expectClose(model.encodeItems(model.params, [3, 4, 5]), out.item, 1e-14)
    const raw = new TwoTower(embeddingTower(6, 4, [], key(42)), embeddingTower(7, 4, [], key(43)), { normalise: false })
    expect(Math.hypot(...toFlat(raw.forward({ user: [0], item: [0] }).user))).not.toBeCloseTo(1, 3)
  })

  it('in-batch softmax training pulls matching users and items together', () => {
    const model = new TwoTower(embeddingTower(8, 6, [], key(44)), embeddingTower(8, 6, [], key(45)), {
      temperature: 0.2,
    })
    const batch = [0, 1, 2, 3, 4, 5, 6, 7]
    const loss = (p: Params): Value => {
      const logits = model.apply(p, { user: batch, item: batch }).logits
      const diagonal = sum(mul(logits, tensor(batch.map((i) => batch.map((j) => (i === j ? 1 : 0))))), 1)
      return mean(sub(logsumexp(logits, 1), diagonal))
    }
    const fit = minimise(model.params, loss, 150, 0.1)
    expect(fit.losses[0] as number).toBeGreaterThan(1.5)
    expect(fit.losses.at(-1) as number).toBeLessThan(0.1)
    const logits = toFlat(model.apply(fit.params, { user: batch, item: batch }).logits)
    for (let i = 0; i < 8; i++)
      expect(logits.slice(i * 8, i * 8 + 8).indexOf(Math.max(...logits.slice(i * 8, i * 8 + 8)))).toBe(i)
  })

  it('the log-Q correction lowers the columns of frequent items', () => {
    expect(
      toArray(
        logQCorrection(
          [
            [1, 1],
            [1, 1],
          ],
          [0, -1],
        ),
      ),
    ).toEqual([
      [1, 2],
      [1, 2],
    ])
  })
})

describe('cross network and DLRM', () => {
  it('the cross layers are the explicit recurrence, and zero weights leave the input plus the bias', () => {
    const net = new CrossNetwork(3, 2, { key: key(50) })
    const x = n(51, [2, 3])
    const xs = toArray(x) as number[][]
    const p = net.params as Record<string, Record<string, Value>>
    const step = (x0: number[], h: number[], layer: string): number[] => {
      const w = toFlat((p[layer] as Record<string, Value>)['weight'] as Value)
      const bias = toFlat((p[layer] as Record<string, Value>)['bias'] as Value)
      const gate = h.reduce((a, v, k) => a + v * (w[k] as number), 0)
      return h.map((v, k) => (x0[k] as number) * gate + (bias[k] as number) + v)
    }
    const want = xs.map((row) => step(row, step(row, row, '0'), '1'))
    expectClose(net.forward(x), tensor(want), 1e-12)
    const zero = new CrossNetwork(3, 1, { key: key(52) })
    zero.loadStateDict({ '0.weight': tensor([0, 0, 0]), '0.bias': tensor([1, 2, 3]) })
    expectClose(zero.forward([[1, 1, 1]]), tensor([[2, 3, 4]]))
    expect(checkPrimitive((v: Value) => net.apply(net.params, v), [x], { rtol: 1e-5 }).failures).toEqual([])
  })

  it('DLRM crosses the dense and the sparse vectors by dot products and trains end to end', () => {
    expect(
      toArray(
        dotInteraction([
          [
            [1, 0],
            [1, 2],
            [0, 3],
          ],
        ]),
      ),
    ).toEqual([[1, 0, 6]])
    expect(dotInteraction(n(60, [5, 4, 3])).shape).toEqual([5, 6])
    const model = new Dlrm(3, [6, 4], 5, { bottomHidden: [4], topHidden: [6], key: key(61) })
    const input = { dense: n(62, [7, 3]), sparse: Array.from({ length: 7 }, (_, i) => [i % 6, i % 4]) }
    expect(model.forward(input).shape).toEqual([7])
    expect(Object.keys(model.stateDict())).toEqual(
      expect.arrayContaining(['table0.weight', 'table1.weight', 'bottom.0.weight', 'top.0.weight']),
    )
    const f = (p: Value): Value => model.apply({ ...model.params, table0: { weight: p } }, input)
    expect(
      checkPrimitive(f, [(model.params['table0'] as Record<string, Value>)['weight'] as Value], { rtol: 1e-5 })
        .failures,
    ).toEqual([])
  })
})

import { describe, expect, it } from 'vitest'
import { logSumExpOf } from '../../base/index.ts'
import { analyticObjective, lbfgs } from '../../optim/index.ts'
import { categorical, foldIn, key, normal } from '../../random/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import { crfLogLikelihood, crfScore, linearChainCrf } from './crf.ts'
import { type GraphCrf, type GraphInput, type GraphLabelling, graphCrf, graphParameterNames } from './graph-crf.ts'
import { type GraphOracle, graphDecode, graphFactorGraph, graphOracle, graphScore } from './graph-oracle.ts'
import { graphPseudoLikelihood } from './graph-pseudo-likelihood.ts'
import {
  fullyConnectedCrf,
  fullyConnectedInput,
  gridCrf,
  gridInput,
  higherOrderChainCrf,
  higherOrderChainInput,
  skipChainCrf,
  skipChainInput,
  treeCrf,
  treeInput,
} from './graph-structures.ts'
import type { StructuredExample } from './oracle.ts'

const randomVector = (seed: number, n: number, scale: number): Float64Array =>
  Float64Array.from(toFlat(normal(key(seed), [n])), (v) => v * scale)

/** Every labelling of n nodes with K labels, by index. */
function* labellings(n: number, K: number): Generator<number[]> {
  const y = new Array<number>(n).fill(0)
  for (let total = K ** n, i = 0; i < total; i++) {
    yield [...y]
    for (let a = n - 1; a >= 0; a--) {
      y[a] = (y[a] ?? 0) + 1
      if ((y[a] ?? 0) < K) break
      y[a] = 0
    }
  }
}

/** log Z, expected counts and the best labelling by enumeration of the features. */
function enumerate(oracle: GraphOracle, w: ArrayLike<number>, x: GraphInput, K: number) {
  const all = [...labellings(x.nodes.length, K)].map((y) => {
    const phi = oracle.features(x, y)
    return { y, phi, score: phi.reduce((s, v, i) => s + v * (w[i] ?? 0), 0) }
  })
  const logZ = logSumExpOf(all.map((a) => a.score))
  const counts = new Float64Array(oracle.size)
  for (const a of all) {
    const p = Math.exp(a.score - logZ)
    a.phi.forEach((v, i) => {
      counts[i] = (counts[i] ?? 0) + p * v
    })
  }
  const best = all.reduce((b, a) => (a.score > b.score ? a : b))
  return { logZ, counts, all, best }
}

/** The penalised negative log-likelihood of labelled graphs through an oracle, with its gradient. */
function nll(
  oracle: GraphOracle,
  data: readonly StructuredExample<GraphInput, GraphLabelling>[],
  w: ArrayLike<number>,
  c2 = 0,
): { value: number; gradient: Float64Array } {
  const gradient = new Float64Array(oracle.size)
  let value = 0
  for (const { x, y } of data) {
    const e = oracle.expectations(w, x)
    const phi = oracle.features(x, y)
    value += e.logPartition - phi.reduce((s, v, i) => s + v * (w[i] ?? 0), 0)
    for (let i = 0; i < gradient.length; i++) gradient[i] = (gradient[i] ?? 0) + (e.counts[i] ?? 0) - (phi[i] ?? 0)
  }
  for (let i = 0; i < gradient.length; i++) {
    value += c2 * (w[i] ?? 0) ** 2
    gradient[i] = (gradient[i] ?? 0) + 2 * c2 * (w[i] ?? 0)
  }
  return { value, gradient }
}

function minimise(f: (w: Float64Array) => { value: number; gradient: Float64Array }, x0: Float64Array): Float64Array {
  const r = lbfgs(
    analyticObjective({
      value: (w) => f(Float64Array.from(w)).value,
      gradient: (w) => f(Float64Array.from(w)).gradient,
    }),
    x0,
    {
      gradTol: 1e-6,
      maxIterations: 200,
    },
  )
  return Float64Array.from(toFlat(r.x as Value))
}

/** The mean absolute difference of the node marginals under two weight vectors (identifiable, unlike the weights). */
function marginalGap(
  oracle: GraphOracle,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  inputs: readonly GraphInput[],
): number {
  let total = 0
  let count = 0
  for (const x of inputs) {
    const pa = oracle.marginals(a, x).nodes
    const pb = oracle.marginals(b, x).nodes
    for (const [i, row] of pa.entries())
      for (const [k, p] of row.entries()) {
        total += Math.abs(p - (pb[i]?.[k] ?? 0))
        count++
      }
  }
  return total / count
}

const attributes = (seed: number, n: number, names: readonly string[]): Record<string, number>[] => {
  const v = randomVector(seed, n * names.length, 1)
  return Array.from({ length: n }, (_, i) => Object.fromEntries(names.map((a, j) => [a, v[i * names.length + j] ?? 0])))
}

const contrast = (a: Record<string, number>, b: Record<string, number>): Record<string, number> => ({
  contrast: Math.abs((a['f'] ?? 0) - (b['f'] ?? 0)),
})

interface Case {
  readonly name: string
  readonly model: GraphCrf
  readonly x: GraphInput
  readonly exact: boolean
}

const nodes = (n: number, seed: number): Record<string, number>[] => attributes(seed, n, ['f', 'g'])

const cases: Case[] = [
  {
    name: 'a tree from a parent array',
    model: treeCrf(['a', 'b', 'c'], ['f', 'g']),
    x: treeInput(nodes(6, 1), { parents: [-1, 0, 0, 1, 1, 2] }),
    exact: true,
  },
  {
    name: 'a forest from an edge list with edge attributes',
    model: treeCrf(['a', 'b'], ['f', 'g'], { edgeAttributes: ['contrast'], symmetricAttributes: true }),
    x: treeInput(
      nodes(7, 2),
      {
        edges: [
          [0, 1],
          [1, 2],
          [3, 2],
          [4, 5],
        ],
      },
      { edgeAttributes: contrast },
    ),
    exact: true,
  },
  { name: 'a 2 x 3 grid', model: gridCrf(['a', 'b'], ['f', 'g']), x: gridInput(nodes(6, 3), 2, 3), exact: true },
  {
    name: 'a 3 x 3 directional 8-neighbourhood grid with edge attributes',
    model: gridCrf(['a', 'b'], ['f'], { directional: true, neighbourhood: 8, edgeAttributes: ['contrast'] }),
    x: gridInput(nodes(9, 4), 3, 3, { directional: true, neighbourhood: 8, edgeAttributes: contrast }),
    exact: true,
  },
  {
    name: 'a skip chain',
    model: skipChainCrf(['a', 'b', 'c'], ['f']),
    x: skipChainInput(nodes(7, 5), { keys: ['x', undefined, 'y', 'x', undefined, 'y', undefined] }),
    exact: true,
  },
  {
    name: 'a second-order chain',
    model: higherOrderChainCrf(['a', 'b', 'c'], ['f', 'g'], 2),
    x: higherOrderChainInput(nodes(6, 6), 2),
    exact: true,
  },
  {
    name: 'a third-order chain',
    model: higherOrderChainCrf(['a', 'b'], ['f'], 3),
    x: higherOrderChainInput(nodes(7, 7), 3),
    exact: true,
  },
  {
    name: 'a fully connected graph',
    model: fullyConnectedCrf(['a', 'b', 'c'], ['f']),
    x: fullyConnectedInput(nodes(5, 8)),
    exact: true,
  },
  {
    name: 'a graph with parallel edges of two types and an isolated node',
    model: graphCrf(['a', 'b'], ['f'], { edgeTypes: ['p', 'q'], symmetric: ['q'] }),
    x: {
      nodes: nodes(5, 9),
      edges: [
        { from: 0, to: 1, type: 'p' },
        { from: 1, to: 0, type: 'q' },
        { from: 1, to: 2, type: 'p' },
        { from: 3, to: 2, type: 'q' },
      ],
    },
    exact: true,
  },
]

describe('graph CRF parameters', () => {
  it('lays the weights out as the node, pair, edge-attribute and clique blocks, with symmetric types packed', () => {
    const m = graphCrf(['a', 'b'], ['f'], {
      edgeTypes: ['p', 'q'],
      symmetric: ['q'],
      edgeAttributes: ['c'],
      cliqueTypes: { t: 3 },
    })
    expect(m.layout.size).toBe(4 + 4 + 3 + 4 + 8)
    expect(graphParameterNames(m)).toHaveLength(m.layout.size)
    expect(graphParameterNames(m).slice(4 + 4, 4 + 4 + 3)).toEqual(['pair:q|a|a', 'pair:q|a|b', 'pair:q|b|b'])
    expect(() => graphCrf(['a', 'b'], [], { weights: [1] })).toThrow()
  })

  it('rejects unknown edge types, self-loops and out-of-range cliques', () => {
    const m = graphCrf(['a', 'b'], [])
    const o = graphOracle(m)
    const w = new Float64Array(o.size)
    expect(() => o.expectations(w, { nodes: [{}, {}], edges: [{ from: 0, to: 1, type: 'zzz' }] })).toThrow(/edge type/)
    expect(() => o.expectations(w, { nodes: [{}, {}], edges: [{ from: 0, to: 0 }] })).toThrow(/self-loop/)
    expect(() => o.expectations(w, { nodes: [{}, {}], edges: [], cliques: [{ type: 'zzz', nodes: [0, 1] }] })).toThrow()
  })
})

describe('graph CRF expectations (exact inference against enumeration)', () => {
  for (const c of cases) {
    it(`${c.name}: log Z and expected counts match enumeration, whichever exact method runs`, () => {
      const K = c.model.labels.length
      const w = randomVector(11, c.model.layout.size, 0.7)
      const truth = enumerate(graphOracle(c.model), w, c.x, K)
      for (const inference of ['auto', 'exact', 'junctionTree'] as const) {
        const e = graphOracle(c.model, { inference }).expectations(w, c.x)
        expect(e.exact).toBe(true)
        expect(e.logPartition).toBeCloseTo(truth.logZ, 9)
        for (let i = 0; i < w.length; i++) expect(e.counts[i]).toBeCloseTo(truth.counts[i] ?? 0, 8)
      }
    })

    it(`${c.name}: decoding finds the best labelling, with and without cost augmentation`, () => {
      const K = c.model.labels.length
      const w = randomVector(12, c.model.layout.size, 1)
      const oracle = graphOracle(c.model)
      const truth = enumerate(oracle, w, c.x, K)
      const d = oracle.decode(w, c.x)
      expect(d.score).toBeCloseTo(truth.best.score, 9)
      const gold = truth.all[3]?.y ?? []
      const against = oracle.decode(w, c.x, { against: gold })
      const augmented = Math.max(...truth.all.map((a) => a.score + oracle.loss(gold, a.y)))
      expect(against.score).toBeCloseTo(augmented, 9)
      const found = truth.all.find((a) => a.y.join() === against.y.map((l) => c.model.labels.indexOf(String(l))).join())
      expect((found?.score ?? Number.NaN) + oracle.loss(gold, against.y)).toBeCloseTo(augmented, 9)
    })
  }

  it('reports the inference chosen: the tree for a forest, a junction tree for a loopy graph, belief propagation when too wide', () => {
    const grid = gridInput(nodes(9, 3), 3, 3)
    const m = gridCrf(['a', 'b'], ['f'])
    expect(graphOracle(m).plan(gridInput(nodes(4, 1), 1, 4))).toMatchObject({
      inference: 'tree',
      exact: true,
    })
    expect(graphOracle(m).plan(grid)).toMatchObject({ inference: 'junctionTree', exact: true })
    expect(graphOracle(m, { maxStates: 4 }).plan(grid)).toMatchObject({ inference: 'bp', exact: false })
    expect(() =>
      graphOracle(m, { maxStates: 4, inference: 'exact' }).expectations(new Float64Array(m.layout.size), grid),
    ).toThrow(/exact inference needs/)
    expect(() => graphOracle(m, { inference: 'tree' }).expectations(new Float64Array(m.layout.size), grid)).toThrow(
      /forest/,
    )
  })

  it('a single node, an empty graph and graphScore', () => {
    const m = graphCrf(['a', 'b'], ['f'], { bias: false, weights: [1, -1, 0, 0, 0, 0] })
    const o = graphOracle(m)
    const one = o.expectations(m.weights, { nodes: [{ f: 2 }], edges: [] })
    expect(one.logPartition).toBeCloseTo(Math.log(Math.exp(2) + Math.exp(-2)), 12)
    expect(o.expectations(m.weights, { nodes: [], edges: [] })).toMatchObject({ logPartition: 0 })
    expect(graphScore(m, { nodes: [{ f: 2 }], edges: [] }, ['b'])).toBe(-2)
    expect(graphDecode(m, { nodes: [{ f: 2 }], edges: [] }).y).toEqual(['a'])
  })
})

describe('graph CRF gradients', () => {
  const finiteDifference = (f: (w: Float64Array) => number, w: Float64Array, i: number): number => {
    const h = 1e-6
    const up = Float64Array.from(w)
    const down = Float64Array.from(w)
    up[i] = (up[i] ?? 0) + h
    down[i] = (down[i] ?? 0) - h
    return (f(up) - f(down)) / (2 * h)
  }

  it('the exact negative log-likelihood gradient matches finite differences on a grid, a tree and a higher-order chain', () => {
    for (const c of [cases[0], cases[2], cases[5]] as Case[]) {
      const oracle = graphOracle(c.model)
      const K = c.model.labels.length
      const y = [...labellings(c.x.nodes.length, K)][5] ?? []
      const w = randomVector(13, oracle.size, 0.5)
      const data = [{ x: c.x, y }]
      const g = nll(oracle, data, w, 0.1).gradient
      for (let i = 0; i < w.length; i++)
        expect(g[i]).toBeCloseTo(
          finiteDifference((v) => nll(oracle, data, v, 0.1).value, w, i),
          5,
        )
    }
  })

  it('the pseudo-likelihood gradient matches finite differences, with edges, attributes and cliques', () => {
    for (const c of [cases[1], cases[3], cases[5]] as Case[]) {
      const K = c.model.labels.length
      const y = [...labellings(c.x.nodes.length, K)][7] ?? []
      const w = randomVector(14, c.model.layout.size, 0.5)
      const data = [{ x: c.x, y }]
      const g = graphPseudoLikelihood(c.model, data, w, { c2: 0.05 }).gradient
      for (let i = 0; i < w.length; i++)
        expect(g[i]).toBeCloseTo(
          finiteDifference((v) => graphPseudoLikelihood(c.model, data, v, { c2: 0.05 }).value, w, i),
          5,
        )
    }
  })

  it('on a graph without edges the pseudo-likelihood is the likelihood', () => {
    const m = graphCrf(['a', 'b', 'c'], ['f', 'g'])
    const data = [{ x: { nodes: nodes(4, 2), edges: [] }, y: ['a', 'c', 'b', 'a'] }]
    const w = randomVector(15, m.layout.size, 0.5)
    const exact = nll(graphOracle(m), data, w)
    const pl = graphPseudoLikelihood(m, data, w)
    expect(pl.value).toBeCloseTo(exact.value, 10)
    for (let i = 0; i < w.length; i++) expect(pl.gradient[i]).toBeCloseTo(exact.gradient[i] ?? 0, 10)
  })
})

describe('approximate inference on a loopy graph', () => {
  const grid = gridInput(nodes(9, 21), 3, 3)
  const model = gridCrf(['a', 'b'], ['f', 'g'])
  const w = randomVector(22, model.layout.size, 0.25)
  const truth = enumerate(graphOracle(model), w, grid, 2)

  it('loopy belief propagation is close to exact for moderate couplings, and says it is not exact', () => {
    const e = graphOracle(model, { inference: 'bp', bp: { damping: 0.3 } }).expectations(w, grid)
    expect(e.exact).toBe(false)
    expect(Math.abs(e.logPartition - truth.logZ)).toBeLessThan(0.05)
    const err = Math.max(...Array.from(e.counts, (c, i) => Math.abs(c - (truth.counts[i] ?? 0))))
    expect(err).toBeLessThan(0.15)
  })

  it('tree-reweighted belief propagation bounds log Z from above and gives usable counts', () => {
    const e = graphOracle(model, { inference: 'trw' }).expectations(w, grid)
    expect(e.exact).toBe(false)
    expect(e.logPartition).toBeGreaterThanOrEqual(truth.logZ - 1e-9)
    expect(e.logPartition - truth.logZ).toBeLessThan(0.3)
    const err = Math.max(...Array.from(e.counts, (c, i) => Math.abs(c - (truth.counts[i] ?? 0))))
    expect(err).toBeLessThan(0.4)
  })

  it('the belief-propagation and tree-reweighted gradients are consistent with their own objectives at a fixed point', () => {
    const y = [0, 1, 1, 0, 1, 0, 0, 1, 1]
    for (const options of [
      { inference: 'bp', bp: { damping: 0.3, tol: 1e-13, maxIter: 2000 } },
      { inference: 'trw', trw: { tol: 1e-13, maxIter: 5000 } },
    ] as const) {
      const oracle = graphOracle(model, options)
      const f = (v: Float64Array): number => nll(oracle, [{ x: grid, y }], v).value
      const g = nll(oracle, [{ x: grid, y }], w).gradient
      for (const i of [0, 3, 7, model.layout.size - 1]) {
        const h = 1e-5
        const up = Float64Array.from(w)
        const down = Float64Array.from(w)
        up[i] = (up[i] ?? 0) + h
        down[i] = (down[i] ?? 0) - h
        expect(g[i]).toBeCloseTo((f(up) - f(down)) / (2 * h), 4)
      }
    }
  })

  it('decoding by max-product, MPLP and a user function agrees with the exact MAP on an easy instance, and the function is called', () => {
    const strong = randomVector(23, model.layout.size, 1.5)
    const oracle = graphOracle(model)
    const best = oracle.decode(strong, grid)
    let called = 0
    for (const map of ['junctionTree', 'mplp', 'maxProduct'] as const) {
      const d = graphOracle(model, { map }).decode(strong, grid)
      expect(d.score).toBeLessThanOrEqual(best.score + 1e-9)
      if (map === 'junctionTree') expect(d.y).toEqual(best.y)
    }
    const custom = graphOracle(model, {
      map: (fg) => {
        called++
        // the same discrete model, solved by enumeration through the factor graph
        const names = fg.kind === 'factorGraph' ? fg.variables.map((v) => v.name) : []
        const exact = graphOracle(model).decode(strong, grid)
        return {
          assignment: Object.fromEntries(names.map((n, i) => [n, String(exact.y[i])])),
          indices: Object.fromEntries(names.map((n, i) => [n, model.labels.indexOf(String(exact.y[i]))])),
          logValue: exact.score,
        }
      },
    }).decode(strong, grid)
    expect(called).toBe(1)
    expect(custom.y).toEqual(best.y)
    expect(graphFactorGraph(model, grid, strong).factors.length).toBe(9 + 12)
  })
})

describe('pseudo-likelihood against maximum likelihood', () => {
  it('both recover a planted 2 x 2 grid model from sampled data; the pseudo-likelihood is the cheaper, rougher estimate', () => {
    const model = gridCrf(['a', 'b'], ['f'])
    const truth = randomVector(31, model.layout.size, 1)
    const oracle = graphOracle(model)
    const data = Array.from({ length: 120 }, (_, s): StructuredExample<GraphInput, GraphLabelling> => {
      const x = gridInput(attributes(100 + s, 4, ['f']), 2, 2)
      const all = [...labellings(4, 2)]
      const scores = all.map((y) => oracle.features(x, y).reduce((a, v, i) => a + v * (truth[i] ?? 0), 0))
      const pick = toFlat(categorical(foldIn(key(5), s), scores, [1], { logits: true }))[0] ?? 0
      return { x, y: all[pick] ?? [] }
    })
    const ml = minimise((w) => nll(oracle, data, w, 0.01), new Float64Array(model.layout.size))
    const pl = minimise((w) => graphPseudoLikelihood(model, data, w, { c2: 0.01 }), new Float64Array(model.layout.size))
    const held = Array.from({ length: 30 }, (_, s) => gridInput(attributes(700 + s, 4, ['f']), 2, 2))
    const mlGap = marginalGap(oracle, ml, truth, held)
    const plGap = marginalGap(oracle, pl, truth, held)
    expect(mlGap).toBeLessThan(0.04)
    expect(plGap).toBeLessThan(0.06)
    expect(marginalGap(oracle, new Float64Array(model.layout.size), truth, held)).toBeGreaterThan(2 * mlGap)
  })
})

describe('learning planted structures', () => {
  /** Sample labellings of an input from the model with the given weights, by enumeration. */
  function sample(
    oracle: GraphOracle,
    truth: Float64Array,
    inputs: readonly GraphInput[],
    K: number,
    seed: number,
  ): StructuredExample<GraphInput, GraphLabelling>[] {
    return inputs.map((x, s) => {
      const all = [...labellings(x.nodes.length, K)]
      const scores = all.map((y) => oracle.features(x, y).reduce((a, v, i) => a + v * (truth[i] ?? 0), 0))
      const pick = toFlat(categorical(foldIn(key(seed), s), scores, [1], { logits: true }))[0] ?? 0
      return { x, y: all[pick] ?? [] }
    })
  }

  it('a tree CRF with instances of different shapes recovers its planted weights, with exact inference throughout', () => {
    const model = treeCrf(['a', 'b'], ['f'], { edgeTypes: ['parent'] })
    const oracle = graphOracle(model)
    const truth = Float64Array.from([0.3, -0.3, 1.5, -1, 0.5, -0.5, -0.4, 0.4])
    expect(truth.length).toBe(model.layout.size)
    const parentsOf = (s: number): number[] => {
      const n = 4 + (s % 4)
      return Array.from({ length: n }, (_, i) => (i === 0 ? -1 : Math.floor((((i * 7 + s * 3) % 11) / 11) * i)))
    }
    const inputs = Array.from({ length: 150 }, (_, s) =>
      treeInput(attributes(200 + s, 4 + (s % 4), ['f']), { parents: parentsOf(s) }),
    )
    const data = sample(oracle, truth, inputs, 2, 6)
    expect(new Set(data.map((d) => d.x.nodes.length)).size).toBeGreaterThan(1)
    const w = minimise((v) => nll(oracle, data, v, 0.01), new Float64Array(model.layout.size))
    const distance = Math.hypot(...Array.from(w, (v, i) => v - (truth[i] ?? 0)))
    expect(distance / Math.hypot(...truth)).toBeLessThan(0.3)
    // the fitted model decodes held-out sampled trees about as well as the truth
    const test = sample(
      oracle,
      truth,
      inputs
        .slice(0, 40)
        .map((x, s) => treeInput(attributes(900 + s, x.nodes.length, ['f']), { parents: parentsOf(s) })),
      2,
      7,
    )
    const accuracy = (weights: Float64Array): number => {
      let right = 0
      let total = 0
      for (const { x, y } of test) {
        const d = oracle.decode(weights, x)
        right += x.nodes.length - oracle.loss(y, d.y)
        total += x.nodes.length
      }
      return right / total
    }
    expect(accuracy(w)).toBeGreaterThan(accuracy(truth) - 0.06)
  })

  it('a grid CRF with a symmetric pairwise type recovers its planted weights; the loopy exact inference is a junction tree', () => {
    const model = gridCrf(['a', 'b'], ['f'])
    const oracle = graphOracle(model)
    expect(oracle.plan(gridInput(attributes(1, 6, ['f']), 2, 3))).toMatchObject({ inference: 'junctionTree' })
    const truth = Float64Array.from([0.2, -0.2, 1.2, -1.2, 0.4, 0.8, 0.9])
    expect(truth.length).toBe(model.layout.size)
    const inputs = Array.from({ length: 150 }, (_, s) => gridInput(attributes(300 + s, 6, ['f']), 2, 3))
    const data = sample(oracle, truth, inputs, 2, 8)
    const w = minimise((v) => nll(oracle, data, v, 0.01), new Float64Array(model.layout.size))
    const held = Array.from({ length: 30 }, (_, s) => gridInput(attributes(800 + s, 6, ['f']), 2, 3))
    const gap = marginalGap(oracle, w, truth, held)
    expect(gap).toBeLessThan(0.04)
    expect(marginalGap(oracle, new Float64Array(model.layout.size), truth, held)).toBeGreaterThan(3 * gap)
    // the same data trained with loopy belief propagation lands near the exact fit
    const bp = minimise(
      (v) => nll(graphOracle(model, { inference: 'bp', bp: { damping: 0.2 } }), data, v, 0.01),
      new Float64Array(model.layout.size),
    )
    expect(marginalGap(oracle, bp, w, held)).toBeLessThan(0.03)
  })
})

describe('skip chains and higher-order chains', () => {
  it('a skip chain labels a position by the repeat of its token elsewhere; a plain chain cannot', () => {
    // Repeated tokens share their label. The first occurrence of a token shows its label clearly; the second is blank.
    const make = (seed: number): { skip: GraphInput; plain: GraphInput; y: string[] } => {
      const u = toFlat(normal(key(seed), [8]))
      const y = Array.from({ length: 8 }, (_, i) => ((u[i] ?? 0) > 0 ? 'p' : 'q'))
      y[5] = y[1] ?? 'p'
      const items = y.map((l, i) => (i === 5 ? {} : { [l === 'p' ? 'isP' : 'isQ']: 1 }))
      const keys = items.map((_, i) => (i === 1 || i === 5 ? 'token' : undefined))
      return { skip: skipChainInput(items, { keys }), plain: skipChainInput(items), y }
    }
    const train = Array.from({ length: 60 }, (_, s) => make(s))
    const test = Array.from({ length: 80 }, (_, s) => make(1000 + s))
    const model = skipChainCrf(['p', 'q'], ['isP', 'isQ'])
    const oracle = graphOracle(model)
    const accuracyAtBlank = (inputOf: (d: ReturnType<typeof make>) => GraphInput): number => {
      const w = minimise(
        (v) =>
          nll(
            oracle,
            train.map((d) => ({ x: inputOf(d), y: d.y })),
            v,
            0.01,
          ),
        new Float64Array(model.layout.size),
      )
      return test.filter((d) => oracle.decode(w, inputOf(d)).y[5] === d.y[5]).length / test.length
    }
    const skip = accuracyAtBlank((d) => d.skip)
    const plain = accuracyAtBlank((d) => d.plain)
    expect(skip).toBeGreaterThan(0.95)
    expect(plain).toBeLessThan(0.75)
  })

  it('a first-order chain has the parameters of the linear-chain CRF and the same log-likelihood', () => {
    const items = nodes(5, 41).map((n) => ({ w1: n['f'] ?? 0, w2: n['g'] ?? 0 }))
    const linear = linearChainCrf(['a', 'b', 'c'], ['w1', 'w2'])
    const model = higherOrderChainCrf(['a', 'b', 'c'], ['w1', 'w2'], 1, { bias: false })
    expect(model.layout.size).toBe(linear.stateWeights.length + linear.transitionWeights.length)
    const w = randomVector(42, model.layout.size, 0.8)
    const weighted = { ...linear, stateWeights: w.slice(0, 6), transitionWeights: w.slice(6) }
    const y = ['a', 'c', 'c', 'b', 'a']
    const e = graphOracle(model).expectations(w, higherOrderChainInput(items, 1))
    expect(crfScore(weighted, items, y) - crfLogLikelihood(weighted, items, y)).toBeCloseTo(e.logPartition, 9)
  })

  it('a second-order chain with zero trigram weights is the first-order chain, and its trigram weights change what it scores', () => {
    const items = nodes(6, 43)
    const first = higherOrderChainCrf(['a', 'b', 'c'], ['f', 'g'], 1)
    const second = higherOrderChainCrf(['a', 'b', 'c'], ['f', 'g'], 2)
    const w1 = randomVector(44, first.layout.size, 0.8)
    const w2 = new Float64Array(second.layout.size)
    w2.set(w1)
    const one = graphOracle(first).expectations(w1, higherOrderChainInput(items, 1))
    const two = graphOracle(second).expectations(w2, higherOrderChainInput(items, 2))
    expect(two.logPartition).toBeCloseTo(one.logPartition, 10)
    for (let i = 0; i < w1.length; i++) expect(two.counts[i]).toBeCloseTo(one.counts[i] ?? 0, 10)
    expect(Array.from(two.counts.slice(w1.length)).every((c) => c >= 0)).toBe(true)
    // the trigram counts are the marginals of the label trigrams: they sum to the number of windows
    expect(Array.from(two.counts.slice(w1.length)).reduce((a, b) => a + b, 0)).toBeCloseTo(4, 10)
    w2[w1.length + 5] = 2
    expect(graphOracle(second).expectations(w2, higherOrderChainInput(items, 2)).logPartition).toBeGreaterThan(
      one.logPartition,
    )
  })

  it('a second-order chain learns a rule on the last two labels that a first-order chain cannot', () => {
    // y_t = y_{t-1} xor y_{t-2}, with the first two labels shown. No evidence anywhere else.
    const make = (seed: number): { x: GraphInput; y: string[] } => {
      const u = toFlat(normal(key(seed), [2]))
      const y = [(u[0] ?? 0) > 0 ? 1 : 0, (u[1] ?? 0) > 0 ? 1 : 0]
      for (let t = 2; t < 7; t++) y.push((y[t - 1] ?? 0) ^ (y[t - 2] ?? 0))
      const items = y.map((l, t) => (t < 2 ? { [`shown${l}`]: 1 } : {}))
      return { x: higherOrderChainInput(items, 2), y: y.map(String) }
    }
    const data = Array.from({ length: 40 }, (_, s) => make(s))
    const fit = (order: number): number => {
      const model = higherOrderChainCrf(['0', '1'], ['shown0', 'shown1'], order)
      const oracle = graphOracle(model)
      const d = data.map(({ x, y }) => ({ x: order === 2 ? x : { ...x, cliques: [] }, y }))
      const w = minimise((v) => nll(oracle, d, v, 0.001), new Float64Array(model.layout.size))
      return nll(oracle, d, w).value
    }
    expect(fit(2)).toBeLessThan(0.1 * fit(1))
  })
})

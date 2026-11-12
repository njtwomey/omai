import { chainDiagnostics, modelRun, modelRunFromTape } from 'omai-engine/exposition'
import { compileGraph, defaultOperators, epMethod, gaussianPrior, graphStructure } from 'omai-engine/message-passing'
import { defineModel, toStructure } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { type SceneNode, structure } from 'omai-engine/scene'
import { session } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import { texError } from '../maths/tex.ts'
import { chosenQuantities, rankView, traceView } from './chain-diagnostics.ts'
import {
  messageTex,
  modelRunConvergence,
  modelRunDecorate,
  modelRunFrames,
  modelRunOverlay,
  modelRunScene,
  modelRunStatus,
} from './model-run.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))
const texts = (nodes: readonly SceneNode[]): string[] =>
  flatten(nodes).flatMap((n) => (n.kind === 'text' ? [n.text] : []))

const pair = structure(
  'model-graph',
  [
    { id: 'a', label: 'a', role: 'latent' },
    { id: 'b', label: 'b', role: 'observed' },
  ],
  [{ from: 'a', to: 'b' }],
)
const run = modelRun({
  structure: pair,
  status: 'converged',
  series: [
    {
      name: 'change',
      label: 'largest change of a belief',
      steps: [1, 2, 3],
      values: [1, 0.1, 0.001],
      direction: 'lower',
    },
    {
      name: 'evidence',
      label: 'log evidence estimate',
      steps: [1, 2, 3],
      values: [-3, -2.2, -2.1],
      direction: 'higher',
    },
    { name: 'improper', label: 'improper updates', steps: [1, 3], values: [0, 2] },
  ],
  steps: [
    {
      step: 1,
      nodes: { a: { mean: 0, sd: 2, text: '0 ± 2' }, b: { observed: 1, text: 'observed 1' } },
      messages: [{ from: 'b', to: 'a', text: 'N(μ = 1, σ² = 4)' }],
    },
    { step: 2, nodes: { a: { mean: 0.8, sd: 1, text: '0.8 ± 1' } } },
    { step: 3, nodes: { a: { mean: 0.9, sd: 0.4, text: '0.9 ± 0.4' } } },
  ],
})

describe('a message as TeX', () => {
  it('sets the forms the engine writes, and nothing else', () => {
    expect(messageTex('N(μ = 0.83, σ² = 2.1)')).toBe('\\Gauss(\\mu = 0.83, \\sigma^2 = 2.1)')
    expect(messageTex('N(μ = -1.5, σ² = 1e-7)')).toBe('\\Gauss(\\mu = -1.5, \\sigma^2 = 1 \\times 10^{-7})')
    expect(messageTex('Beta(2, 3)')).toBe('\\Beta(2, 3)')
    expect(messageTex('Gamma(2, rate 0.5)')).toBe('\\GammaD(2, \\text{rate } 0.5)')
    expect(messageTex('p = 0.7')).toBe('p = 0.7')
    expect(messageTex('improper (precision -1)')).toBe('\\text{improper (precision } -1 \\text{)}')
    expect(messageTex('N(μ = x, σ² = 1)')).toBeUndefined()
    expect(messageTex('[0.167, 0.167, … (6)]')).toBeUndefined()
    expect(messageTex('η = [1, 2]')).toBeUndefined()
  })
  it('gives TeX that parses with the shared macros', () => {
    for (const text of [
      'N(μ = 0.83, σ² = 2.1)',
      'Beta(2, 3)',
      'Gamma(2, rate 0.5)',
      'p = 0.7',
      'improper (precision -1)',
    ]) {
      const tex = messageTex(text) as string
      expect(texError(tex), text).toBeUndefined()
    }
  })
})

describe('the overlay of a model run', () => {
  it('labels the edge a message went along, as the structure draws it, and badges the nodes with beliefs', () => {
    const o = modelRunOverlay(run, 1)
    expect(o.edges).toEqual([{ from: 'a', to: 'b', label: 'N(μ = 1, σ² = 4)' }])
    expect([...(o.active ?? [])].sort()).toEqual(['a', 'b'])
    expect(o.badges?.['a']).toEqual({ μ: '0', σ: '2' })
    expect(o.badges?.['b']).toEqual({ obs: 1 })
    // As TeX when asked: the diagram reads a line between dollar signs as TeX
    expect(modelRunOverlay(run, 1, { tex: true }).edges).toEqual([
      { from: 'a', to: 'b', label: '$\\Gauss(\\mu = 1, \\sigma^2 = 4)$' },
    ])
    expect(modelRunFrames(run, { tex: true })[0]?.edges?.[0]?.label).toBe('$\\Gauss(\\mu = 1, \\sigma^2 = 4)$')
    // A later step has no messages of its own, and keeps the beliefs said before.
    const later = modelRunOverlay(run, 2)
    expect(later.edges).toEqual([])
    expect(later.badges?.['b']).toEqual({ obs: 1 })
    expect(later.badges?.['a']).toEqual({ μ: '0.8', σ: '1' })
    expect(modelRunFrames(run)).toHaveLength(3)
  })

  it('writes improper beliefs, probabilities, vectors and R-hat as what they are', () => {
    const s = structure(
      'g',
      ['i', 'p', 'k', 'v', 'r'].map((id) => ({ id, label: id, role: 'latent' as const })),
      [],
    )
    const r = modelRun({
      structure: s,
      status: 'running',
      series: [],
      steps: [
        {
          step: 1,
          nodes: {
            i: { improper: true, text: 'improper' },
            p: { mean: 0.7, sd: 0.4, probs: [0.3, 0.7], text: 'p = 0.7' },
            k: { probs: [0.1, 0.2, 0.7], text: 'state 2' },
            v: {
              mean: 2,
              elements: [
                { name: 'v[0]', mean: 1, sd: 1 },
                { name: 'v[1]', mean: 3, sd: 1 },
              ],
              text: '2 coords',
            },
            r: { mean: 1, sd: 0.1, rhat: 1.0123, text: '1 ± 0.1' },
          },
        },
      ],
    })
    const b = modelRunOverlay(r).badges
    expect(b).toEqual({
      i: { improper: 'yes' },
      p: { p: '0.7' },
      k: { state: 2 },
      v: { μ̄: '2' },
      r: { μ: '1', σ: '0.1', R̂: '1.01' },
    })
  })
})

describe('the glyphs and the scene', () => {
  it('draws an interval glyph for a belief on a span the whole run keeps, bars for probabilities, nothing for observed', () => {
    const at = (step: number) => modelRunDecorate(run, step)
    const box = { x: 50, y: 20, width: 40, height: 40 }
    const node = { id: 'a', label: 'a', role: 'latent' as const }
    const first = at(1)(node, box) ?? []
    const last = at(3)(node, box) ?? []
    expect(first.map((n) => n.kind)).toEqual(['line', 'line', 'circle'])
    const bar = (nodes: readonly SceneNode[]): number => {
      const l = nodes[1]
      return l?.kind === 'line' ? Math.abs(l.x2 - l.x1) : 0
    }
    // The belief narrows over the run: the bar is shorter at the end, on the same scale.
    expect(bar(last)).toBeLessThan(bar(first))
    expect(at(1)({ id: 'b', label: 'b', role: 'observed' }, box)).toBeUndefined()
    expect(at(1)(node, { ...box, width: 12 })).toBeUndefined()
    const bars = modelRunDecorate(
      modelRun({
        structure: pair,
        status: 'running',
        series: [],
        steps: [{ step: 1, nodes: { a: { probs: [0.25, 0.75], text: 'p' } } }],
      }),
    )(node, box)
    expect(bars?.map((n) => n.kind)).toEqual(['rect', 'rect'])
  })

  it('draws the diagram with the beliefs in the nodes and the message on the edge, and keeps every node in place across steps', () => {
    const first = modelRunScene(run, 1)
    expect(texts(first.children).join('|')).toContain('μ 0 · σ 2')
    expect(texts(first.children)).toContain('N(μ = 1, σ² = 4)')
    const last = modelRunScene(run, 3)
    expect(texts(last.children).join('|')).toContain('μ 0.9 · σ 0.4')
    expect(texts(last.children)).not.toContain('N(μ = 1, σ² = 4)')
    const where = (s: ReturnType<typeof modelRunScene>, id: string): unknown => {
      const g = flatten(s.children).find((n) => n.kind === 'group' && n.id === id)
      const c = g?.kind === 'group' ? g.children.find((x) => x.kind === 'circle') : undefined
      return c?.kind === 'circle' ? [c.cx, c.cy, c.r] : undefined
    }
    expect(where(first, 'a')).toEqual(where(last, 'a'))
    expect(modelRunScene(run, 3, { glyphs: false }).children.length).toBeLessThanOrEqual(last.children.length)
  })

  it('says where the run is: the step, running or how it ended, improper updates, the change and the evidence', () => {
    expect(modelRunStatus(run)).toBe('step 3 · converged · change 0.001 · evidence -2.1 · improper updates 2')
    expect(modelRunStatus(run, 2)).toBe('step 2 · running · change 0.1 · evidence -2.2 · improper updates 0')
  })

  it('charts the change and the evidence with a cursor, and nothing when there are no series', () => {
    const o = modelRunConvergence(run, 2)
    expect(o).toBeDefined()
    expect(((o ?? {}).grid as unknown[]).length).toBe(2)
    expect(JSON.stringify(o)).toContain('markLine')
    expect(modelRunConvergence(modelRun({ structure: pair, status: 'running', series: [], steps: [] }))).toBeUndefined()
  })
})

describe('on a real run', () => {
  const graph = compileGraph(
    {
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
      ],
      factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
    },
    defaultOperators(),
  )
  it('draws the factor graph of an EP run with its messages on the factor edges', () => {
    const st = graphStructure({
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
      ],
      factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
    })
    const r = modelRunFromTape(st, { run: session(epMethod(graph), { steps: 6, mode: 'eager', logs: 'all' }).run() })
    const s = modelRunScene(r)
    expect(texts(modelRunScene(r, 0).children).some((t) => t.startsWith('N(μ = '))).toBe(false)
    const early = modelRunScene(r, r.steps.find((x) => (x.messages ?? []).length > 0)?.step)
    expect(texts(early.children).some((t) => t.startsWith('N(μ = '))).toBe(true)
    expect(texts(s.children).join('|')).toContain('μ 0.564 · σ 0.826')
  })

  it('draws a plate diagram of a spec the same way', () => {
    const model = defineModel('m', (b) => {
      const y = b.data('y', { dims: ['row'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), y)
    })
    const r = modelRun({
      structure: toStructure(model),
      status: 'running',
      series: [],
      steps: [{ step: 1, nodes: { mu: { mean: 1, sd: 0.5, text: '1 ± 0.5' } } }],
    })
    const ids = flatten(modelRunScene(r).children).flatMap((n) =>
      n.kind === 'group' && n.id !== undefined ? [n.id] : [],
    )
    expect(ids).toEqual(expect.arrayContaining(['mu', 'y']))
  })
})

describe('chain charts', () => {
  const draws = {
    a: [
      [1, 2, 1, 2, 1, 2],
      [2, 1, 2, 1, 2, 1],
    ],
    b: [
      [1, 1, 1, 1, 1, 1],
      [3, 3, 3, 3, 3, 3],
    ],
  }
  const rec = chainDiagnostics(draws, {
    bins: 3,
    summary: [{ site: 'b', name: 'b', rhat: 2.5 }],
    divergent: [
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ],
  })

  it('chooses the quantities with the worst R-hat first, or the ones named', () => {
    expect(chosenQuantities(rec).map((q) => q.name)).toEqual(['b', 'a'])
    expect(chosenQuantities(rec, { max: 1 }).map((q) => q.name)).toEqual(['b'])
    expect(chosenQuantities(rec, { quantities: ['a'] }).map((q) => q.name)).toEqual(['a'])
  })

  it('draws a trace panel per quantity, a line per chain, and the divergent transitions as points', () => {
    const o = traceView(rec)
    expect((o.grid as unknown[]).length).toBe(2)
    const series = (o.series as { name?: string }[]).map((s) => s.name)
    expect(series.filter((n) => n === 'chain 0')).toHaveLength(2)
    expect(series).toContain('divergent')
  })

  it('draws a rank panel per quantity: a stepped line per chain and the expected count', () => {
    const o = rankView(rec, { quantities: ['b'] })
    expect((o.grid as unknown[]).length).toBe(1)
    const chains = (o.series as { name?: string; data?: unknown[] }[]).filter((s) => s.name?.startsWith('chain'))
    expect(chains.map((s) => s.name)).toEqual(['chain 0', 'chain 1'])
    // Three bins, two points each: a plateau per bin.
    expect(chains[0]?.data).toHaveLength(6)
    expect(JSON.stringify(o)).toContain('markLine')
  })
})

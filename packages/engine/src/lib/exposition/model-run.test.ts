// Inference on a model graph: the record's checks, the readers of real tapes (message passing, a sampler, anytime
// snapshots), the natural parameters of a message as words, and the diagnostics of chains.
import { describe, expect, it } from 'vitest'
import { MODELS } from '../../testing/probprog.ts'
import {
  compileGraph,
  defaultOperators,
  epAlgorithm,
  epMethod,
  gaussianPrior,
  graphStructure,
  mpSnapshot,
  runAnytime as runAnytimeMp,
} from '../message-passing/index.ts'
import { nutsMethod } from '../probprog/inference/nuts-method.ts'
import * as d from '../probprog/model/dists/index.ts'
import { defineModel, toStructure } from '../probprog/model/index.ts'
import { structure } from '../scene/index.ts'
import { Log, session } from '../trace/index.ts'
import { chainDiagnostics, chainsFromTape, rankCounts } from './chain-diagnostics.ts'
import { modelRun, modelRunAt } from './model-run.ts'
import { describeMessage, modelRunFromTape, nodeStateOf, nodeStatesOfRows } from './model-run-tape.ts'

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

describe('modelRun', () => {
  const s = structure(
    'model-graph',
    [
      { id: 'a', label: 'a', role: 'latent' },
      { id: 'b', label: 'b', role: 'latent' },
    ],
    [{ from: 'a', to: 'b' }],
  )

  it('checks its ids, its order and its series against the structure', () => {
    const ok = { structure: s, status: 'running' as const, series: [], steps: [{ step: 1, nodes: {} }] }
    expect(modelRun(ok).kind).toBe('modelRun')
    expect(() => modelRun({ ...ok, steps: [{ step: 1, nodes: { c: { text: 'c' } } }] })).toThrow(/node c/)
    expect(() =>
      modelRun({
        ...ok,
        steps: [
          { step: 2, nodes: {} },
          { step: 1, nodes: {} },
        ],
      }),
    ).toThrow(/out of order/)
    expect(() =>
      modelRun({ ...ok, steps: [{ step: 1, nodes: {}, messages: [{ from: 'a', to: 'z', text: '' }] }] }),
    ).toThrow(/unknown nodes/)
    expect(() => modelRun({ ...ok, series: [{ name: 'change', label: 'c', steps: [1], values: [] }] })).toThrow(
      /series change/,
    )
  })

  it('reads the state at a step as everything said so far, the messages of that step only', () => {
    const r = modelRun({
      structure: s,
      status: 'running',
      series: [],
      steps: [
        { step: 1, nodes: { a: { mean: 1, text: '1' } }, messages: [{ from: 'a', to: 'b', text: 'm1' }] },
        { step: 3, nodes: { b: { mean: 2, text: '2' } } },
      ],
    })
    expect(Object.keys(modelRunAt(r, 3)?.nodes ?? {})).toEqual(['a', 'b'])
    expect(modelRunAt(r, 2)?.messages.map((m) => m.text)).toEqual(['m1'])
    expect(modelRunAt(r, 3)?.messages).toEqual([])
    expect(modelRunAt(r)?.step).toBe(3)
    expect(modelRunAt(modelRun({ structure: s, status: 'running', series: [], steps: [] }))).toBeUndefined()
  })
})

describe('describeMessage and the marginals', () => {
  it('reads natural parameters by family, and says when a message is improper', () => {
    expect(describeMessage('gaussian', [0.5, 2])).toBe('N(μ = 0.25, σ² = 0.5)')
    expect(describeMessage('gaussian', [1, -1])).toBe('improper (precision -1)')
    expect(describeMessage('gamma', [2, -4])).toBe('Gamma(3, rate 4)')
    expect(describeMessage('beta', [1, 2])).toBe('Beta(2, 3)')
    expect(describeMessage('bernoulli', [0])).toBe('p = 0.5')
    expect(describeMessage('discrete', [0, 0])).toBe('[0.5, 0.5]')
    expect(describeMessage('dirichlet', [0, 1])).toBe('Dir(1, 2)')
    expect(describeMessage('gaussian-mv', [0, 0, 2, 0, 0, 4])).toBe('N-2(precision diag 2, 4)')
    expect(describeMessage('weird', [1, 2])).toBe('η = [1, 2]')
  })

  it('reads a marginal of every family the engine reports', () => {
    expect(nodeStateOf({ family: 'gaussian', mean: 0.5, variance: 4 })).toMatchObject({
      mean: 0.5,
      sd: 2,
      text: '0.5 ± 2',
    })
    expect(nodeStateOf({ family: 'bernoulli', p: 0.25 })).toMatchObject({ probs: [0.75, 0.25], text: 'p = 0.25' })
    expect(nodeStateOf({ family: 'discrete', probs: [0.2, 0.8] })?.text).toBe('state 1 (0.8)')
    expect(nodeStateOf({ family: 'gaussian', observed: 2 })).toMatchObject({ observed: 2, text: 'observed 2' })
    expect(nodeStateOf({ family: 'gaussian', precision: 0, h: 0, improper: true })).toMatchObject({ improper: true })
    expect(nodeStateOf({ family: 'gaussian-mv', mean: [1, 3], variance: [4, 9] })?.elements).toEqual([
      { name: '[0]', mean: 1, sd: 2 },
      { name: '[1]', mean: 3, sd: 3 },
    ])
    expect(nodeStateOf('nope')).toBeUndefined()
    expect(nodeStateOf({ family: 'gaussian' })).toBeUndefined()
  })

  it('reads the rows of a posterior summary per site and per coordinate, with the worst R-hat and least ESS', () => {
    const states = nodeStatesOfRows([
      { site: 'mu', name: 'mu', mean: 1, sd: 0.5, lower: 0, upper: 2, rhat: 1.01, ess: 400 },
      { site: 'w', name: 'w[0]', mean: 1, sd: 0.5, rhat: 1.0, ess: 300 },
      { site: 'w', name: 'w[1]', mean: 3, sd: 0.5, rhat: 1.2, ess: 90 },
      { nonsense: true },
    ])
    expect(states['mu']).toMatchObject({ text: '1 ± 0.5', lower: 0, upper: 2, rhat: 1.01 })
    expect(states['w']).toMatchObject({ text: '2 coords, mean 2', rhat: 1.2, ess: 90 })
    expect(states['w[1]']?.text).toBe('3 ± 0.5')
  })
})

describe('modelRunFromTape: message passing', () => {
  it('reads an EP session: beliefs by variable, messages in words between factor and variable, the series', () => {
    const m = epMethod(graph)
    const s = session(m, { steps: 6, mode: 'eager', logs: 'all' })
    const r = modelRunFromTape(
      graphStructure({
        variables: graph.variables.map((v) => ({
          name: v.name,
          family: v.family.spec,
          ...(v.observed === undefined ? {} : { observed: v.observed }),
        })),
        factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
      }),
      { run: s.run() },
    )
    expect(r.status).toBe('converged')
    expect(r.method).toBe('mp')
    const last = modelRunAt(r)
    expect(last?.nodes['x']?.text).toBe('0.564 ± 0.826')
    expect(last?.nodes['y']?.text).toBe('observed 1')
    const first = r.steps.find((x) => (x.messages ?? []).length > 0)
    expect(first?.messages?.map((x) => [x.from, x.to, x.operator])).toEqual([
      ['f0', 'x', 'prior'],
      ['f1', 'x', 'probit'],
    ])
    expect(first?.messages?.[0]?.text).toBe('N(μ = 0, σ² = 1)')
    expect(r.series.map((x) => x.name)).toEqual(expect.arrayContaining(['change', 'evidence', 'improper']))
    expect(r.series.find((x) => x.name === 'evidence')?.direction).toBe('higher')
  })

  it('draws a plate diagram of a spec: a factor id is its site, so a message lands on an edge of the picture', () => {
    const model = defineModel('m', (b) => {
      const y = b.data('y', { dims: ['row'] })
      b.observe('y', d.normal(b.latent('mu', d.normal(0, 1)), 1), y)
    })
    const st = toStructure(model)
    const events = [
      {
        type: 'message',
        step: 1,
        kind: 'mp/marginals',
        payload: { mu: { family: 'gaussian', mean: 1, variance: 0.25 } },
      },
      {
        type: 'message',
        step: 1,
        kind: 'mp/message',
        from: 'factor:y',
        to: 'mu',
        payload: { operator: 'gaussian', eta: [2, 4] },
      },
    ]
    const r = modelRunFromTape(st, { run: { events } })
    expect(r.steps[0]?.messages).toEqual([{ from: 'y', to: 'mu', text: 'N(μ = 0.5, σ² = 0.25)', operator: 'gaussian' }])
    expect(r.steps[0]?.nodes['mu']?.text).toBe('1 ± 0.5')
  })

  it('accepts a run in progress and anytime snapshots alone, and a prefix of its own', () => {
    const st = graphStructure({
      variables: [
        { name: 'x', family: { kind: 'gaussian' } },
        { name: 'y', family: { kind: 'bernoulli' }, observed: 1 },
      ],
      factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
    })
    const s = session(epMethod(graph), { steps: 6, mode: 'eager', logs: 'all' })
    const events = s.run().events
    // The tape cut short: no end event, so the run is still going.
    const partial = modelRunFromTape(st, {
      run: { events: events.filter((e) => e.type !== 'end' && ('step' in e ? e.step : 0) < 1) },
    })
    expect(partial.status).toBe('running')
    // Snapshots as runAnytime hands them out, with no tape at all.
    const seen: { step: number; snapshot: unknown }[] = []
    let now = 0
    runAnytimeMp(graph, epAlgorithm(graph), {
      budgetMs: 1000,
      clock: () => (now += 1),
      onSnapshot: (snap, i) => seen.push({ step: i.t, snapshot: snap }),
    })
    const fromSnaps = modelRunFromTape(st, { snapshots: seen })
    expect(fromSnaps.status).toBe('running')
    expect(modelRunAt(fromSnaps)?.nodes['x']?.text).toBe('0.564 ± 0.826')
    expect(fromSnaps.series.map((x) => x.name)).toContain('change')
    expect(mpSnapshot(graph, epAlgorithm(graph).init(undefined as never)).sweeps).toBe(0)
    const other = modelRunFromTape(
      st,
      {
        run: {
          events: [
            {
              type: 'message',
              step: 2,
              kind: 'ep2/marginals',
              payload: { x: { family: 'gaussian', mean: 1, variance: 1 } },
            },
          ],
        },
      },
      { prefix: 'ep2/' },
    )
    expect(other.method).toBe('ep2')
    expect(other.steps).toHaveLength(1)
  })

  it('thins to the most steps asked, keeping the last', () => {
    const st = structure('g', [{ id: 'x', label: 'x', role: 'latent' }], [])
    const events = Array.from({ length: 100 }, (_, i) => ({
      type: 'message',
      step: i,
      kind: 'mp/marginals',
      payload: { x: { family: 'gaussian', mean: i, variance: 1 } },
    }))
    const r = modelRunFromTape(st, { run: { events } }, { maxSteps: 10 })
    expect(r.steps.length).toBeLessThanOrEqual(11)
    expect(r.steps.at(-1)?.step).toBe(99)
  })
})

describe('modelRunFromTape: a sampler', () => {
  it('reads the summary rows of nuts/snapshot onto the sites of the spec, with R-hat and ESS', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    const s = session(m, {
      config: { numChains: 2, numWarmup: 30, numDraws: 30 },
      steps: 200,
      mode: 'eager',
      logs: { 'mcmc/snapshot': Log() },
    })
    const st = structure('model-graph', [{ id: 'mu', label: 'mu', role: 'latent' }], [])
    const r = modelRunFromTape(st, { run: s.run() })
    const mu = modelRunAt(r)?.nodes['mu']
    expect(mu?.mean).toBeTypeOf('number')
    expect(mu?.sd).toBeGreaterThan(0)
    expect(mu?.rhat).toBeTypeOf('number')
    expect(r.method).toBe('mcmc')
  })
})

describe('chain diagnostics', () => {
  it('ranks the pooled draws into parts per chain: flat for chains that agree, a slope for chains that do not', () => {
    expect(
      rankCounts(
        [
          [1, 3, 5, 7],
          [2, 4, 6, 8],
        ],
        2,
      ),
    ).toEqual([
      [2, 2],
      [2, 2],
    ])
    expect(
      rankCounts(
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
        2,
      ),
    ).toEqual([
      [4, 0],
      [0, 4],
    ])
    // Ties share their mean rank.
    expect(
      rankCounts(
        [
          [1, 1],
          [1, 1],
        ],
        2,
      ),
    ).toEqual([
      [0, 2],
      [0, 2],
    ])
  })

  it('makes the record from draws, a summary and the divergent flags, and refuses ragged or empty chains', () => {
    const r = chainDiagnostics(
      {
        'w[0]': [
          [1, 2, 3, 4],
          [2, 3, 4, 5],
        ],
      },
      {
        bins: 2,
        summary: [{ site: 'w', name: 'w[0]', rhat: 1.02, ess: 120 }],
        divergent: [
          [0, 1, 0, 0],
          [0, 0, 0, 0],
        ],
      },
    )
    const q = r.quantities[0]
    expect(q).toMatchObject({ name: 'w[0]', site: 'w', mean: 3, rhat: 1.02, ess: 120 })
    expect(q?.ranks.counts).toHaveLength(2)
    expect(r.divergences).toEqual([{ chain: 0, draw: 1 }])
    expect(chainDiagnostics({ a: [[1, 2, 3, 4, 5, 6]] }, { maxDraws: 3 }).quantities[0]?.chains[0]).toEqual([1, 3, 6])
    expect(() => chainDiagnostics({ a: [[1, 2], [1]] })).toThrow(/chain 1/)
    expect(() => chainDiagnostics({ a: [] })).toThrow(/no draws/)
  })

  it('reads the draws of a sampler tape without the warm-up, by coordinate and chain', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    const s = session(m, {
      config: { numChains: 2, numWarmup: 20, numDraws: 15 },
      steps: 100,
      mode: 'eager',
      logs: {
        'mcmc/position': Log({ detail: 'full' }),
        'mcmc/divergent': Log({ detail: 'full' }),
        'mcmc/stage': Log(),
      },
    })
    const { draws, divergent } = chainsFromTape(s.run())
    expect(Object.keys(draws)).toEqual(['position[0]'])
    expect(draws['position[0]']).toHaveLength(2)
    expect(draws['position[0]']?.[0]).toHaveLength(15)
    expect(divergent[0]).toHaveLength(15)
    const rec = chainDiagnostics(draws, { divergent })
    expect(rec.quantities[0]?.ranks.counts[0]?.reduce((a, b) => a + b, 0)).toBe(15)
  })
})

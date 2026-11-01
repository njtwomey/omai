import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { undirectedGraph } from './graphs.ts'
import {
  cardinalitySearchContract,
  eliminationOrder,
  eliminationOrderAlgorithm,
  eliminationOrderContract,
  inducedWidth,
  isChordal,
  maximumCardinalitySearch,
  maximumCardinalitySearchAlgorithm,
  treewidth,
  triangulate,
} from './triangulation.ts'
import { eliminationHeuristics } from './triangulation-options.ts'

const grid = undirectedGraph([
  ['a', 'b'],
  ['b', 'c'],
  ['d', 'e'],
  ['e', 'f'],
  ['g', 'h'],
  ['h', 'i'],
  ['a', 'd'],
  ['d', 'g'],
  ['b', 'e'],
  ['e', 'h'],
  ['c', 'f'],
  ['f', 'i'],
])
const square = undirectedGraph([
  ['a', 'b'],
  ['b', 'c'],
  ['c', 'd'],
  ['d', 'a'],
])
const star = undirectedGraph([
  ['h', 'a'],
  ['h', 'b'],
  ['h', 'c'],
  ['a', 'b'],
])
const card = { a: 2, b: 3, c: 4, d: 2, e: 3, f: 2, g: 5, h: 2, i: 2 }

// The numbers the functions gave before they were run over the algorithms.
const oldOrders: Record<string, Record<string, string[]>> = {
  grid: {
    minDegree: ['a', 'c', 'g', 'i', 'b', 'd', 'e', 'f', 'h'],
    minFill: ['a', 'c', 'b', 'g', 'd', 'e', 'f', 'h', 'i'],
    minWeight: ['g', 'i', 'a', 'c', 'b', 'e', 'd', 'f', 'h'],
    weightedMinFill: ['g', 'i', 'h', 'a', 'c', 'b', 'd', 'e', 'f'],
  },
  square: {
    minDegree: ['a', 'b', 'c', 'd'],
    minFill: ['a', 'b', 'c', 'd'],
    minWeight: ['a', 'c', 'b', 'd'],
    weightedMinFill: ['a', 'b', 'c', 'd'],
  },
  star: {
    minDegree: ['c', 'a', 'b', 'h'],
    minFill: ['a', 'b', 'c', 'h'],
    minWeight: ['c', 'b', 'a', 'h'],
    weightedMinFill: ['a', 'b', 'c', 'h'],
  },
}
const graphs = { grid, square, star }

describe('eliminationOrderAlgorithm', () => {
  it('is eliminationOrder: the same orders as before, for every criterion', () => {
    for (const [name, g] of Object.entries(graphs)) {
      for (const [h, order] of Object.entries(oldOrders[name] ?? {})) {
        const vs = [...g.keys()]
        expect(eliminationOrder(g, vs, h as 'minFill', card)).toEqual(order)
        const s = runAlgorithm(eliminationOrderAlgorithm(g, vs, { heuristic: h as 'minFill', cardinality: card }), {
          maxSteps: vs.length,
        })
        expect(s.state.order).toEqual(order)
        expect(s.status).toBe('converged')
      }
    }
  })

  it('triangulates as before: fill-ins, cliques and the order', () => {
    const t = triangulate(grid)
    expect(t.fillIns).toEqual([
      ['b', 'd'],
      ['b', 'f'],
      ['d', 'f'],
      ['d', 'h'],
      ['f', 'h'],
    ])
    expect(t.cliques).toEqual([
      ['a', 'b', 'd'],
      ['b', 'c', 'f'],
      ['b', 'd', 'e', 'f'],
      ['d', 'g', 'h'],
      ['d', 'e', 'f', 'h'],
      ['f', 'h', 'i'],
    ])
    expect(triangulate(square, ['a', 'b', 'c', 'd']).fillIns).toEqual([['b', 'd']])
    expect(triangulate(star).cliques).toEqual([
      ['a', 'b', 'h'],
      ['c', 'h'],
    ])
    expect(maximumCardinalitySearch(grid)).toEqual(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    expect(maximumCardinalitySearch(star)).toEqual(['a', 'b', 'h', 'c'])
    expect(treewidth(grid, 'heuristic')).toBe(3)
  })

  it('orders a subset of the variables and leaves the others in the graph', () => {
    expect(eliminationOrder(grid, ['e', 'a', 'i'], 'minFill', card)).toEqual(['a', 'i', 'e'])
  })

  it('resumes from a JSON checkpoint to an identical end', () => {
    const vs = [...grid.keys()]
    const alg = eliminationOrderAlgorithm(grid, vs)
    const whole = runAlgorithm(alg, { maxSteps: vs.length })
    const half = runAlgorithm(alg, { maxSteps: 4 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    expect(stateFromJson(stored)).toEqual(half.state)
    const rest = runAlgorithm(alg, { maxSteps: vs.length - 4, start: { state: stored, step: 4, serialised: true } })
    expect(rest.steps).toBe(vs.length)
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
  })

  it('logs its contract: a vertex message and the cost, fill and width per step', () => {
    const rec = recorder()
    const vs = [...square.keys()]
    runAlgorithm(eliminationOrderAlgorithm(square, vs), {
      maxSteps: 4,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, messages: true },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('order/fill')).toEqual([1, 0, 0, 0])
    expect(metric('order/width')).toEqual([2, 2, 2, 2])
    expect(metric('order/cost')).toHaveLength(4)
    const messages = rec.events.flatMap((e) => (e.type === 'message' && e.kind === 'order/vertex' ? [e.payload] : []))
    expect(messages.map((m) => (m as { vertex: string }).vertex)).toEqual(['a', 'b', 'c', 'd'])
    expect(Object.keys(eliminationOrderContract)).toEqual(['order/vertex', 'order/cost', 'order/fill', 'order/width'])
  })

  it('the width of the run is the induced width of its order, and a perfect order adds no fill-in', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: 0, max: 6 }), fc.integer({ min: 0, max: 6 })), { maxLength: 14 }),
        fc.constantFrom('minFill', 'minDegree', 'minWeight', 'weightedMinFill'),
        (pairs, h) => {
          const g = undirectedGraph(
            pairs.filter(([a, b]) => a !== b).map(([a, b]): [string, string] => [`v${a}`, `v${b}`]),
            Array.from({ length: 7 }, (_, i) => `v${i}`),
          )
          const vs = [...g.keys()]
          const s = runAlgorithm(eliminationOrderAlgorithm(g, vs, { heuristic: h }), { maxSteps: vs.length }).state
          const chordal = triangulate(g, s.order)
          return s.width === inducedWidth(g, s.order) && isChordal(chordal.graph) && s.order.length === vs.length
        },
      ),
      { numRuns: 60 },
    )
  })

  it('runs every option of the heuristic table, and follows a given order', () => {
    for (const name of eliminationHeuristics.names) {
      const cost = eliminationHeuristics.make({ kind: name })
      const alg = eliminationOrderAlgorithm(grid, [...grid.keys()], { heuristic: cost, cardinality: card })
      expect(runAlgorithm(alg, { maxSteps: 9 }).state.order).toEqual(oldOrders['grid']?.[name])
    }
    const fixed = eliminationOrderAlgorithm(square, [...square.keys()], { order: ['d', 'c', 'b', 'a'] })
    expect(runAlgorithm(fixed, { maxSteps: 4 }).state.order).toEqual(['d', 'c', 'b', 'a'])
    expect(() => eliminationOrderAlgorithm(square, ['a', 'b'], { order: ['a', 'a'] })).toThrow(/order/)
  })
})

describe('maximumCardinalitySearchAlgorithm', () => {
  it('visits as the function did, resumes from JSON, and logs the weight of each visit', () => {
    const alg = maximumCardinalitySearchAlgorithm(grid)
    const whole = runAlgorithm(alg, { maxSteps: 9 }).state
    expect(whole.order).toEqual(maximumCardinalitySearch(grid))
    const half = runAlgorithm(alg, { maxSteps: 3 })
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
    const rest = runAlgorithm(alg, { maxSteps: 6, start: { state: stored, step: 3, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 9, tracer: rec, policy: { ...policies.minimal, metrics: true, messages: true } })
    const weights = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'mcs/weight' ? [e.value] : []))
    expect(weights).toHaveLength(9)
    expect(weights[0]).toBe(0)
    expect(Object.keys(cardinalitySearchContract)).toEqual(['mcs/vertex', 'mcs/weight'])
  })

  it('reversed, it is a perfect elimination order exactly when the graph is chordal', () => {
    expect(isChordal(star)).toBe(true)
    expect(isChordal(square)).toBe(false)
    expect(triangulate(star, maximumCardinalitySearch(star).reverse()).fillIns).toEqual([])
  })
})

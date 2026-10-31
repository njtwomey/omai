import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../../random/index.ts'
import {
  nullTracer,
  recorder,
  runAlgorithm,
  startRun,
  stateFromJson,
  stateToJson,
  stepContext,
  tracePolicy,
} from '../../trace/index.ts'
import {
  type NeatGenome,
  type NeatState,
  neatAlgorithm,
  neatNetwork,
  solvesXor,
  XOR_CASES,
  xorFitness,
} from './neat.ts'

// A hand-built network for XOR: two hidden units computing OR and AND, and an output that is OR and not AND.
const xorByHand: NeatGenome = {
  nodes: [
    { id: 0, kind: 'input' },
    { id: 1, kind: 'input' },
    { id: 2, kind: 'bias' },
    { id: 3, kind: 'output' },
    { id: 4, kind: 'hidden' },
    { id: 5, kind: 'hidden' },
  ],
  connections: [
    { innovation: 0, from: 0, to: 4, weight: 4, enabled: true },
    { innovation: 1, from: 1, to: 4, weight: 4, enabled: true },
    { innovation: 2, from: 2, to: 4, weight: -2, enabled: true },
    { innovation: 3, from: 0, to: 5, weight: 4, enabled: true },
    { innovation: 4, from: 1, to: 5, weight: 4, enabled: true },
    { innovation: 5, from: 2, to: 5, weight: -6, enabled: true },
    { innovation: 6, from: 4, to: 3, weight: 6, enabled: true },
    { innovation: 7, from: 5, to: 3, weight: -12, enabled: true },
    { innovation: 8, from: 2, to: 3, weight: -3, enabled: true },
  ],
  fitness: 0,
  species: 0,
}

const cyclic = (g: NeatGenome): boolean => {
  const state = new Map<number, number>()
  const visit = (n: number): boolean => {
    if (state.get(n) === 1) return true
    if (state.get(n) === 2) return false
    state.set(n, 1)
    const found = g.connections.some((c) => c.from === n && visit(c.to))
    state.set(n, 2)
    return found
  }
  return g.nodes.some((n) => visit(n.id))
}

describe('NEAT networks', () => {
  it('evaluates a network with a hidden layer that no network without one can match', () => {
    const net = neatNetwork(xorByHand)
    expect(solvesXor(net)).toBe(true)
    expect(xorFitness(net)).toBeGreaterThan(15)
    for (const [x, target] of XOR_CASES) expect(Math.abs((net(x)[0] ?? 0) - target)).toBeLessThan(0.1)
    // A node without inputs gives one half, and the bias is a constant 1.
    const lonely = neatNetwork({ ...xorByHand, connections: [] })
    expect(lonely([1, 1])).toEqual([0.5])
    const initial = neatAlgorithm({ populationSize: 40 }).init(key(0))
    expect(initial.population.some((g) => solvesXor(neatNetwork(g)))).toBe(false)
    expect(initial.population.every((g) => g.connections.length === 3 && g.nodes.length === 4)).toBe(true)
  })
})

describe('NEAT evolution', () => {
  it('keeps one innovation number per structural change and never builds a cycle', () => {
    const alg = neatAlgorithm({ populationSize: 60, target: Number.POSITIVE_INFINITY })
    const result = runAlgorithm(alg, { maxSteps: 60, seed: 11 })
    const byEnds = new Map<string, Set<number>>()
    for (const g of result.state.population) {
      expect(cyclic(g)).toBe(false)
      for (const c of g.connections)
        byEnds.set(`${c.from}>${c.to}`, (byEnds.get(`${c.from}>${c.to}`) ?? new Set()).add(c.innovation))
    }
    for (const [ends, numbers] of byEnds) expect(numbers.size, ends).toBe(1)
    // The same innovation number never names two different connections either.
    const byNumber = new Map<number, string>()
    for (const g of result.state.population)
      for (const c of g.connections)
        expect(byNumber.set(c.innovation, `${c.from}>${c.to}`).get(c.innovation)).toBe(`${c.from}>${c.to}`)
    expect(result.state.population.some((g) => g.nodes.length > 4)).toBe(true) // complexification happened
    expect(result.state.nextInnovation).toBeGreaterThan(3)
  })

  it('solves XOR with a hidden unit in most runs, speciating along the way', () => {
    let solved = 0
    for (let seed = 0; seed < 8; seed++) {
      const result = runAlgorithm(neatAlgorithm({}), { maxSteps: 300, seed })
      expect(result.state.representatives.length, `seed ${seed}`).toBeGreaterThan(1)
      if (result.state.solved) {
        solved += 1
        expect(solvesXor(neatNetwork(result.state.best)), `seed ${seed}`).toBe(true)
        expect(
          result.state.best.nodes.some((n) => n.kind === 'hidden'),
          `seed ${seed}`,
        ).toBe(true)
      }
    }
    expect(solved).toBeGreaterThanOrEqual(4)
  }, 60_000)

  it('records the generations on the tape and has a state that survives a round trip through JSON', () => {
    const rec = recorder()
    const alg = neatAlgorithm({ populationSize: 40, target: Number.POSITIVE_INFINITY })
    const result = runAlgorithm(alg, {
      maxSteps: 25,
      seed: 2,
      tracer: rec,
      policy: tracePolicy({
        metrics: ['generation/best', 'generation/nodes'],
        tensors: true,
        roles: ['output'],
        messages: true,
      }),
    })
    const names = rec
      .run()
      .events.filter((e) => e.type === 'metric')
      .map((e) => (e as { name: string }).name)
    expect(names.filter((n) => n === 'generation/best')).toHaveLength(25)
    expect(names).toContain('generation/nodes')
    const messages = rec.run().events.filter((e) => e.type === 'message') as unknown as { kind: string }[]
    expect(messages.some((m) => m.kind === 'structure/added')).toBe(true)
    const back = stateFromJson(stateToJson(result.state)) as NeatState
    expect(back.generation).toBe(25)
    expect(back.population).toEqual(result.state.population)
    expect(back.innovations).toEqual(result.state.innovations)
    // Resuming from the stored state continues the same run.
    const again = alg.step(
      back,
      stepContext(25, foldIn(key(2), 25), startRun(nullTracer, { name: 'neat', seed: 2, config: {} })),
    )
    expect(again.generation).toBe(26)
    expect(() => neatAlgorithm({ populationSize: 2 })).toThrow()
  })
})

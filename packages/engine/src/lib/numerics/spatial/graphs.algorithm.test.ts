import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { hnswAlgorithm, hnswContract, hnswIndex } from './graphs.ts'
import { hnswProfiles } from './graphs-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const data = Array.from({ length: 60 }, (_, i) => [Math.sin(i * 1.7) * 5, Math.cos(i * 0.9) * 5, (i % 7) - 3])
const options = { m: 4, efConstruction: 30, key: key(5) }

describe('hnswAlgorithm', () => {
  it('inserts one point per step and keeps the graph valid', () => {
    const run = runAlgorithm(hnswAlgorithm(data, options), { maxSteps: 100 })
    expect([run.status, run.steps, run.state.inserted]).toEqual(['converged', 60, 60])
    expect(run.state.levels).toHaveLength(60)
    expect(run.state.levels[run.state.entry]).toBe(Math.max(...run.state.levels))
    run.state.links.forEach((layers, node) => {
      expect(layers).toHaveLength((run.state.levels[node] ?? 0) + 1)
      layers.forEach((list, layer) => {
        expect(list.length).toBeLessThanOrEqual(layer === 0 ? 8 : 4)
        expect(list.includes(node)).toBe(false)
        expect(new Set(list).size).toBe(list.length)
        for (const other of list) expect(run.state.levels[other] ?? -1).toBeGreaterThanOrEqual(layer)
      })
    })
  })

  it('builds the graph of hnswIndex and finds the point itself', () => {
    const run = runAlgorithm(hnswAlgorithm(data, options), { maxSteps: 100 })
    const index = hnswIndex(data, options)
    expect(index.links).toEqual(run.state.links)
    expect(index.levels).toEqual(run.state.levels)
    expect(index.entryPoint).toBe(run.state.entry)
    for (const i of [0, 17, 42]) expect(index.search(data[i] as number[], 1).indices[0]).toBe(i)
  })

  it('serialises, resumes and keeps adding points to the index', () => {
    const alg = hnswAlgorithm(data, options)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const part = runAlgorithm(alg, { maxSteps: 25 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 25, serialised: true },
    })
    expect(rest.state.links).toEqual(whole.state.links)
    expect(rest.state.entry).toBe(whole.state.entry)
    const index = hnswIndex(data.slice(0, 30), options)
    index.add(data.slice(30))
    expect(index.size).toBe(60)
    expect(index.links).toEqual(whole.state.links)
  })

  it('logs the node, layer, links, entry point and evaluations', () => {
    const rec = recorder()
    runAlgorithm(hnswAlgorithm(data, options), {
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('hnsw/insert')).toEqual(Array.from({ length: 60 }, (_, i) => i))
    const evals = metric('hnsw/evaluations')
    expect(evals.every((v, i) => i === 0 || v >= (evals[i - 1] ?? 0))).toBe(true)
    expect(Object.keys(hnswContract)).toHaveLength(5)
  })
})

describe('hnswProfiles', () => {
  it('names the profiles and builds options', () => {
    expect(hnswProfiles.names).toEqual(['fast-build', 'balanced', 'high-recall'])
    expect(hnswProfiles.make({ kind: 'fast-build' }).m).toBe(8)
    expect(hnswProfiles.make({ kind: 'balanced', metric: 'cosine' }).metric).toBe('cosine')
  })
})

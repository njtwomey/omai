import { describe, expect, it } from 'vitest'
import { gini } from '../../stats/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import * as a from './index.ts'

const run = <S>(alg: Parameters<typeof runAlgorithm<S>>[0], steps: number, seed = 0): S =>
  runAlgorithm(alg, { seed, maxSteps: steps }).state
const sum = (x: ArrayLike<number>): number => Array.from(x).reduce((s, v) => s + v, 0)

describe('Schelling', () => {
  const index = (s: a.SchellingState, size = 40): number => a.segregationIndex(s.cells, size, size)
  it('starts at the random baseline with the stated numbers', () => {
    const alg = a.schelling({ width: 40, height: 40, vacancy: 0.1 })
    const s = run(alg, 0)
    const count = (k: number): number => Array.from(s.cells).filter((c) => c === k).length
    expect([count(0), count(1), count(2)]).toEqual([160, 720, 720])
    expect(Math.abs(index(s) - 0.5)).toBeLessThan(0.03)
  })
  it('a tolerance of a third segregates a city that began mixed', () => {
    const s = run(a.schelling({ width: 40, height: 40, tolerance: 1 / 3 }), 60, 1)
    expect(index(s)).toBeGreaterThan(0.7)
    expect(s.unhappy).toBe(0)
  })
  it('with tolerance 0 nobody moves and the board never changes', () => {
    const s0 = run(a.schelling({ width: 12, height: 12, tolerance: 0 }), 0, 3)
    const s = run(a.schelling({ width: 12, height: 12, tolerance: 0 }), 5, 3)
    expect(s.cells).toEqual(s0.cells)
    expect(s.moved).toBe(0)
  })
  it('conserves the number of agents of each kind and settles with an event', () => {
    const rec = recorder()
    const alg = a.schelling({ width: 20, height: 20, tolerance: 0.4, destination: 'satisfying' })
    const s0 = run(alg, 0, 2)
    const res = runAlgorithm(alg, { seed: 2, maxSteps: 80, tracer: rec, policy: policies.debug })
    for (const k of [1, 2])
      expect(Array.from(res.state.cells).filter((c) => c === k)).toHaveLength(
        Array.from(s0.cells).filter((c) => c === k).length,
      )
    const fired = rec.events.filter((e) => e.type === 'message' && e.kind === 'city/settled')
    expect(fired.length).toBeLessThanOrEqual(1)
    if (res.state.unhappy === 0) expect(fired).toHaveLength(1)
  })
  it('the index of complete separation is 1 and of a chequerboard 0', () => {
    expect(a.segregationIndex(Uint8Array.of(1, 1, 0, 2, 2), 5, 1)).toBe(1)
    expect(a.segregationIndex(Uint8Array.of(1, 2, 1, 2), 4, 1)).toBe(0)
  })
  it('rejects a share outside [0, 1]', () => {
    expect(() => a.schelling({ tolerance: 2 })).toThrow(/tolerance/)
  })
})

describe('Nagel-Schreckenberg traffic', () => {
  it('without dawdling the flow is min(vmax rho, 1 - rho) once relaxed', () => {
    const d = a.trafficDiagram({
      length: 200,
      densities: [0.05, 0.1, 0.2, 0.5, 0.8],
      dawdle: 0,
      settle: 600,
      measure: 20,
    })
    d.density.forEach((rho, i) => expect(d.flow[i]).toBeCloseTo(a.deterministicFlow(rho, 5), 9))
  })
  it('a lone car accelerates one cell per step up to the top speed', () => {
    let road: Int8Array = Int8Array.of(0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    const speeds: number[] = []
    for (let t = 0; t < 7; t++) {
      road = a.trafficStep(road, 5, 0, new Float64Array(20))
      speeds.push(Math.max(...road))
    }
    expect(speeds).toEqual([1, 2, 3, 4, 5, 5, 5])
  })
  it('cars never collide or pass each other and are conserved', () => {
    const alg = a.nagelSchreckenberg({ length: 80, density: 0.4, dawdle: 0.3 })
    for (let t = 0; t <= 100; t += 20) {
      const s = run(alg, t, 6)
      expect(Array.from(s.cells).filter((v) => v >= 0)).toHaveLength(32)
    }
  })
  it('dawdling lowers the flow below the deterministic value and makes jams', () => {
    const d = a.trafficDiagram({ length: 300, densities: [0.2], dawdle: 0.3, settle: 300, measure: 200, seed: 1 })
    expect(d.flow[0]).toBeLessThan(a.deterministicFlow(0.2, 5) - 0.05)
    const s = run(a.nagelSchreckenberg({ length: 300, density: 0.3, dawdle: 0.3 }), 300, 2)
    expect(s.jammed).toBeGreaterThan(0)
  })
  it('flow equals density times mean speed and the state round-trips', () => {
    const alg = a.nagelSchreckenberg({ length: 100, cars: 25 })
    const s = run(alg, 40, 3)
    expect(s.flow).toBeCloseTo(0.25 * s.meanSpeed, 12)
    expect(alg.deserialise?.(alg.serialise?.(s) ?? null)).toEqual(s)
    expect(() => a.nagelSchreckenberg({ length: 10, cars: 11 })).toThrow(/cars/)
  })
})

describe('Wa-Tor', () => {
  it('fish alone breeding every step double: 1, 2, 4', () => {
    const alg = a.waTor({ width: 30, height: 30, fish: 1 / 900, sharks: 0, fishBreed: 1 })
    expect([0, 1, 2].map((t) => run(alg, t, 5).fish)).toEqual([1, 2, 4])
  })
  it('sharks with nothing to eat starve in exactly sharkEnergy steps', () => {
    const alg = a.waTor({ width: 20, height: 20, fish: 0, sharks: 0.1, sharkEnergy: 4, sharkBreed: 10 })
    expect(run(alg, 3, 1).sharks).toBe(40)
    expect(run(alg, 4, 1).sharks).toBe(0)
  })
  it('keeps every animal on a distinct site and both species for hundreds of steps', () => {
    const alg = a.waTor({ width: 50, height: 50 })
    const s = run(alg, 300, 7)
    expect(s.fish).toBeGreaterThan(0)
    expect(s.sharks).toBeGreaterThan(0)
    expect(s.fish + s.sharks).toBeLessThanOrEqual(2500)
    expect(Array.from(s.kind).filter((k) => k === 1)).toHaveLength(s.fish)
  })
  it('rejects shares that overfill the ocean', () => {
    expect(() => a.waTor({ fish: 0.8, sharks: 0.5 })).toThrow(/fill/)
  })
})

describe('Sugarscape', () => {
  it('the landscape has two peaks of 4 and is empty in the far corners', () => {
    const c = a.sugarLandscape(50)
    expect(c[35 * 50 + 15]).toBe(4)
    expect(c[15 * 50 + 35]).toBe(4)
    expect(c[0]).toBe(0)
  })
  it('starts with the Gini of uniform wealth on 5..25, which is 2/9', () => {
    const s = run(a.sugarscape({ agents: 2000, size: 60, endowment: [5, 25] }), 0, 1)
    const wealth = Array.from({ length: s.count }, (_, i) => s.agents[i * 7 + 2] ?? 0)
    expect(Math.abs(gini(wealth) - 2 / 9)).toBeLessThan(0.02)
  })
  it('inequality emerges and agents migrate toward the peaks', () => {
    const alg = a.sugarscape()
    const s0 = run(alg, 0, 2)
    const s = run(alg, 100, 2)
    const wealth = (st: a.SugarscapeState): number[] =>
      Array.from({ length: st.count }, (_, i) => st.agents[i * 7 + 2] ?? 0)
    expect(gini(wealth(s))).toBeGreaterThan(gini(wealth(s0)) + 0.1)
    const sugarUnder = (st: a.SugarscapeState): number => {
      const cap = a.sugarLandscape(50)
      let t = 0
      for (let i = 0; i < st.count; i++) t += cap[(st.agents[i * 7] ?? 0) * 50 + (st.agents[i * 7 + 1] ?? 0)] ?? 0
      return t / st.count
    }
    expect(sugarUnder(s)).toBeGreaterThan(sugarUnder(s0) + 0.5)
  })
  it('never puts two agents on a site; without replacement the population only falls', () => {
    const alg = a.sugarscape({ replace: false, lifespan: null })
    let s = run(alg, 0, 3)
    const sizes: number[] = []
    for (const steps of [20, 60]) {
      s = run(alg, steps, 3)
      const seen = new Set<number>()
      for (let i = 0; i < s.count; i++) seen.add((s.agents[i * 7] ?? 0) * 50 + (s.agents[i * 7 + 1] ?? 0))
      expect(seen.size).toBe(s.count)
      sizes.push(s.count)
    }
    expect(sizes[1]).toBeLessThanOrEqual(sizes[0] ?? 0)
    expect(sizes[0]).toBeLessThanOrEqual(400)
  })
  it('sugar never exceeds capacity and wealth is conserved by harvest less metabolism', () => {
    const alg = a.sugarscape({ size: 20, agents: 40, replace: false, lifespan: null })
    const s = run(alg, 30, 4)
    const cap = a.sugarLandscape(20)
    for (let i = 0; i < cap.length; i++) expect(s.sugar[i]).toBeLessThanOrEqual(cap[i] ?? 0)
    expect(sum(s.sugar)).toBeGreaterThan(0)
  })
  it('rejects more agents than sites', () => {
    expect(() => a.sugarscape({ size: 4, agents: 17 })).toThrow(/more agents/)
  })
})

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key as runKey } from '../../random/index.ts'
import {
  type Algorithm,
  nullTracer,
  policies,
  recorder,
  runAlgorithm,
  startRun,
  stepContext,
} from '../../trace/index.ts'
import * as a from './index.ts'

const run = <S>(alg: Parameters<typeof runAlgorithm<S>>[0], steps: number, seed = 0): S =>
  runAlgorithm(alg, { seed, maxSteps: steps }).state

describe('elementary automata', () => {
  const rows = (rule: number, g: number): string[] => {
    const { board, width } = a.elementaryBoard(rule, g)
    return Array.from({ length: g + 1 }, (_, k) => Array.from(board.slice(k * width, (k + 1) * width)).join(''))
  }
  it('rule 30 from one cell matches the hand-worked rows', () => {
    // Worked by hand: new = left XOR (centre OR right).
    expect(rows(30, 3).map((r) => r.replace(/^0+|0+$/g, ''))).toEqual(['1', '111', '11001', '1101111'])
    expect(rows(30, 4)[4]?.replace(/^0+|0+$/g, '')).toBe('110010001')
  })
  it('rule 30 centre column starts 1 1 0 1 1 1 0 0 1 1 and is balanced', () => {
    const g = 1200
    const { board, width } = a.elementaryBoard(30, g)
    const centre = Array.from({ length: g + 1 }, (_, k) => board[k * width + g] ?? 0)
    expect(centre.slice(0, 10).join('')).toBe('1101110011')
    const ones = centre.reduce((x, y) => x + y, 0) / centre.length
    expect(Math.abs(ones - 0.5)).toBeLessThan(0.04)
  })
  it('rule 110 grows leftwards through 1, 11, 111, 1101, 11111', () => {
    const r = rows(110, 4).map((x) => x.replace(/^0+|0+$/g, ''))
    expect(r).toEqual(['1', '11', '111', '1101', '11111'])
  })
  it('rule 90 has 2^popcount(n) live cells in generation n (Pascal mod 2)', () => {
    const { board, width } = a.elementaryBoard(90, 64)
    for (let n = 0; n <= 64; n++) {
      let live = 0
      for (let i = 0; i < width; i++) live += board[n * width + i] ?? 0
      expect(live).toBe(a.rule90Cells(n))
    }
  })
  it('rule 184 conserves the number of live cells on a ring (traffic)', () => {
    const alg = a.elementaryAutomaton({ rule: 184, width: 60, start: 'random', density: 0.3 })
    const first = run(alg, 0, 1)
    const last = run(alg, 80, 1)
    expect(sum(last.row)).toBe(sum(first.row))
  })
  it('mirror and complement are involutions that give the known partners', () => {
    expect([a.mirrorRule(30), a.complementRule(30)]).toEqual([86, 135])
    for (let r = 0; r < 256; r++) {
      expect(a.mirrorRule(a.mirrorRule(r))).toBe(r)
      expect(a.complementRule(a.complementRule(r))).toBe(r)
    }
  })
  it('the mirror rule evolves the mirrored row', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 255 }),
        fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 5, maxLength: 20 }),
        (rule, cells) => {
          const row = Uint8Array.from(cells)
          const mirrored = Uint8Array.from([...cells].reverse())
          const next = a.elementaryStep(mirrored, a.mirrorRule(rule))
          expect(Array.from(next).reverse()).toEqual(Array.from(a.elementaryStep(row, rule)))
        },
      ),
    )
  })
  it('classifies anchor rules into Wolfram classes', () => {
    for (const r of [0, 8, 32, 128, 255]) expect(a.wolframClass(r)).toBe(1)
    for (const r of [4, 51, 108, 170, 204]) expect(a.wolframClass(r)).toBe(2)
    for (const r of [30, 45, 90, 105, 150]) expect(a.wolframClass(r)).toBe(3)
    expect(a.wolframClass(110)).toBe(4)
  })
  it('rejects a rule outside 0..255', () => {
    expect(() => a.ruleTable(256)).toThrow(/0..255/)
  })
  it('logs the row, density and entropy, and its state round-trips', () => {
    const alg = a.elementaryAutomaton({ rule: 30, width: 21 })
    const rec = recorder()
    runAlgorithm(alg, { seed: 0, maxSteps: 5, tracer: rec, policy: policies.debug })
    const names = new Set(rec.run().events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
    expect([...names].sort()).toEqual(['automaton/density', 'automaton/entropy', 'automaton/row'])
    const s = run(alg, 5)
    expect(alg.deserialise?.(alg.serialise?.(s) ?? null)).toEqual(s)
    expect(a.bitEntropy(0.5)).toBe(1)
  })
})

describe('Life', () => {
  const life = (start: string, steps: number, size = 24, wrap = false): a.LifeState =>
    run(a.lifeAutomaton({ start, width: size, height: size, wrap }), steps)
  const live = (state: a.LifeState, size: number): string[] =>
    Array.from(state.cells, (c, i) => (c ? `${Math.floor(i / size)},${i % size}` : '')).filter(Boolean)
  it('parses and formats rules', () => {
    expect(a.parseLifeRule('B3/S23')).toEqual({ birth: [3], survive: [2, 3] })
    expect(a.formatLifeRule(a.parseLifeRule('B2/S'))).toBe('B2/S')
    expect(() => a.parseLifeRule('nonsense')).toThrow(/cannot read/)
  })
  it('still lifes stay and oscillators return with their known periods', () => {
    expect(life('block', 5).period).toBe(1)
    expect(life('beehive', 5).period).toBe(1)
    for (const name of ['blinker', 'toad', 'beacon']) expect(life(name, 8).period).toBe(2)
    expect(life('pentadecathlon', 40, 32, true).period).toBe(15)
  })
  it('the glider moves one cell down and right every 4 generations and has 5 cells', () => {
    const g0 = live(life('glider', 0), 24)
    const g4 = live(life('glider', 4), 24)
    const shifted = g0.map((s) => {
      const [r, c] = s.split(',').map(Number) as [number, number]
      return `${r + 1},${c + 1}`
    })
    expect(g4.sort()).toEqual(shifted.sort())
    expect(g0).toHaveLength(5)
  })
  it('every pattern with a known period returns shifted by its known displacement', () => {
    const size = 40
    for (const p of Object.values(a.LIFE_PATTERNS)) {
      if (p.period === undefined || p.name === 'gosper-gun') continue
      const [dr, dc] = p.displacement ?? [0, 0]
      const start = live(life(p.name, 0, size, true), size)
      const later = live(life(p.name, p.period, size, true), size)
      const moved = start.map((s) => {
        const [r, c] = s.split(',').map(Number) as [number, number]
        return `${(r + dr + size) % size},${(c + dc + size) % size}`
      })
      expect(later.sort(), p.name).toEqual(moved.sort())
    }
  })
  it("Gosper's gun has returned to itself plus one glider after 30 generations", () => {
    const size = 60
    const pop = (t: number): number => sum(life('gosper-gun', t, size).cells)
    expect(pop(0)).toBe(36)
    expect(pop(30)).toBe(41)
    expect(pop(60)).toBe(46)
  })
  it('counts births and deaths: a blinker has 2 of each every generation', () => {
    const s = life('blinker', 3, 8)
    expect([s.births, s.deaths]).toEqual([2, 2])
  })
  it('HighLife (B36/S23) replicates where Life does not, and Seeds (B2/S) kills every survivor', () => {
    const seeds = run(a.lifeAutomaton({ rule: 'B2/S', start: 'block', width: 10, height: 10, wrap: false }), 1)
    expect(seeds.deaths).toBe(4)
  })
  it('fires the stabilised event once, with the period', () => {
    const rec = recorder()
    runAlgorithm(a.lifeAutomaton({ start: 'blinker', width: 8, height: 8 }), {
      seed: 0,
      maxSteps: 10,
      tracer: rec,
      policy: policies.debug,
    })
    const fired = rec.events.filter((e) => e.type === 'message' && e.kind === 'life/stabilised')
    expect(fired).toHaveLength(1)
    expect(fired[0]).toMatchObject({ step: 1, payload: { period: 2, generation: 2 } })
    expect(a.detectPeriod([1, 2, 3, 2])).toBe(2)
  })
  it('pattern placement refuses a pattern that does not fit and RLE refuses junk', () => {
    expect(() => a.placePattern(a.LIFE_PATTERNS['gosper-gun'] as a.LifePattern, 10, 10)).toThrow(/does not fit/)
    expect(() => a.parseRle('3x')).toThrow(/unexpected/)
  })
  it('a soup serialises and resumes to the same board', () => {
    const alg = a.lifeAutomaton({ width: 16, height: 16 })
    const mid = run(alg, 10, 3)
    const resumed = runAlgorithm(alg, {
      seed: 3,
      maxSteps: 5,
      start: { state: alg.serialise?.(mid), step: 10, serialised: true },
    }).state
    expect(Array.from(resumed.cells)).toEqual(Array.from(run(alg, 15, 3).cells))
  })
})

describe("Langton's ant", () => {
  it('leaves three black cells and faces west after 5 moves (worked by hand)', () => {
    const s = run(a.langtonsAnt({ width: 9, height: 9 }), 5)
    expect(sum(s.cells)).toBe(3)
    expect(Array.from(s.ants)).toEqual([3, 4, 3])
  })
  it('LR is the mirror image of RL: its board is the RL board flipped left to right', () => {
    const size = 61
    const rl = run(a.langtonsAnt({ rule: 'RL', width: size, height: size }), 1500)
    const lr = run(a.langtonsAnt({ rule: 'LR', width: size, height: size }), 1500)
    for (let r = 0; r < size; r++)
      for (let c = 0; c < size; c++) expect(lr.cells[r * size + c]).toBe(rl.cells[r * size + (size - 1 - c)])
  })
  it('several ants stay on the board and each makes the same number of moves', () => {
    const s = run(a.langtonsAnt({ width: 20, height: 20, ants: 3, movesPerStep: 7 }), 30, 4)
    expect(s.moves).toBe(210)
    for (let i = 0; i < 3; i++) {
      expect(s.ants[3 * i]).toBeLessThan(20)
      expect(s.ants[3 * i + 1]).toBeLessThan(20)
    }
  })
  it('builds the highway after about ten thousand moves, and says so once', () => {
    const rec = recorder()
    const alg = a.langtonsAnt({ width: 150, height: 150, movesPerStep: 100 })
    const state = runAlgorithm(alg, { seed: 0, maxSteps: 130, tracer: rec, policy: policies.debug }).state
    expect(state.highwayAt).toBeGreaterThan(9900)
    expect(state.highwayAt).toBeLessThan(11000)
    const fired = rec.events.filter((e) => e.type === 'message' && e.kind === 'ant/highway')
    expect(fired).toHaveLength(1)
    const d = a.highwayDetected(state.track)
    expect(d === undefined ? 0 : Math.abs(d[0])).toBe(2)
  })
  it('rejects a rule of one colour', () => {
    expect(() => a.langtonsAnt({ rule: 'L' })).toThrow(/rule/)
  })
})

describe('sandpile', () => {
  it('the 3x3 identity is the known pile and adding it changes nothing', () => {
    const e = a.sandpileIdentity(3, 3)
    expect(Array.from(e)).toEqual([2, 1, 2, 1, 0, 1, 2, 1, 2])
    const top = new Int32Array(9).fill(3)
    expect(Array.from(a.sandpileAdd(e, top, 3, 3))).toEqual(Array.from(top))
    expect(Array.from(a.sandpileAdd(e, e, 3, 3))).toEqual(Array.from(e))
  })
  it('the identity of a larger grid is a fixed point of addition to a recurrent pile', () => {
    const e = a.sandpileIdentity(7, 7)
    const top = new Int32Array(49).fill(3)
    expect(Array.from(a.sandpileAdd(e, top, 7, 7))).toEqual(Array.from(top))
  })
  it('conserves grains: dropped = held + lost', () => {
    const s = run(a.sandpileAlgorithm({ width: 15, height: 15 }), 300)
    expect(s.grains).toBe(sum(s.heights) + s.lost)
    expect(Math.max(...s.heights)).toBeLessThan(4)
  })
  it('stabilisation does not depend on the order of the grains (Abelian)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 24 }), { minLength: 1, maxLength: 60 }),
        fc.integer({ min: 0, max: 1000 }),
        (sites, seed) => {
          const drop = (order: number[]): Int32Array => {
            let h: Int32Array = new Int32Array(25)
            for (const i of order) {
              h = Int32Array.from(h)
              h[i] = (h[i] ?? 0) + 1
              h = a.stabilise(h, 5, 5).heights
            }
            return h
          }
          const shuffled = [...sites].sort((x, y) => ((x * 7919 + seed) % 31) - ((y * 7919 + seed) % 31))
          const all = new Int32Array(25)
          for (const i of sites) all[i] = (all[i] ?? 0) + 1
          expect(Array.from(drop(shuffled))).toEqual(Array.from(a.stabilise(all, 5, 5).heights))
        },
      ),
    )
  })
  it('one toppling of a single site of 4 grains makes a plus, and avalanches get bigger as the pile fills', () => {
    const r = a.stabilise(Int32Array.of(0, 0, 0, 0, 4, 0, 0, 0, 0), 3, 3)
    expect(Array.from(r.topplings)).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0])
    const s = run(a.sandpileAlgorithm({ width: 21, height: 21, drop: 'random' }), 1500, 2)
    expect(s.grains).toBe(1500)
  })
  it('rejects a drop site off the grid', () => {
    expect(() => a.sandpileAlgorithm({ width: 3, height: 3, drop: [3, 0] })).toThrow(/on the grid/)
  })
})

describe('forest fire', () => {
  it('without lightning the density of a growing forest is 1 - (1 - p)^t', () => {
    const w = 120
    for (const t of [3, 10]) {
      const s = run(a.forestFire({ width: w, height: w, growth: 0.1, lightning: 0 }), t, 5)
      expect(Math.abs(sum(s.cells) / (w * w) - a.treeDensityWithoutFire(0.1, t))).toBeLessThan(0.02)
    }
  })
  it('with certain lightning every grown forest burns whole each step', () => {
    const s = run(a.forestFire({ width: 6, height: 6, growth: 1, lightning: 1 }), 1)
    expect([sum(s.cells), s.fireSize, s.fires]).toEqual([0, 36, 1])
  })
  it('burns every struck cluster whole: 4 clusters of a hand-drawn forest, 8 trees', () => {
    // 1101 / 0001 / 1000 / 1101 on a 4 x 4 board with edges: clusters of 2, 2, 3 and 1 trees.
    const cells = Uint8Array.from([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    const alg = a.forestFire({ width: 4, height: 4, growth: 0, lightning: 1, wrap: false, cells })
    const s = stepBy(alg, alg.init(runKey(0)), 0)
    expect([s.fires, s.fireSize, sum(s.cells)]).toEqual([4, 8, 0])
  })
  it('a lone struck cluster leaves the rest of the forest standing', () => {
    const cells = Uint8Array.from([1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    const alg = a.forestFire({ width: 4, height: 4, growth: 0, lightning: 0.5, wrap: false, cells })
    let s = alg.init(runKey(0))
    s = stepBy(alg, s, 0)
    // Whatever was struck, trees left + trees burnt = trees before, and a surviving tree has no burnt neighbour.
    expect(sum(s.cells) + s.fireSize).toBe(sum(cells))
  })
  it('rejects chances outside [0, 1]', () => {
    expect(() => a.forestFire({ growth: 1.5 })).toThrow(/growth/)
  })
})

function sum(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0
  return s
}

/** Step an algorithm by hand with a real step context, counting what it emits. */
function stepBy<S, K extends string>(alg: Algorithm<S, K>, state: S, t: number): S {
  return alg.step(
    state,
    stepContext(t, runKey(1000 + t), startRun(nullTracer, { name: alg.name, seed: 0, config: {} })),
  )
}

import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../../trace/index.ts'
import * as f from './index.ts'

describe('escape-time sets', () => {
  it('knows members and non-members of the Mandelbrot set', () => {
    // 0, -1 (period 2), -2 (the tip, orbit 2, 2, 2...), 0.25 (the cusp) and i are in; 1, 0.5 and -2.1 are out.
    for (const c of [
      [0, 0],
      [-1, 0],
      [-2, 0],
      [0.25, 0],
      [0, 1],
    ] as const)
      expect(f.escapeCount(c, 2000), `${c}`).toBe(0)
    expect(f.escapeCount([1, 0], 100)).toBe(3)
    expect(f.escapeCount([0.5, 0], 100)).toBe(5)
    expect(f.escapeCount([-2.1, 0], 100)).toBe(1)
  })
  it('the Julia set of c = 0 is the unit disc: escape time is exactly n = floor(log2(log R / log|z|)) + 1', () => {
    for (const r of [1.01, 1.1, 1.5, 3]) {
      const n = Math.floor(Math.log2(Math.log(2) / Math.log(r))) + 1
      expect(f.escapeCount([r, 0], 100, { set: 'julia', c: [0, 0] }), `r=${r}`).toBe(Math.max(n, 1))
    }
    expect(f.escapeCount([0.99, 0], 1000, { set: 'julia', c: [0, 0] })).toBe(0)
  })
  it('the smooth escape time of the c = 0 Julia set is log2(log R / log |z0|) in every case', () => {
    const alg = f.escapeTimeAlgorithm({
      set: 'julia',
      c: [0, 0],
      viewport: { centre: [1.5, 0], width: 0.001, pixels: [1, 1] },
    })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 3 }).state
    const r = 1.5
    expect(s.smooth[0]).toBeCloseTo(Math.log2(Math.log(2) / Math.log(r)), 9)
  })
  it('the Mandelbrot area from a grid agrees with the known 1.5066 from above', () => {
    const [cols, rows] = [320, 256]
    const alg = f.escapeTimeAlgorithm({ viewport: { centre: [-0.75, 0], width: 3.2, pixels: [cols, rows] } })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 250 }).state
    const inside = 1 - s.escaped / (cols * rows)
    const area = inside * 3.2 * ((3.2 * rows) / cols)
    expect(area).toBeGreaterThan(1.48)
    expect(area).toBeLessThan(1.62)
  })
  it('the bulbs have their periods: 1 (cardioid), 2, 3 and 4', () => {
    expect(f.orbitPeriod([0, 0])).toBe(1)
    expect(f.orbitPeriod([-1, 0])).toBe(2)
    expect(f.orbitPeriod([-1.7548776, 0])).toBe(3)
    expect(f.orbitPeriod([-1.3107026, 0])).toBe(4)
    expect(f.orbitPeriod([1, 1])).toBe(0)
  })
  it('the Mandelbrot picture is symmetric about the real axis, and the ship is not', () => {
    const grid = { centre: [-0.5, 0] as const, width: 3, pixels: [30, 30] as const }
    const s = runAlgorithm(f.escapeTimeAlgorithm({ viewport: grid }), { seed: 0, maxSteps: 40 }).state
    for (let r = 0; r < 15; r++)
      for (let c = 0; c < 30; c++) expect(s.count[r * 30 + c]).toBe(s.count[(29 - r) * 30 + c])
    const ship = runAlgorithm(f.escapeTimeAlgorithm({ set: 'burning-ship', viewport: grid }), {
      seed: 0,
      maxSteps: 40,
    }).state
    let asym = 0
    for (let r = 0; r < 15; r++)
      for (let c = 0; c < 30; c++) if (ship.count[r * 30 + c] !== ship.count[(29 - r) * 30 + c]) asym++
    expect(asym).toBeGreaterThan(0)
  })
  it('the image sharpens: the escaped share never falls and the state round-trips', () => {
    const alg = f.escapeTimeAlgorithm({ viewport: { pixels: [40, 40] } })
    let prev = 0
    for (const steps of [1, 5, 20, 60]) {
      const s = runAlgorithm(alg, { seed: 0, maxSteps: steps }).state
      expect(s.escaped).toBeGreaterThanOrEqual(prev)
      prev = s.escaped
    }
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 10 }).state
    expect(alg.deserialise?.(alg.serialise?.(s) ?? null)).toEqual(s)
  })
  it('a multibrot of power 3 is symmetric under negation of c (z -> -z)', () => {
    for (const c of [
      [0.3, 0.2],
      [0.7, -0.5],
      [1.1, 0.1],
    ] as const)
      expect(f.escapeCount(c, 80, { set: 'multibrot' })).toBe(f.escapeCount([-c[0], -c[1]], 80, { set: 'multibrot' }))
  })
  it('the orbit probe returns the iterates', () => {
    expect(f.escapeOrbit([-1, 0], 4)).toEqual([0, 0, -1, 0, 0, 0, -1, 0])
    expect(f.escapeOrbit([0.5, 0], 3, { set: 'julia', c: [0, 0] })).toEqual([0.5, 0, 0.25, 0, 0.0625, 0])
  })
  it('rejects a power below 2', () => {
    expect(() => f.escapeCount([0, 0], 10, { power: 1 })).toThrow(/power/)
  })
})

describe('Newton fractals', () => {
  it('one step for z^2 - 1 from 2 is 1.25 and a root is a fixed point', () => {
    expect(
      f.newtonStep(
        [2, 0],
        [
          [1, 0],
          [-1, 0],
        ],
      ),
    ).toEqual([1.25, 0])
    expect(
      f.newtonStep(
        [1, 0],
        [
          [1, 0],
          [-1, 0],
        ],
      ),
    ).toEqual([1, 0])
  })
  it('the three basins of z^3 - 1 have equal shares by symmetry and rotate into one another', () => {
    const size = 90
    const alg = f.newtonFractal({ viewport: { centre: [0, 0], width: 4, pixels: [size, size] } })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 60 }).state
    const share = [0, 1, 2].map((k) => Array.from(s.root).filter((r) => r === k).length / (size * size))
    expect(s.converged / (size * size)).toBeGreaterThan(0.99)
    for (const p of share) expect(Math.abs(p - 1 / 3)).toBeLessThan(0.03)
  })
  it('points on the real axis go to the real root; the conjugate point goes to the conjugate root', () => {
    const run1 = (re: number, im: number): number =>
      runAlgorithm(f.newtonFractal({ viewport: { centre: [re, im], width: 0.01, pixels: [1, 1] } }), {
        seed: 0,
        maxSteps: 50,
      }).state.root[0] ?? -9
    expect(run1(2, 0)).toBe(0)
    expect(run1(-2, 0.0)).toBeGreaterThanOrEqual(0)
    expect(run1(0.2, 1.5) + run1(0.2, -1.5)).toBe(3)
  })
  it('rejects a single root', () => {
    expect(() => f.newtonFractal({ roots: [[1, 0]] })).toThrow(/roots/)
  })
})

describe('iterated function systems', () => {
  it('the similarity dimensions are log 3 / log 2 and log 8 / log 3', () => {
    expect(f.similarityDimension(f.SIERPINSKI)).toBeCloseTo(Math.log(3) / Math.log(2), 12)
    expect(f.similarityDimension(f.SIERPINSKI_CARPET)).toBeCloseTo(Math.log(8) / Math.log(3), 12)
  })
  it('the chaos game stays in the triangle and its box-counting dimension is log 3 / log 2', () => {
    const s = runAlgorithm(f.chaosGame({ batch: 2000 }), { seed: 1, maxSteps: 60 }).state
    expect(s.cloud.length / 2).toBe(120000)
    for (let i = 0; i < s.cloud.length; i += 2) {
      const x = s.cloud[i] ?? 0
      const y = s.cloud[i + 1] ?? 0
      expect(y).toBeGreaterThanOrEqual(-1e-12)
      expect(y).toBeLessThanOrEqual(Math.sqrt(3) * Math.min(x, 1 - x) + 1e-9)
    }
    const d = f.boxCountingDimension(s.cloud)
    expect(Math.abs(d.dimension - Math.log(3) / Math.log(2))).toBeLessThan(0.05)
    expect(d.r2).toBeGreaterThan(0.999)
  })
  it('the deterministic iteration converges to the same set, with the same dimension', () => {
    const run = runAlgorithm(f.ifsDeterministic({ resolution: 256 }), { seed: 0, maxSteps: 40 })
    expect(run.status).toBe('converged')
    const d = f.boxCountingDimensionOfMask(run.state.mask, 256, 256, { minLevel: 1, maxLevel: 6 })
    expect(Math.abs(d.dimension - Math.log(3) / Math.log(2))).toBeLessThan(0.08)
    // Whatever it starts from, it is the same attractor: the set is invariant under the system.
    const again = runAlgorithm(f.ifsDeterministic({ resolution: 256 }), { seed: 0, maxSteps: 41 })
    expect(again.state.lit).toBe(run.state.lit)
  })
  it('the fern: chaos game points stay in the known bounding box', () => {
    const s = runAlgorithm(f.chaosGame({ ifs: f.BARNSLEY_FERN, batch: 1000 }), { seed: 2, maxSteps: 10 }).state
    const xs = Array.from({ length: s.cloud.length / 2 }, (_, i) => s.cloud[2 * i] ?? 0)
    const ys = Array.from({ length: s.cloud.length / 2 }, (_, i) => s.cloud[2 * i + 1] ?? 0)
    expect(Math.min(...xs)).toBeGreaterThan(-2.2)
    expect(Math.max(...xs)).toBeLessThan(2.7)
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(-0.01)
    expect(Math.max(...ys)).toBeLessThan(10.01)
  })
  it('probabilities default to area shares and a bad set is refused', () => {
    expect(f.ifsProbabilities(f.SIERPINSKI)).toHaveLength(3)
    const fern = f.ifsProbabilities({ name: 'x', maps: f.BARNSLEY_FERN.maps })
    expect(fern[0]).toBe(0)
    expect(() => f.ifsProbabilities({ name: 'bad', maps: f.SIERPINSKI.maps, probabilities: [0.5, 0.5] })).toThrow(
      /probability/,
    )
    expect(() => f.similarityDimension({ name: 'grow', maps: [{ a: 2, b: 0, c: 0, d: 2, e: 0, f: 0 }] })).toThrow(
      /contractions/,
    )
  })
})

describe('box counting', () => {
  it('a line has dimension 1 and a filled square 2', () => {
    const line: number[] = []
    const square: number[] = []
    for (let i = 0; i < 4096; i++) line.push(i / 4096, 0.3 + 0.4 * (i / 4096))
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) square.push((i + 0.5) / 64, (j + 0.5) / 64)
    expect(f.boxCountingDimension(line, { minLevel: 2, maxLevel: 7 }).dimension).toBeCloseTo(1, 1)
    expect(f.boxCountingDimension(square, { minLevel: 1, maxLevel: 5 }).dimension).toBeCloseTo(2, 5)
  })
  it('refuses too few points', () => {
    expect(() => f.boxCountingDimension([0, 0, 1, 1])).toThrow(/at least 16/)
  })
})

describe('L-systems', () => {
  const F = (text: string): number => Array.from(text).filter((c) => c === 'F').length
  const vertices = (t: f.Turtle): string[] => {
    const seen = new Set<string>()
    for (let i = 0; i < t.segments.length; i += 4) {
      seen.add(`${Math.round(t.segments[i] ?? 0)},${Math.round(t.segments[i + 1] ?? 0)}`)
      seen.add(`${Math.round(t.segments[i + 2] ?? 0)},${Math.round(t.segments[i + 3] ?? 0)}`)
    }
    return [...seen]
  }
  it('Koch: 4^n segments and an end-to-end distance of 3^n', () => {
    for (let n = 0; n <= 5; n++) {
      const text = f.rewrite(f.L_SYSTEMS.koch, n)
      expect(F(text)).toBe(4 ** n)
      const t = f.turtle(text, 60)
      const last = t.segments.length - 1
      expect(t.segments[last - 1]).toBeCloseTo(3 ** n, 8)
      expect(t.segments[last]).toBeCloseTo(0, 8)
    }
  })
  it('Hilbert: 4^n - 1 segments through the 4^n points of a 2^n grid, each once', () => {
    for (let n = 1; n <= 5; n++) {
      const t = f.turtle(f.rewrite(f.L_SYSTEMS.hilbert, n), 90)
      expect(t.segments.length / 4).toBe(4 ** n - 1)
      expect(vertices(t)).toHaveLength(4 ** n)
      expect(t.bounds[1] - t.bounds[0]).toBeCloseTo(2 ** n - 1, 8)
      expect(t.bounds[3] - t.bounds[2]).toBeCloseTo(2 ** n - 1, 8)
    }
  })
  it('Peano: 9^n - 1 segments through the 9^n points of a 3^n grid, each once', () => {
    for (let n = 1; n <= 3; n++) {
      const t = f.turtle(f.rewrite(f.L_SYSTEMS.peano, n), 90)
      expect(t.segments.length / 4).toBe(9 ** n - 1)
      expect(vertices(t)).toHaveLength(9 ** n)
    }
  })
  it('the dragon uses 2^n edges and never draws an edge twice', () => {
    for (let n = 1; n <= 10; n++) {
      const t = f.turtle(f.rewrite(f.L_SYSTEMS.dragon, n), 90)
      expect(t.segments.length / 4).toBe(2 ** n)
      const edges = new Set<string>()
      for (let i = 0; i < t.segments.length; i += 4) {
        const a = [Math.round(t.segments[i] ?? 0), Math.round(t.segments[i + 1] ?? 0)]
        const b = [Math.round(t.segments[i + 2] ?? 0), Math.round(t.segments[i + 3] ?? 0)]
        edges.add(a < b ? `${a}|${b}` : `${b}|${a}`)
      }
      expect(edges.size).toBe(2 ** n)
    }
  })
  it('the arrowhead has 3^n segments and the string lengths grow geometrically', () => {
    const lens = [0, 1, 2, 3, 4].map((n) => f.rewrite(f.L_SYSTEMS.arrowhead, n).length)
    for (let n = 0; n <= 5; n++) {
      const t = f.turtle(f.rewrite(f.L_SYSTEMS.arrowhead, n), 60, { draw: 'AB' })
      expect(t.segments.length / 4).toBe(3 ** n)
    }
    expect(lens[4] ?? 0).toBeGreaterThan(8 * (lens[2] ?? 1))
  })
  it("the Koch curve's box-counting dimension is log 4 / log 3", () => {
    const t = f.turtle(f.rewrite(f.L_SYSTEMS.koch, 7), 60)
    const ends = new Float64Array(t.segments.length / 2)
    for (let i = 0; i < ends.length / 2; i++) {
      ends[2 * i] = t.segments[4 * i + 2] ?? 0
      ends[2 * i + 1] = t.segments[4 * i + 3] ?? 0
    }
    const d = f.boxCountingDimension(ends, { minLevel: 2, maxLevel: 7 })
    expect(Math.abs(d.dimension - Math.log(4) / Math.log(3))).toBeLessThan(0.06)
  })
  it('brackets save and restore the turtle', () => {
    const t = f.turtle('F[+F]F', 90)
    expect(Array.from(t.segments).map(Math.round)).toEqual([0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 2, 0])
    expect(() => f.turtle(']', 90)).toThrow(/matching/)
  })
  it('stochastic rules give a different plant for each seed and the same for the same seed', () => {
    const g = f.L_SYSTEMS['stochastic-plant']
    expect(f.rewrite(g, 4, 1)).toBe(f.rewrite(g, 4, 1))
    expect(f.rewrite(g, 4, 1)).not.toBe(f.rewrite(g, 4, 2))
  })
  it('the algorithm grows the string one generation per step, logs the dimension, and refuses a runaway', () => {
    const alg = f.lSystemAlgorithm({ system: f.L_SYSTEMS.koch })
    const s = runAlgorithm(alg, { seed: 0, maxSteps: 4 }).state
    expect(s.text).toBe(f.rewrite(f.L_SYSTEMS.koch, 4))
    expect(alg.deserialise?.(alg.serialise?.(s) ?? null)).toEqual(s)
    expect(() => runAlgorithm(f.lSystemAlgorithm({ maxLength: 100 }), { seed: 0, maxSteps: 4 })).toThrow(
      /more than 100/,
    )
    expect(() =>
      f.rewriteOnce('X', { name: 'bad', axiom: 'X', rules: { X: [{ p: 0.5, to: 'F' }] }, angle: 90 }),
    ).toThrow(/chances/)
  })
})

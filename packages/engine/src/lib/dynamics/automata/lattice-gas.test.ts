import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../../trace/index.ts'
import * as a from './index.ts'

const at = (
  model: a.LatticeGasModel,
  width: number,
  height: number,
  cells: Uint8Array,
  steps: number,
  seed = 0,
): Uint8Array => {
  const alg = a.latticeGas({ model, width, height, block: 1 })
  return runAlgorithm({ ...alg, init: () => ({ cells: Uint8Array.from(cells), time: 0 }) }, { seed, maxSteps: steps })
    .state.cells
}

describe('lattice gas rules', () => {
  it('HPP: a head-on pair turns through a right angle and then flies apart', () => {
    const cells = new Uint8Array(25)
    cells[12] = 0b0101 // east + west at the middle of a 5 x 5 torus
    const after = at('hpp', 5, 5, cells, 1)
    expect(after[7]).toBe(0b0010) // north-mover at row 1, column 2
    expect(after[17]).toBe(0b1000) // south-mover at row 3
    expect(after.reduce((s, v) => s + v, 0)).toBe(0b1010)
  })
  it('HPP: a lone particle goes straight round the torus in `width` steps', () => {
    const cells = new Uint8Array(6 * 4)
    cells[1 * 6 + 2] = 0b0001
    expect(Array.from(at('hpp', 6, 4, cells, 6))).toEqual(Array.from(cells))
    expect(at('hpp', 6, 4, cells, 1)[1 * 6 + 3]).toBe(0b0001)
  })
  it('FHP: a head-on pair turns by 60 degrees one way or the other, and the triples swap', () => {
    const cells = new Uint8Array(6 * 6)
    cells[2 * 6 + 2] = 0b001001 // east + west
    const outcomes = new Set<string>()
    for (let seed = 0; seed < 12; seed++) {
      const once = at('fhp', 6, 6, cells, 1, seed)
      outcomes.add(Array.from(once, (v, k) => (v ? `${k}:${v}` : '')).join(' '))
      expect(a.gasMassMomentum(once, 'fhp').mass).toBe(2)
    }
    expect(outcomes.size).toBeGreaterThan(1)
    const triple = new Uint8Array(36)
    triple[14] = 0b010101
    const flipped = at('fhp', 6, 6, triple, 1)
    expect(a.gasMassMomentum(flipped, 'fhp').mass).toBe(3)
  })
  it('FHP streaming is a lattice permutation: each direction is undone by the opposite one', () => {
    for (let r = 0; r < 6; r++)
      for (let c = 0; c < 6; c++)
        for (let d = 0; d < 6; d++) {
          const [rr, cc] = a.gasNeighbour('fhp', r, c, d, 6, 6)
          expect(a.gasNeighbour('fhp', rr, cc, (d + 3) % 6, 6, 6)).toEqual([r, c])
        }
  })
  it('conserves mass and momentum exactly, for any start, on both lattices', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('hpp', 'fhp') as fc.Arbitrary<a.LatticeGasModel>,
        fc.integer({ min: 0, max: 10_000 }),
        fc.double({ min: 0.05, max: 0.6, noNaN: true }),
        (model, seed, density) => {
          const alg = a.latticeGas({ model, width: 24, height: 24, density, block: 4, bias: [0.1, 0] })
          const start = runAlgorithm(alg, { seed, maxSteps: 0 }).state
          const end = runAlgorithm(alg, { seed, maxSteps: 40 }).state
          const g0 = a.gasMassMomentum(start.cells, model)
          const g1 = a.gasMassMomentum(end.cells, model)
          expect(g1.mass).toBe(g0.mass)
          expect(g1.momentum[0]).toBeCloseTo(g0.momentum[0], 9)
          expect(g1.momentum[1]).toBeCloseTo(g0.momentum[1], 9)
        },
      ),
      { numRuns: 20 },
    )
  })
  it('solids reflect: mass is conserved, and a particle aimed at a wall comes back', () => {
    const solid = new Uint8Array(6 * 3)
    solid[1 * 6 + 3] = 1
    const cells = new Uint8Array(18)
    cells[1 * 6 + 1] = 0b0001
    // East from column 1: step 1 at column 2, step 2 arrives at the wall (column 3), step 3 reflects (collision) and
    // leaves column 3 westward to column 2, step 4 at column 1.
    const alg = a.latticeGas({ width: 6, height: 3, block: 3, solid })
    const run = (t: number): Uint8Array =>
      runAlgorithm({ ...alg, init: () => ({ cells: Uint8Array.from(cells), time: 0 }) }, { seed: 0, maxSteps: t }).state
        .cells
    expect(run(2)[1 * 6 + 3]).toBe(0b0001)
    expect(run(3)[1 * 6 + 2]).toBe(0b0100)
    expect(run(4)[1 * 6 + 1]).toBe(0b0100)
  })
  it('coarse-graining a uniform random gas gives its density and almost no flow', () => {
    const alg = a.latticeGas({ model: 'fhp', width: 128, height: 128, density: 0.25, block: 32 })
    const s = runAlgorithm(alg, { seed: 3, maxSteps: 10 }).state
    const f = a.coarseGrainGas(s.cells, 128, 128, 32, 'fhp')
    for (const d of f.density) expect(Math.abs(d - 6 * 0.25)).toBeLessThan(0.1)
    for (const v of f.vx) expect(Math.abs(v)).toBeLessThan(0.1)
  })
  it('a pushed gas keeps flowing: the mean velocity follows the bias', () => {
    const alg = a.latticeGas({ width: 64, height: 64, density: 0.2, bias: [0.4, 0], block: 64 })
    const s = runAlgorithm(alg, { seed: 5, maxSteps: 20 }).state
    expect(a.coarseGrainGas(s.cells, 64, 64, 64, 'hpp').vx[0]).toBeGreaterThan(0.1)
  })
  it('validates its arguments and the coarse-graining block', () => {
    expect(() => a.latticeGas({ model: 'fhp', height: 5, block: 1 })).toThrow(/even/)
    expect(() => a.latticeGas({ width: 10, height: 10, block: 4 })).toThrow(/block/)
    expect(() => a.coarseGrainGas(new Uint8Array(9), 3, 3, 2, 'hpp')).toThrow(/divide/)
  })
  it('the vorticity of solid-body rotation is 2', () => {
    const ux = Float64Array.from({ length: 25 }, (_, i) => Math.floor(i / 5) - 2)
    const uy = Float64Array.from({ length: 25 }, (_, i) => (i % 5) - 2)
    expect(a.curl2d(ux, uy, 5, 5)[12]).toBe(2)
  })
})

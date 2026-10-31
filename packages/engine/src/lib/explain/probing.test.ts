import { describe, expect, it } from 'vitest'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { controlTask, probeAccuracy, probeLayers, probeSelectivity } from './index.ts'

// 12 word types, each with a fixed 3-d embedding; the property is whether the type id is even.
const types = Array.from({ length: 240 }, (_, i) => (i * 7) % 12)
const parity = types.map((t) => t % 2)
const readable = types.map((t) => [parity[types.indexOf(t)] ?? 0, 0.3 * Math.sin(t)])

describe('probes', () => {
  it('reads a linearly encoded property and not an absent one', () => {
    const yes = probeAccuracy(readable, parity, { key: key(0) })
    expect(yes.testAccuracy).toBe(1)
    expect(yes.baseline).toBeLessThan(0.7)
    const noise = types.map((_, i) => [Math.sin(i * 12.9898), Math.cos(i * 78.233)])
    const no = probeAccuracy(noise, parity, { key: key(0) })
    expect(no.testAccuracy).toBeLessThan(0.7)
  })
  it('agrees across solvers of the slot', () => {
    const newton = probeAccuracy(readable, parity, { key: key(0) })
    const lbfgs = probeAccuracy(readable, parity, {
      key: key(0),
      solver: smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-10 }),
    })
    expect(lbfgs.testAccuracy).toBe(newton.testAccuracy)
  })
  it('control labels follow the type and the label frequencies', () => {
    const c = controlTask(types, parity, key(3))
    for (let i = 0; i < types.length; i++) expect(c[i]).toBe(c[types.indexOf(types[i] as number)])
    expect(new Set(c).size).toBeLessThanOrEqual(2)
  })
  it('selectivity is high when the layer carries the property, low when the probe can memorise', () => {
    const good = probeSelectivity(readable, parity, types, { key: key(1) })
    expect(good.task.testAccuracy).toBe(1)
    // The control labels are arbitrary per type, so a 2-d readable axis cannot fit them.
    expect(good.selectivity).toBeGreaterThan(0.2)
    // A one-hot layer lets the probe memorise any labelling of the types: the control is fit as well as the task.
    const oneHot = types.map((ty) => Array.from({ length: 12 }, (_, j) => (j === ty ? 1 : 0)))
    const memo = probeSelectivity(oneHot, parity, types, { key: key(1), alpha: 1e-4 })
    expect(memo.control.testAccuracy).toBeGreaterThan(0.95)
    expect(memo.selectivity).toBeLessThan(good.selectivity - 0.2)
  })
  it('maps accuracy down a stack of layers', () => {
    const early = types.map((_, i) => [Math.sin(i * 12.9898), Math.cos(i * 78.233)])
    const r = probeLayers([early, readable], parity, { key: key(2), types })
    expect(r.accuracy[1]).toBe(1)
    expect(r.accuracy[0] as number).toBeLessThan(0.75)
    expect(r.selectivity?.length).toBe(2)
  })
  it('rejects labels that do not match the rows', () => {
    expect(() => probeAccuracy([[1], [2]], [0], { key: key(0) })).toThrow()
  })
})

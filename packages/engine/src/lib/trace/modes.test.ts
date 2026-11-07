import { describe, expect, it } from 'vitest'
import { gradientDescentAlgorithm, newtonAlgorithm } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { defaults, dict, real } from '../space/index.ts'
import { square, sub, sum, type Value } from '../tensor/index.ts'
import { runAlgorithm, scalar } from './index.ts'
import { type ModeSpec, modes } from './modes.ts'

// A problem: the minimiser of a bowl centred at `centre`; two modes reach it their own way.
interface Bowl {
  readonly centre: readonly number[]
}
// A type alias, not an interface: the fitted parameters must be plain JSON, and an alias has the index signature.
type Params = {
  readonly x: number[]
}
interface WithX {
  readonly x: Float64Array
}

const objective =
  (b: Bowl) =>
  (x: Value): Value =>
    sum(square(sub(x, b.centre)))

const gradient: ModeSpec<Bowl, WithX, Params> = {
  description: 'gradient descent with a fixed step',
  space: dict({ learningRate: real(0.01, 0.5, { default: 0.2 }) }),
  make: (b, knobs) =>
    gradientDescentAlgorithm(
      objective(b),
      b.centre.map(() => 0),
      {
        step: 'fixed',
        learningRate: knobs['learningRate'] as number,
        gradTol: 1e-8,
      },
    ),
  params: (s) => ({ x: Array.from(s.x) }),
  needs: { landscape: ['smooth'] },
}

const newton: ModeSpec<Bowl, WithX, Params> = {
  description: "Newton's method",
  space: dict({}),
  make: (b) =>
    newtonAlgorithm(
      objective(b),
      b.centre.map(() => 0),
    ),
  params: (s) => ({ x: Array.from(s.x) }),
  extras: { 'newton/steps': scalar({ description: 'the steps taken' }) },
}

const solver = modes<Bowl, Params>({ gradient, newton }, { default: 'newton', label: 'solver' })

describe('modes', () => {
  it('is one union space over the modes’ knobs, with the tag naming the mode', () => {
    expect(solver.tag).toBe('kind')
    expect(solver.names).toEqual(['gradient', 'newton'])
    expect(defaults(solver.space)).toEqual({ kind: 'newton' })
    expect(solver.modeOf({ kind: 'gradient', learningRate: 0.1 }).name).toBe('gradient')
    expect(solver.modeOf(undefined).name).toBe('newton')
    expect(solver.modeOf({ kind: 'nope' }).name).toBe('newton')
    // inherited object names are not modes
    expect(solver.modeOf({ kind: 'constructor' }).name).toBe('newton')
    expect(solver.modeOf({ kind: 'toString' }).name).toBe('newton')
  })

  it('dispatches make and params to the mode a configuration names, and keeps the mode’s extras', () => {
    const bowl: Bowl = { centre: [1, -2] }
    for (const config of [{ kind: 'gradient', learningRate: 0.3 }, { kind: 'newton' }]) {
      const algorithm = solver.make(bowl, config, key(0))
      const { state } = runAlgorithm(algorithm, { seed: 0, maxSteps: 200 })
      const p = solver.params(config, state)
      expect(p.x[0]).toBeCloseTo(1, 5)
      expect(p.x[1]).toBeCloseTo(-2, 5)
    }
    expect(Object.keys(solver.extras({ kind: 'newton' }))).toEqual(['newton/steps'])
    expect(solver.extras({ kind: 'gradient' })).toEqual({})
  })

  it('reads the validity table: a mode admits the other slots’ values or not, primitive or union', () => {
    expect(solver.admissible({ landscape: 'smooth' })).toEqual(['gradient', 'newton'])
    expect(solver.admissible({ landscape: 'rough' })).toEqual(['newton'])
    expect(solver.admissible({ landscape: { kind: 'smooth', bumps: 0 } })).toEqual(['gradient', 'newton'])
    expect(solver.admissible({})).toEqual(['newton'])
    expect(() =>
      solver.make({ centre: [0] }, { kind: 'gradient', learningRate: 0.1 }, key(0), { landscape: 'rough' }),
    ).toThrow(/does not admit/)
    expect(() => modes({})).toThrow(/at least one/)
  })
})

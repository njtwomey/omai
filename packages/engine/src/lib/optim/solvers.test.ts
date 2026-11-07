import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { type DictSpace, defaults, grid } from '../space/index.ts'
import { mul, square, sub, sum, type Value, zeros } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type SlotState, slotAlgorithm } from './slot.ts'
import { loggedSolver, type SmoothProblem, smoothSolvers } from './solvers.ts'

// An ill-conditioned bowl centred at (1, -2): f(x) = (x0 - 1)^2 + 10 (x1 + 2)^2.
const bowl: SmoothProblem = {
  objective: (x: Value): Value => sum(mul(square(sub(x, [1, -2])), [1, 10])),
  x0: [3, 3],
}

describe('smoothSolvers', () => {
  it('is one slot over seven solvers with the knobs of the chosen one', () => {
    expect(smoothSolvers.names).toEqual([
      'gradientDescent',
      'conjugateGradient',
      'newton',
      'bfgs',
      'lbfgs',
      'coordinateDescent',
      'gradient',
    ])
    // The learning rate is a knob of the fixed and Barzilai-Borwein steps only.
    expect(defaults(smoothSolvers.space)).toEqual({ kind: 'gradientDescent', step: 'line-search', gradTol: 1e-5 })
    expect(smoothSolvers.modeOf({ kind: 'lbfgs', memory: 5 }).name).toBe('lbfgs')
    expect(grid(smoothSolvers.space, 2).length).toBeGreaterThan(6)
  })

  it('reaches the minimum with every solver, each in its own way, with the same fitted parameters', () => {
    const configs = [
      ...smoothSolvers.names.map((name) => ({
        ...(defaults(smoothSolvers.space.options[name] as DictSpace) as object),
        kind: name,
      })),
      { kind: 'gradientDescent', step: 'bb1', learningRate: 0.01, gradTol: 1e-6 },
    ]
    for (const config of configs) {
      const algorithm = smoothSolvers.make(bowl, config, key(0))
      const { state } = runAlgorithm(algorithm, { seed: 0, maxSteps: 500 })
      const { x } = smoothSolvers.params(config, state)
      expect(x[0], JSON.stringify(config)).toBeCloseTo(1, 3)
      expect(x[1], JSON.stringify(config)).toBeCloseTo(-2, 3)
    }
  })

  // The gradient norm was `Math.hypot(...grad)`, whose spread overflows the call stack at about 130,000 parameters.
  it('logs the gradient norm of a large parameter vector through loggedSolver and slotAlgorithm', () => {
    const n = 200_000
    const problem: SmoothProblem = { objective: (x: Value): Value => sum(square(sub(x, 1))), x0: zeros([n]) }
    const config = { kind: 'gradientDescent', step: 'fixed', learningRate: 0.25 }
    // The gradient 2 (x - 1) is -1 everywhere after one step of 0.25 from zero: its norm is sqrt(n).
    const expected = Math.sqrt(n)
    const logged = loggedSolver(problem, config, key(0))
    const seen: Record<string, number> = {}
    const ctx = {
      log: (path: string, v: number) => {
        seen[path] = v
      },
      key: key(1),
    }
    logged.step(logged.init(key(0)), ctx as never)
    expect(seen['solver/gradNorm']).toBeCloseTo(expected, 6)
    const slotted = slotAlgorithm<SlotState>({
      name: 'big',
      config: {},
      schema: { metrics: {}, tensors: {} },
      problem,
      solver: config,
      tol: 1e-12,
      view: () => ({}),
      observe: () => ({ metrics: {} }),
    })
    const state = slotted.step(slotted.init(key(0)), { key: key(1) } as never)
    expect(slotted.observe?.(state, 1).metrics?.['gradNorm']).toBeCloseTo(expected, 6)
  })
})

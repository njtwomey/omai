import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { type DictSpace, defaults, grid } from '../../../lib/space/index.ts'
import { mul, square, sub, sum, type Value } from '../../../lib/tensor/index.ts'
import { runAlgorithm } from '../../../lib/trace/index.ts'
import { type SmoothProblem, smoothSolvers } from './solvers.ts'

// An ill-conditioned bowl centred at (1, -2): f(x) = (x0 - 1)^2 + 10 (x1 + 2)^2.
const bowl: SmoothProblem = {
  objective: (x: Value): Value => sum(mul(square(sub(x, [1, -2])), [1, 10])),
  x0: [3, 3],
}

describe('smoothSolvers', () => {
  it('is one slot over six solvers with the knobs of the chosen one', () => {
    expect(smoothSolvers.names).toEqual([
      'gradientDescent',
      'conjugateGradient',
      'newton',
      'bfgs',
      'lbfgs',
      'coordinateDescent',
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
})

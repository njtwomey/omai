import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { add, concat, mul, sliceAxis, stack, sub, sum, type Value } from '../../tensor/index.ts'
import { solveDae } from './dae.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

// Robertson with y3 = 1 - y1 - y2 enforced as the algebraic equation.
const robertson = {
  f: (_t: number, y: Value, z: Value): Value => {
    const [a, b, c] = [at(y, 0), at(y, 1), at(z, 0)]
    const r1 = mul(a, 0.04)
    const r2 = mul(mul(b, c), 1e4)
    const r3 = mul(mul(b, b), 3e7)
    return stack([sub(r2, r1), sub(sub(r1, r2), r3)])
  },
  g: (_t: number, y: Value, z: Value): Value => stack([sub(add(add(at(y, 0), at(y, 1)), at(z, 0)), 1)]),
}

const FNS: Readonly<Record<string, GoldenFn>> = {
  robertson_dae: (_x, p) => {
    const s = solveDae(robertson, [0, 40], [1, 0], [0], {
      method: p['method'] as 'Radau' | 'BDF',
      rtol: 1e-8,
      atol: 1e-12,
      tEval: [0.4, 4, 40],
    })
    return concat([s.yDiff, s.zAlg], 1)
  },
}

describe('dynamics/ode/dae against scipy (ODE form)', () => {
  runFixture('dynamics/ode/dae', FNS, it)
})

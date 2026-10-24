// Compiled functions against sympy.lambdify with numpy: elementwise values on arrays (NaN where numpy gives NaN).
import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { erf } from '../numerics/special/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { compile, parse } from './index.ts'

const fixture = loadFixture('symbolic/compile')

describe('compile (sympy lambdify)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${String(c.inputs[0])}`, () => {
      const f = compile(parse(String(c.inputs[0])), ['x', 'y'], { functions: { erf } })
      const out = f(decode(c.inputs[1]) as Tensor, decode(c.inputs[2]) as Tensor)
      expectMatches(out, c.output, c.rtol, c.atol, c.name)
    })
  }
})

import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { PROBLEMS } from '../../../testing/optim-problems.ts'
import {
  add,
  mul,
  sliceAxis,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { cobyla, type InequalityConstraint } from './cobyla.ts'
import { type Bounds, nelderMead, powell } from './derivative-free.ts'
import type { MinimizeResult } from './objective.ts'

const num = (v: unknown): number => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : (v as number))

function boundsOf(p: Readonly<Record<string, unknown>>): Bounds | undefined {
  const b = p['bounds'] as unknown[][] | undefined
  return b?.map((pair) => [num(pair[0]), num(pair[1])] as const)
}

function solution(r: MinimizeResult): Value {
  return stack([...toFlat(r.x as Value), r.fun].map((v) => tensor(v)))
}

const xy = (x: Value): [Value, Value] => [sliceAxis(x, 0, 0, 1), sliceAxis(x, 0, 1, 2)]
const CONSTRAINTS: Readonly<Record<string, InequalityConstraint>> = {
  line: (x) => {
    const [a, b] = xy(x as Value)
    return sum(sub(sub(1, a), b))
  },
  disc: (x) => {
    const [a, b] = xy(x as Value)
    return sum(sub(1.5, add(square(a), square(b))))
  },
  tutorial1: (x) => {
    const [a, b] = xy(x as Value)
    return sum(add(sub(a, mul(b, 2)), 2))
  },
  tutorial2: (x) => {
    const [a, b] = xy(x as Value)
    return sum(sub(sub(6, a), mul(b, 2)))
  },
  tutorial3: (x) => {
    const [a, b] = xy(x as Value)
    return sum(add(sub(mul(b, 2), a), 2))
  },
}

describe('derivative-free methods against scipy.optimize.minimize', () => {
  for (const c of loadFixture('optim/minimize/derivative_free').cases) {
    it(c.name, () => {
      const x0 = decode(c.inputs[0]) as Tensor
      const f = PROBLEMS[c.params['problem'] as string] as (x: Value) => Value
      const p = c.params
      const bounds = boundsOf(p)
      const common = {
        ...(bounds === undefined ? {} : { bounds }),
        ...(p['maxiter'] === undefined ? {} : { maxIterations: p['maxiter'] as number }),
        ...(p['maxfev'] === undefined ? {} : { maxEvaluations: p['maxfev'] as number }),
      }
      if (c.fn === 'cobyla') {
        const cons = (p['constraints'] as string[]).map((k) => CONSTRAINTS[k] as InequalityConstraint)
        const r = cobyla((x) => f(x as Value), x0, {
          constraints: cons,
          rhobeg: 0.5,
          rhoend: p['rhoend'] as number,
          maxEvaluations: p['maxfev'] as number,
          ...(bounds === undefined ? {} : { bounds }),
        })
        expect(r.converged).toBe(true)
        expectMatches(solution(r), c.output, c.rtol, c.atol, c.name)
        return
      }
      const r =
        c.fn === 'nelderMead'
          ? nelderMead((x) => f(x as Value), x0, {
              ...common,
              ...(p['xatol'] === undefined ? {} : { xatol: p['xatol'] as number }),
              ...(p['fatol'] === undefined ? {} : { fatol: p['fatol'] as number }),
              ...(p['adaptive'] === undefined ? {} : { adaptive: p['adaptive'] as boolean }),
            })
          : powell((x) => f(x as Value), x0, {
              ...common,
              ...(p['xtol'] === undefined ? {} : { xtol: p['xtol'] as number }),
              ...(p['ftol'] === undefined ? {} : { ftol: p['ftol'] as number }),
            })
      expectMatches(solution(r), c.output, c.rtol, c.atol, c.name)
      // Nelder-Mead makes exactly scipy's moves; Powell's Brent searches can part by an evaluation or two when the
      // library functions (tan, arctan) differ in the last bit.
      if (c.fn === 'nelderMead') expect(r.nfev).toBe(p['nfev'])
      else expect(Math.abs(r.nfev - (p['nfev'] as number))).toBeLessThanOrEqual(0.05 * (p['nfev'] as number))
    })
  }
})

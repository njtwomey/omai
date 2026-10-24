import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { cos, mul, sliceAxis, stack, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { event } from './problem.ts'
import { type OdeMethodName, odeint, solveIvp } from './solve-ivp.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

// The problems of oracle/cases/dynamics/ode/solve_ivp.py.
const PROBLEMS: Readonly<
  Record<string, { f: (t: number, y: Value) => Value; y0: number[]; span: [number, number]; tEval: number[] }>
> = {
  decay: {
    f: (_t, y) => stack([mul(at(y, 0), -0.5), mul(at(y, 1), -2)]),
    y0: [1, 2],
    span: [0, 3],
    tEval: [0.5, 1, 2, 3],
  },
  lotka: {
    f: (_t, y) => {
      const [a, b] = [at(y, 0), at(y, 1)]
      return stack([sub(mul(a, 1.5), mul(a, b)), sub(mul(a, b), mul(b, 3))])
    },
    y0: [10, 5],
    span: [0, 4],
    tEval: [0.7, 1.9, 3.1, 4],
  },
  forced: {
    f: (t, y) => {
      const [x, v] = [at(y, 0), at(y, 1)]
      return stack([v, sub(sub(cos(tensor(1.3 * t)), x), mul(v, 0.1))])
    },
    y0: [1, 0],
    span: [0, 6],
    tEval: [1, 2.5, 4, 6],
  },
  vdp: {
    f: (_t, y) => {
      const [x, v] = [at(y, 0), at(y, 1)]
      return stack([v, sub(mul(mul(sub(1, mul(x, x)), v), 10), x)])
    },
    y0: [2, 0],
    span: [0, 3],
    tEval: [0.5, 1.5, 3],
  },
  robertson: {
    f: (_t, y) => {
      const [a, b, c] = [at(y, 0), at(y, 1), at(y, 2)]
      const r1 = mul(a, 0.04)
      const r2 = mul(mul(b, c), 1e4)
      const r3 = mul(mul(b, b), 3e7)
      return stack([sub(r2, r1), sub(sub(r1, r2), r3), r3])
    },
    y0: [1, 0, 0],
    span: [0, 40],
    tEval: [0.4, 4, 40],
  },
}

const problem = (p: Readonly<Record<string, unknown>>): (typeof PROBLEMS)[string] => {
  const out = PROBLEMS[p['problem'] as string]
  if (out === undefined) throw new Error(`no problem ${String(p['problem'])}`)
  return out
}

const FNS: Readonly<Record<string, GoldenFn>> = {
  solve_ivp: (_x, p) => {
    const pr = problem(p)
    const method = p['method'] as OdeMethodName
    const rtol = p['tol'] as number
    const atol = (p['atol_solver'] as number | undefined) ?? rtol * 1e-2
    const what = p['what'] as string
    if (what === 'dense') {
      const sol = solveIvp(pr.f, pr.span, pr.y0, { method, rtol, atol, denseOutput: true })
      const probe = Array.from({ length: 7 }, (_, i) => pr.span[0] + ((pr.span[1] - pr.span[0]) * i) / 6)
      return (sol.sol as NonNullable<typeof sol.sol>).many(probe)
    }
    if (what === 't') return tensor(solveIvp(pr.f, pr.span, pr.y0, { method, rtol, atol }).t as number[])
    const sol = solveIvp(pr.f, pr.span, pr.y0, { method, rtol, atol, tEval: pr.tEval })
    return what === 'nfev' ? sol.nfev : sol.y
  },
  events: (_x, p) => {
    const pr = PROBLEMS['forced'] as (typeof PROBLEMS)[string]
    const sol = solveIvp(pr.f, [0, 20], [1, 0], {
      method: p['method'] as OdeMethodName,
      rtol: 1e-9,
      atol: 1e-11,
      events: [
        event((_t, y) => toFlat(y)[0] ?? 0, { terminal: true, direction: -1 }),
        event((_t, y) => toFlat(y)[1] ?? 0),
      ],
    })
    const which = p['which'] as number
    return which < 0 ? (sol.t[sol.t.length - 1] ?? 0) : tensor(sol.tEvents[which] as number[])
  },
  odeint: (_x, p) => {
    const pr = problem(p)
    return odeint(pr.f, pr.y0, [pr.span[0], ...pr.tEval])
  },
}

describe('dynamics/ode/solve_ivp against scipy.integrate', () => {
  runFixture('dynamics/ode/solve_ivp', FNS, it)
})

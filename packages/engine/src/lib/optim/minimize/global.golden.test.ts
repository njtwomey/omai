import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { PROBLEMS } from '../../../testing/optim-problems.ts'
import { add, sliceAxis, square, stack, sub, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import type { Bounds } from './derivative-free.ts'
import { cmaes, type DifferentialEvolutionStrategy, differentialEvolution, particleSwarm } from './evolutionary.ts'
import { basinHopping, dualAnnealing, simulatedAnnealing } from './global.ts'
import type { MinimizeResult } from './objective.ts'

const num = (v: unknown): number => (v === 'inf' ? Infinity : v === '-inf' ? -Infinity : (v as number))
const boundsOf = (p: Readonly<Record<string, unknown>>): Bounds | undefined =>
  (p['bounds'] as unknown[][] | undefined)?.map((b) => [num(b[0]), num(b[1])] as const)

/** The solution and value, or the value alone when the minimiser is not unique. */
function solution(r: MinimizeResult, unique: boolean): Value {
  return stack([...(unique ? toFlat(r.x as Value) : []), r.fun].map((v) => tensor(v)))
}

const disc = (x: unknown): Value => {
  const v = x as Value
  return sum(sub(1.5, add(square(sliceAxis(v, 0, 0, 1)), square(sliceAxis(v, 0, 1, 2)))))
}

function run(
  fn: string,
  f: (x: unknown) => Value,
  x0: Tensor | undefined,
  p: Readonly<Record<string, unknown>>,
): MinimizeResult {
  const bounds = boundsOf(p) ?? []
  const start = x0 ?? tensor(bounds.map(([lo, hi]) => (lo + hi) / 2))
  switch (fn) {
    case 'differentialEvolution':
      return differentialEvolution(f, start, {
        bounds,
        seed: 1,
        tol: 1e-6,
        strategy: p['strategy'] as DifferentialEvolutionStrategy,
        updating: p['updating'] as 'immediate' | 'deferred',
        ...(p['constrained'] === true ? { constraints: [disc] } : {}),
      })
    case 'cmaes':
      return cmaes(f, start, { sigma0: p['sigma0'] as number, seed: 1, tolFun: 1e-14, tolX: 1e-12 })
    case 'particleSwarm':
      return particleSwarm(f, start, { bounds, seed: 1 })
    case 'dualAnnealing':
      return dualAnnealing(f, start, { bounds, seed: 1, maxIterations: 300 })
    case 'basinHopping':
      return basinHopping(f, start, { seed: 1, maxIterations: p['niter'] as number })
    default:
      return simulatedAnnealing(f, start, {
        ...(boundsOf(p) === undefined ? {} : { bounds }),
        seed: 1,
        maxIterations: 400,
      })
  }
}

for (const fixture of ['optim/minimize/evolutionary', 'optim/minimize/global']) {
  describe(`${fixture} against scipy and pycma (minima found)`, () => {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const f = PROBLEMS[c.params['problem'] as string] as (x: Value) => Value
        const x0 = c.inputs[0] === undefined ? undefined : (decode(c.inputs[0]) as Tensor)
        const r = run(c.fn, (x) => f(x as Value), x0, c.params)
        expect(Number.isFinite(r.fun)).toBe(true)
        expectMatches(solution(r, c.params['unique'] === true), c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}

import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { problem } from '../../../testing/optim-problems.ts'
import {
  add,
  concat,
  mul,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { metricSeries, recorder, runTree } from '../../trace/index.ts'
import * as O from '../index.ts'

const at = (x: Value, i: number): Value => sliceAxis(x, 0, i, i + 1)

interface Constrained {
  readonly f: (x: Value) => Value
  readonly constraints: readonly O.Constraint[]
}

const PROBLEMS: Readonly<Record<string, Constrained>> = {
  tutorial: {
    f: (x) => sum(square(sub(x, tensor([1, 2.5])))),
    constraints: [
      {
        type: 'ineq',
        fun: (x: Value) =>
          concat([
            add(sub(at(x, 0), mul(at(x, 1), 2)), 2),
            add(sub(neg1(at(x, 0)), mul(at(x, 1), 2)), 6),
            add(add(neg1(at(x, 0)), mul(at(x, 1), 2)), 2),
          ]),
      },
    ],
  },
  hs071: {
    f: (x) => sum(add(mul(mul(at(x, 0), at(x, 3)), add(add(at(x, 0), at(x, 1)), at(x, 2))), at(x, 2))),
    constraints: [
      { type: 'eq', fun: (x: Value) => sub(sum(square(x)), 40) },
      { type: 'ineq', fun: (x: Value) => sub(mul(mul(at(x, 0), at(x, 1)), mul(at(x, 2), at(x, 3))), 25) },
    ],
  },
  rosen: {
    f: problem('rosen'),
    constraints: [
      { type: 'eq', fun: (x: Value) => sub(sub(1, mul(at(x, 0), 2)), at(x, 1)) },
      {
        type: 'ineq',
        fun: (x: Value) =>
          concat([
            sub(sub(1, at(x, 0)), mul(at(x, 1), 2)),
            sub(sub(1, square(at(x, 0))), at(x, 1)),
            add(sub(1, square(at(x, 0))), at(x, 1)),
          ]),
      },
    ],
  },
  sphere: {
    f: (x) => sum(square(x)),
    constraints: [
      { type: 'eq', fun: (x: Value) => sub(sum(x), 1) },
      { type: 'ineq', fun: (x: Value) => sub(at(x, 0), 0.5) },
    ],
  },
  disc: {
    f: (x) => add(square(sub(at(x, 0), 2)), square(sub(at(x, 1), 1))),
    constraints: [{ type: 'ineq', fun: (x: Value) => sub(sub(1, square(at(x, 0))), square(at(x, 1))) }],
  },
}

function neg1(x: Value): Value {
  return mul(x, -1)
}

const boundsOf = (lower: Tensor, upper: Tensor): O.Bounds | undefined =>
  toFlat(lower).length === 0 ? undefined : { lower: toFlat(lower), upper: toFlat(upper) }

describe('constrained minimisation against scipy.optimize', () => {
  for (const fixture of ['optim/minimize/slsqp', 'optim/minimize/constrained']) {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const [x0, lower, upper] = c.inputs.map((v) => decode(v) as Tensor) as [
          Tensor,
          Tensor | undefined,
          Tensor | undefined,
        ]
        const name = c.params['problem'] as string
        const bounds = lower === undefined || upper === undefined ? undefined : boundsOf(lower, upper)
        const withBounds = bounds === undefined ? {} : { bounds }
        let r: O.MinimizeResult
        if (c.fn === 'projectedGradient') {
          r = O.projectedGradient(problem(name), x0, { ...withBounds, gradTol: 1e-10, maxIterations: 50000 })
        } else {
          const p = PROBLEMS[name] as Constrained
          r =
            c.fn === 'slsqp'
              ? O.slsqp(p.f, x0, { constraints: p.constraints, ...withBounds, ftol: 1e-12 })
              : c.fn === 'augmentedLagrangian'
                ? O.augmentedLagrangian(p.f, x0, { constraints: p.constraints, ...withBounds, tol: 1e-9 })
                : c.fn === 'penaltyMethod'
                  ? O.penaltyMethod(p.f, x0, { constraints: p.constraints, ...withBounds, tol: 1e-6 })
                  : O.barrierMethod(p.f, x0, { constraints: p.constraints, tol: 1e-10 })
        }
        expect(r.converged).toBe(true)
        expectMatches(concat([r.x as Value, tensor([r.fun])]), c.output, c.rtol, c.atol, c.name)
      })
    }
  }
})

describe('constrained methods', () => {
  it('SLSQP reports multipliers satisfying stationarity', () => {
    const p = PROBLEMS['sphere'] as Constrained
    const r = O.slsqp(p.f, [1, 1, 1], { constraints: p.constraints, ftol: 1e-12 })
    // grad f = 2x = lambda * 1 + mu * e0 at (0.5, 0.25, 0.25): lambda = 0.5, mu = 0.5.
    expect(Array.from(r.multipliers).map((v) => Math.round(v * 1e6) / 1e6)).toEqual([0.5, 0.5])
  })
  it('outer loops run their inner solves as child runs and trace the violation', () => {
    const rec = recorder()
    const p = PROBLEMS['sphere'] as Constrained
    O.augmentedLagrangian(p.f, [1, 1, 1], { constraints: p.constraints, tracer: rec })
    const [root] = runTree(rec.run())
    expect(root?.children.length).toBeGreaterThan(0)
    expect(root?.children[0]?.run.name).toBe('lbfgsb')
    const v = metricSeries(rec.run(), 'violation').values
    expect((v.at(-1) ?? 1) < 1e-6).toBe(true)
  })
  it('projected gradient onto a simplex with a custom projection', () => {
    const r = O.projectedGradient((x: Value) => sum(square(sub(x, tensor([0.8, 0.6, -0.2])))), [1, 0, 0], {
      project: (x) => Float64Array.from(toFlat(O.projectSimplex(tensor(Array.from(x))) as Value)),
      gradTol: 1e-10,
    })
    expect(toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8)).toEqual([0.6, 0.4, 0])
    for (const step of ['armijo', 'fixed'] as const) {
      const s = O.projectedGradient((x: Value) => sum(square(sub(x, 2))), [0, 0], {
        bounds: { upper: 1 },
        step,
        learningRate: 0.2,
      })
      expect(toFlat(s.x as Value)).toEqual([1, 1])
    }
  })
  it('barrier method rejects equality constraints and infeasible starts', () => {
    const disc = PROBLEMS['disc'] as Constrained
    expect(() => O.barrierMethod(disc.f, [2, 2], { constraints: disc.constraints })).toThrow(/strictly feasible/)
    expect(() =>
      O.barrierMethod(disc.f, [0, 0], { constraints: [{ type: 'eq', fun: (x: Value) => sum(x) }] }),
    ).toThrow()
  })
  it('constraint sets stack values and Jacobians', () => {
    const cs = O.constraintSet((PROBLEMS['rosen'] as Constrained).constraints, [0.5, 0])
    const j = cs.jacobians([0.5, 0])
    expect([cs.me, cs.mi]).toEqual([1, 3])
    expect(Array.from(j.eq)).toEqual([-2, -1])
    expect(j.ineq).toHaveLength(6)
  })
})

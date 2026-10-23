import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import {
  add,
  arctan2,
  concat,
  exp,
  mul,
  neg,
  pow,
  reshape,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import * as O from '../index.ts'

const at = (x: Value, i: number): Value => sliceAxis(x, 0, i, i + 1)

const expModel = (t: Value, p: Value): Value => add(mul(at(p, 0), exp(mul(neg(at(p, 1)), t))), at(p, 2))

function residuals(name: string, t: Tensor, y: Tensor): O.ResidualFunction {
  switch (name) {
    case 'rosenbrock':
      return (x: Value) => concat([mul(sub(at(x, 1), square(at(x, 0))), 10), sub(1, at(x, 0))])
    case 'helical':
      return (x: Value) => {
        const theta = mul(arctan2(at(x, 1), at(x, 0)), 1 / (2 * Math.PI))
        return concat([
          mul(sub(at(x, 2), mul(theta, 10)), 10),
          mul(sub(sqrt(add(square(at(x, 0)), square(at(x, 1)))), 1), 10),
          at(x, 2),
        ])
      }
    case 'beale':
      return (x: Value) => concat([1.5, 2.25, 2.625].map((c, k) => sub(c, mul(at(x, 0), sub(1, pow(at(x, 1), k + 1))))))
    default:
      return (x: Value) => sub(expModel(t, x), y)
  }
}

const solution = (r: O.LeastSquaresResult): Value => concat([r.x as Value, tensor([r.cost])])
const tight = { xtol: 1e-12, ftol: 1e-12, gradTol: 1e-12 }

describe('least squares against scipy.optimize', () => {
  const fixtures = [
    'least_squares',
    'levenberg_marquardt',
    'least_squares_bounded',
    'least_squares_trf',
    'least_squares_linear',
  ]
  for (const c of fixtures.flatMap((f) => loadFixture(`optim/minimize/${f}`).cases)) {
    it(c.name, () => {
      const inputs = c.inputs.map((v) => decode(v) as Tensor)
      switch (c.fn) {
        case 'levenbergMarquardt':
        case 'gaussNewton': {
          const [x0, t, y] = inputs as [Tensor, Tensor, Tensor]
          const f = residuals(c.params['residuals'] as string, t, y)
          const r = c.fn === 'gaussNewton' ? O.gaussNewton(f, x0, tight) : O.levenbergMarquardt(f, x0, tight)
          expect(r.converged).toBe(true)
          expectMatches(solution(r), c.output, c.rtol, c.atol, c.name)
          break
        }
        case 'trf':
        case 'dogbox': {
          const [x0, t, y, lb, ub] = inputs as [Tensor, Tensor, Tensor, Tensor | undefined, Tensor | undefined]
          const f = residuals(c.params['residuals'] as string, t, y)
          const loss = c.params['loss'] as O.RobustLoss | undefined
          const r = (c.fn === 'trf' ? O.trf : O.dogbox)(f, x0, {
            ...tight,
            ...(lb === undefined || ub === undefined ? {} : { bounds: { lower: toFlat(lb), upper: toFlat(ub) } }),
            ...(loss === undefined ? {} : { loss, fScale: 0.1 }),
          })
          expect(r.converged).toBe(true)
          const out = lb === undefined ? solution(r) : concat([solution(r), tensor(r.activeMask)])
          expectMatches(out, c.output, c.rtol, c.atol, c.name)
          break
        }
        case 'curveFit': {
          const [t, y, sigma] = inputs as [Tensor, Tensor, Tensor]
          const lower = c.params['lower'] as number[] | undefined
          const upper = c.params['upper'] as number[] | undefined
          const fit = O.curveFit(
            (tt, p) => expModel(tt, p as Value),
            t,
            y,
            lower === undefined ? [1, 1, 0] : [1, 1, 0.5],
            {
              ...(c.params['hasSigma'] === true ? { sigma } : {}),
              absoluteSigma: c.params['absoluteSigma'] as boolean,
              ...(lower === undefined || upper === undefined ? {} : { bounds: { lower, upper } }),
              ...tight,
            },
          )
          expectMatches(concat([fit.popt as Value, reshape(fit.pcov, [-1])]), c.output, c.rtol, c.atol, c.name)
          break
        }
        case 'nnls': {
          const [a, b] = inputs as [Tensor, Tensor]
          const r = O.nnls(a, b)
          expect(r.converged).toBe(true)
          expectMatches(concat([r.x, tensor([r.residualNorm])]), c.output, c.rtol, c.atol, c.name)
          break
        }
        case 'lsqLinear': {
          const [a, b, lb, ub] = inputs as [Tensor, Tensor, Tensor, Tensor]
          const r = O.lsqLinear(a, b, { bounds: { lower: toFlat(lb), upper: toFlat(ub) }, tol: 1e-12 })
          expect(r.converged).toBe(true)
          expectMatches(concat([r.x, tensor([r.cost])]), c.output, c.rtol, c.atol, c.name)
          break
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

describe('least squares', () => {
  it('reports the residuals, Jacobian and optimality', () => {
    const r = O.levenbergMarquardt((x: Value) => sub(mul(x, x), tensor([4, 9])), [1, 1])
    expect(r.residuals.shape).toEqual([2])
    expect(r.jacobian.shape).toEqual([2, 2])
    expect(r.optimality).toBeLessThan(1e-6)
    expect(r.njev).toBeGreaterThan(1)
  })
  it('levenbergMarquardt needs as many residuals as parameters', () => {
    expect(() => O.levenbergMarquardt((x: Value) => stack([add(at(x, 0), at(x, 1))]), [0, 0])).toThrow(/residuals/)
  })
  it('dogbox rejects a start outside the bounds', () => {
    expect(() => O.dogbox((x: Value) => x, [2], { bounds: [[0, 1]] })).toThrow(/outside/)
  })
  it('leastSquares defaults to the trust region reflective method and takes others', () => {
    const f = (x: Value): Value => sub(x, tensor([3, -3]))
    expect(O.leastSquares(f, [0, 0]).activeMask).toEqual([0, 0])
    expect(O.leastSquares(f, [0, 0], { bounds: { lower: -1, upper: 1 } }).activeMask).toEqual([1, -1])
    expect(O.leastSquares(f, [0, 0], { method: O.dogbox, bounds: { lower: -1, upper: 1 } }).activeMask).toEqual([1, -1])
  })
  it('curveFit reports infinite covariance when there are no degrees of freedom', () => {
    const fit = O.curveFit((t, p) => mul(p as Value, t), [1], [2], [1])
    expect(toFlat(fit.pcov)).toEqual([Infinity])
  })
})

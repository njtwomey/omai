import { describe, expect, it } from 'vitest'
import { grad, jacfwd, vmap } from '../autodiff/index.ts'
import { isWhiteboxError } from '../base/index.ts'
import { erf } from '../numerics/special/index.ts'
import { tensor, toArray, type Value } from '../tensor/index.ts'
import { compile, diff, evaluate, evaluateComplex, parse, sym } from './index.ts'

const x = sym('x')
const close = (a: number, b: number, tol = 1e-10): void => {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)))
}

describe('compile', () => {
  it('evaluates elementwise with broadcasting', () => {
    const f = compile(parse('x*y + 1'), ['x', 'y'])
    expect(toArray(f([1, 2, 3], 2))).toEqual([3, 5, 7])
    const c = compile(parse('3'), ['x'])
    expect(toArray(c([1, 2]))).toEqual([3, 3])
  })

  it('symbolic diff agrees with grad of the compiled function', () => {
    const exprs = [
      'x^3*sin(x)',
      'exp(-x^2/2)/sqrt(2*pi)',
      'log(1 + x^2)*atan(x)',
      'tanh(x)^2 + cosh(x)',
      'sqrt(x^2 + 1)/(x + 3)',
      'x^x',
      'Piecewise((x^2, x < 1), (2*x - 1, True))',
      'asin(x/3) + acos(x/4) + sec(x)',
    ]
    for (const s of exprs) {
      const e = parse(s)
      const f = compile(e, [x])
      const df = compile(diff(e, x), [x])
      for (const v of [0.3, 0.9, 1.7]) {
        const g = toArray(grad((t: Value) => f(t))(tensor(v))) as number
        close(g, toArray(df(v)) as number, 1e-9)
      }
    }
  })

  it('second derivatives agree with nested grad', () => {
    const e = parse('sin(x)*exp(x)')
    const f = compile(e, [x])
    const d2 = compile(diff(e, [x, 2]), [x])
    const g2 = grad((t: Value) => grad((u: Value) => f(u))(t))(tensor(0.4))
    close(toArray(g2) as number, toArray(d2(0.4)) as number, 1e-9)
  })

  it('gradients of several variables and vectorisation', () => {
    const e = parse('x^2*y + sin(x*y)')
    const f = compile(e, ['x', 'y'])
    const gx = compile(diff(e, 'x'), ['x', 'y'])
    const gy = compile(diff(e, 'y'), ['x', 'y'])
    const g = grad((p: { x: Value; y: Value }) => f(p.x, p.y))({ x: tensor(0.7), y: tensor(-1.2) })
    close(toArray(g.x) as number, toArray(gx(0.7, -1.2)) as number)
    close(toArray(g.y) as number, toArray(gy(0.7, -1.2)) as number)
    const batched = vmap((t: Value) => f(t, 2))(tensor([0.1, 0.2, 0.3]))
    expect((toArray(batched) as number[]).length).toBe(3)
    const j = jacfwd((t: Value) => f(t, 2))(tensor([0.1, 0.2]))
    expect(toArray(j)).toHaveLength(2)
  })

  it('takes implementations of erf and undefined functions', () => {
    const f = compile(parse('erf(x) + g(x)'), [x], { functions: { erf, g: (v: Value) => v } })
    close(toArray(f(0.5)) as number, 0.5204998778130465 + 0.5, 1e-12)
    expect(() => compile(parse('erf(x)'), [x])(0.5)).toThrow()
  })

  it('rejects unbound symbols and complex constants', () => {
    expect(() => compile(parse('x + y'), [x])).toThrow()
    try {
      compile(parse('I*x'), [x])(1)
      expect.unreachable()
    } catch (e) {
      expect(isWhiteboxError(e) && e.code === 'not-implemented').toBe(true)
    }
  })
})

describe('evaluate', () => {
  it('evaluates to doubles and complex numbers', () => {
    expect(evaluate(parse('x^2 + 1'), { x: 3 })).toBe(10)
    expect(Number.isNaN(evaluate(parse('sqrt(x)'), { x: -1 }))).toBe(true)
    expect(evaluateComplex(parse('sqrt(x)'), { x: -4 })).toEqual([0, 2])
    expect(evaluateComplex(parse('exp(I*pi)'))).toEqual([-1, 0])
    close(evaluate(parse('erf(x)'), { x: 1 }, { erf: Math.tanh }), Math.tanh(1))
    expect(() => evaluate(parse('x + y'), { x: 1 })).toThrow()
  })
})

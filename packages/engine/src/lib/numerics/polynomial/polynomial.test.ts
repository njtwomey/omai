import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { item, sum, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as P from './index.ts'

const canonical = (f: P.PartialFractions): number[][] => {
  const rows = f.poles.re.map((re, i) => {
    const im = f.poles.im[i] ?? 0
    let power = 1
    for (let j = i - 1; j >= 0 && f.poles.re[j] === re && f.poles.im[j] === im; j--) power++
    return [re, im, power, f.residues.re[i] ?? 0, f.residues.im[i] ?? 0]
  })
  return rows.sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || (a[1] ?? 0) - (b[1] ?? 0) || (a[2] ?? 0) - (b[2] ?? 0))
}

describe('polynomials against numpy and scipy', () => {
  for (const c of loadFixture('numerics/polynomial/polynomial').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const f = (i: number): number[] => toFlat(t[i] as Tensor)
      const p = c.params
      const family = p['family'] as P.Family
      const jacobi = { alpha: p['alpha'] as number, beta: p['beta'] as number }
      const actual: Value | number[] | number[][] =
        c.fn === 'polyval'
          ? P.polyval(t[0] as Tensor, t[1] as Tensor)
          : c.fn === 'polyfit'
            ? P.polyfit(t[0] as Tensor, t[1] as Tensor, p['deg'] as number, t[2])
            : c.fn === 'polyder'
              ? P.polyder(f(0), p['m'] as number, p['scl'] as number)
              : c.fn === 'polyint'
                ? P.polyint(f(0), 2, [1, -2], 0.5)
                : c.fn === 'polymul'
                  ? P.polymul(f(0), f(1))
                  : c.fn === 'polydiv'
                    ? (({ quotient, remainder }) => [...toFlat(quotient), ...toFlat(remainder)])(P.polydiv(f(0), f(1)))
                    : c.fn === 'polyroots'
                      ? (({ re, im }) => [toFlat(re), toFlat(im)])(P.polyroots(f(0)))
                      : c.fn === 'polyfromroots'
                        ? P.polyFromRoots(f(0))
                        : c.fn === 'orthoval'
                          ? P.orthoval(family, t[0] as Tensor, t[1] as Tensor, jacobi)
                          : c.fn === 'orthofit'
                            ? P.orthofit(family, t[0] as Tensor, t[1] as Tensor, 5)
                            : c.fn === 'orthoToPower'
                              ? P.orthoToPower(family, f(0))
                              : c.fn === 'powerToOrtho'
                                ? P.powerToOrtho(family, f(0))
                                : c.fn === 'residue'
                                  ? canonical(P.residue(f(0), f(1)))
                                  : c.fn === 'residueDirect'
                                    ? P.residue(f(0), f(1)).direct.length === 0
                                      ? []
                                      : P.residue(f(0), f(1)).direct
                                    : c.fn === 'residuez'
                                      ? canonical(P.residuez(f(0), f(1)))
                                      : c.fn === 'residuezDirect'
                                        ? P.residuez(f(0), f(1)).direct
                                        : (() => {
                                            const r = P.residue(f(0), f(1))
                                            const back = P.invres(r.residues, r.poles, r.direct)
                                            return [...back.numerator.re, ...back.denominator.re]
                                          })()
      expectMatches(Array.isArray(actual) ? tensor(actual as number[]) : actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('polynomial properties', () => {
  it('polyval is differentiable in points and coefficients', () => {
    expect(item(grad((x: Value) => P.polyval(x, [1, 2, 3]))(tensor(2)) as Value)).toBe(14)
    expect(toFlat(grad((c: Value) => P.polyval(2, c))(tensor([1, 2, 3])) as Value)).toEqual([1, 2, 4])
    const g = toFlat(grad((c: Value) => sum(P.orthoval('legendre', [0.5], c)))(tensor([1, 1, 1])) as Value)
    expect(g[2]).toBeCloseTo(-0.125, 14)
  })

  it('conversions round-trip and Gauss nodes are the roots of the degree-n member', () => {
    for (const family of ['legendre', 'chebyshev', 'hermite', 'hermite_e', 'laguerre'] as const) {
      const c = [0.3, -1, 2, 0.5, -0.7]
      for (const [i, v] of toFlat(P.powerToOrtho(family, P.orthoToPower(family, c))).entries())
        expect(v).toBeCloseTo(c[i] ?? 0, 10)
      const nodes = Array.from(P.orthogauss(family, 5).nodes.data)
      const member = toFlat(P.orthoval(family, nodes, [0, 0, 0, 0, 0, 1]))
      const scale = Math.max(
        ...toFlat(
          P.orthoval(
            family,
            nodes.map((v) => v + 1),
            [0, 0, 0, 0, 0, 1],
          ),
        ).map(Math.abs),
      )
      for (const v of member) expect(Math.abs(v) / scale).toBeLessThan(1e-9)
    }
  })

  it('polydiv and polymul invert each other; roots rebuild the polynomial', () => {
    const a = [3, -1, 0, 2, 5]
    const b = [1, 2]
    const { quotient, remainder } = P.polydiv(a, b)
    for (const [i, v] of toFlat(P.polyadd(P.polymul(quotient, b), remainder)).entries()) {
      expect(v).toBeCloseTo(a[i] ?? 0, 12)
    }
    const rebuilt = toFlat(P.polyFromRoots(P.polyroots(a))).map((v) => v * 5)
    for (const [i, v] of rebuilt.entries()) expect(v).toBeCloseTo(a[i] ?? 0, 9)
  })
})

describe('the power-basis functions take tensor-likes and return tensors (migration, D-009)', () => {
  const calls: [string, (c: TensorLike) => Value | number[]][] = [
    ['polytrim', (c) => P.polytrim(c)],
    ['polyder', (c) => P.polyder(c)],
    ['polyint', (c) => P.polyint(c, 1, [2])],
    ['polyadd', (c) => P.polyadd(c, [1, 1])],
    ['polysub', (c) => P.polysub(c, [1, 1])],
    ['polymul', (c) => P.polymul(c, [1, 1])],
    ['polydiv', (c) => P.polydiv(c, [1, 1]).remainder],
    ['companion', (c) => P.companion(c)],
    ['polyroots', (c) => P.polyroots(c).re],
    ['polyFromRoots', (c) => P.polyFromRoots(c)],
    ['orthoToPower', (c) => P.orthoToPower('legendre', c)],
    ['powerToOrtho', (c) => P.powerToOrtho('legendre', c)],
  ]
  it('a tensor, a typed array and an array agree; wrong rank is a shape error; traced is rejected', () => {
    for (const [name, f] of calls) {
      const typed = Float64Array.from([2, 1, 3])
      const expected = toFlat(f([2, 1, 3]) as Value)
      expect(toFlat(f(typed) as Value), name).toEqual(expected)
      expect(toFlat(f(tensor([2, 1, 3])) as Value), name).toEqual(expected)
      expect(
        () =>
          f(
            tensor([
              [1, 2],
              [3, 4],
            ]),
          ),
        name,
      ).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(() => grad((c: Value) => sum(f(c as TensorLike) as Value))(tensor([2, 1, 3])), name).toThrow(
        expect.objectContaining({ code: 'not-differentiable' }),
      )
    }
  })
})

describe('charpoly', () => {
  const a = tensor([
    [1, 2, 0.5],
    [-0.3, 0.7, 1.1],
    [0.4, -1.2, 2],
  ])

  it('matches det(xI - A) at sample points and has a conformant derivative', () => {
    const c = toFlat(P.charpoly(a) as Tensor)
    expect(c[3]).toBe(1)
    const det3 = (m: readonly number[]): number => {
      const [p, q, r, s, t, u, v, w, z] = m.map((x) => x)
      const f = (x: number | undefined): number => x ?? 0
      return (
        f(p) * (f(t) * f(z) - f(u) * f(w)) - f(q) * (f(s) * f(z) - f(u) * f(v)) + f(r) * (f(s) * f(w) - f(t) * f(v))
      )
    }
    for (const x of [-1.5, 0, 0.8, 2.3]) {
      const value = (c[0] ?? 0) + x * ((c[1] ?? 0) + x * ((c[2] ?? 0) + x * (c[3] ?? 0)))
      expect(value).toBeCloseTo(det3(toFlat(a).map((v, k) => (k % 4 === 0 ? x - v : -v))), 12)
    }
    expect(toFlat(P.charpoly([[0]] as number[][]) as Tensor)).toEqual([0, 1])
    const report = checkPrimitive((x) => P.charpoly(x), [a], { vmap: false })
    expect(report.failures).toEqual([])
  })
})

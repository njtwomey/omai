import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import {
  charpoly,
  det,
  eigenvals,
  eigenvects,
  evaluateComplex,
  identityMatrix,
  inverse,
  matMul,
  matrix,
  nullspace,
  parse,
  sym,
  toLatex,
  toText,
  transpose,
} from './index.ts'

const m = (rows: string[][]): ReturnType<typeof matrix> => matrix(rows.map((r) => r.map((v) => parse(v))))
const text = (a: ReturnType<typeof matrix>): string[][] => a.map((r) => r.map(toText))

describe('matrices', () => {
  it('multiplies, transposes and inverts', () => {
    const a = m([
      ['a', 'b'],
      ['c', 'd'],
    ])
    expect(text(transpose(a))).toEqual([
      ['a', 'c'],
      ['b', 'd'],
    ])
    expect(text(inverse(a))).toEqual([
      ['d/(a*d - b*c)', '-b/(a*d - b*c)'],
      ['-c/(a*d - b*c)', 'a/(a*d - b*c)'],
    ])
    const b = m([
      ['2', '1', '0'],
      ['1', '2', '1'],
      ['0', '1', '2'],
    ])
    expect(text(matMul(b, inverse(b)))).toEqual(text(identityMatrix(3)))
    expect(() =>
      inverse(
        m([
          ['1', '2'],
          ['2', '4'],
        ]),
      ),
    ).toThrow()
    expect(() => det(matrix([[1, 2]]))).toThrow()
  })
  it('computes determinants and characteristic polynomials', () => {
    expect(
      toText(
        det(
          m([
            ['x', '1', '0'],
            ['0', 'x', '1'],
            ['1', '0', 'x'],
          ]),
        ),
      ),
    ).toBe('x**3 + 1')
    expect(
      toText(
        charpoly(
          m([
            ['1', '2'],
            ['3', '4'],
          ]),
          sym('lam'),
        ),
      ),
    ).toBe('lam**2 - 5*lam - 2')
    expect(toText(det(identityMatrix(0)))).toBe('1')
  })
  it('finds eigenvalues, eigenvectors and null spaces', () => {
    const vals = eigenvals(
      m([
        ['2', '1'],
        ['1', '2'],
      ]),
    )
    expect(vals.map((v) => toText(v.value))).toEqual(['1', '3'])
    const jordan = eigenvals(
      m([
        ['3', '1'],
        ['0', '3'],
      ]),
    )
    expect(jordan).toEqual([{ value: parse('3'), multiplicity: 2 }])
    const vecs = eigenvects(
      m([
        ['3', '1'],
        ['0', '3'],
      ]),
    )
    expect(vecs[0]?.vectors.map((v) => v.map(toText))).toEqual([['1', '0']])
    const rot = eigenvals(
      m([
        ['0', '-1'],
        ['1', '0'],
      ]),
    ).map((v) => toText(v.value))
    expect(rot).toEqual(['-I', 'I'])
    // Eigenvectors with irrational eigenvalues satisfy A v = lambda v.
    const a = m([
      ['1', '2'],
      ['3', '4'],
    ])
    for (const e of eigenvects(a)) {
      const v = e.vectors[0] ?? []
      const av = matMul(
        a,
        v.map((c) => [c]),
      )
      av.forEach((row, i) => {
        const lhs = evaluateComplex(row[0] ?? parse('0'))
        const rhs = evaluateComplex(parse(`(${toText(e.value)})*(${toText(v[i] ?? parse('0'))})`))
        expect(Math.hypot(lhs[0] - rhs[0], lhs[1] - rhs[1])).toBeLessThan(1e-9)
      })
    }
    expect(
      nullspace(
        m([
          ['1', '2'],
          ['2', '4'],
        ]),
      ).map((v) => v.map(toText)),
    ).toEqual([['-2', '1']])
  })
  it('typesets matrices as SymPy', () => {
    expect(
      toLatex(
        m([
          ['1', 'x'],
          ['y', '2'],
        ]),
      ),
    ).toBe('\\left[\\begin{matrix}1 & x\\\\y & 2\\end{matrix}\\right]')
  })
  it('rejects ragged rows', () => {
    try {
      matrix([[1, 2], [3]])
      expect.unreachable()
    } catch (e) {
      expect(isWhiteboxError(e) && e.code === 'shape').toBe(true)
    }
  })
})

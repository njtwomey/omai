import { describe, expect, it } from 'vitest'
import { assertFinite, assertShape, invariant, isOmaiError, type OmaiError } from './index.ts'

describe('the error helpers', () => {
  it('invariant, assertFinite and assertShape throw a coded OmaiError with the details, and pass what is right', () => {
    expect(() => invariant(1 + 1 === 2, 'value', 'arithmetic')).not.toThrow()
    const failed = (f: () => unknown): OmaiError => {
      try {
        f()
      } catch (e) {
        return e as OmaiError
      }
      throw new Error('did not throw')
    }
    const broken = failed(() => invariant(false, 'domain', 'x must be positive', { x: -1 }))
    expect([broken.code, broken.message, broken.details, broken.name, broken instanceof Error]).toEqual([
      'domain',
      'x must be positive',
      { x: -1 },
      'OmaiError',
      true,
    ])
    // assertFinite returns its argument and names it
    expect([assertFinite(2.5, 'scale'), assertFinite(-0, 'z'), assertFinite(1e308, 'big')]).toEqual([2.5, -0, 1e308])
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const e = failed(() => assertFinite(bad, 'scale'))
      expect([e.code, e.message, e.details]).toEqual(['value', `scale must be finite, got ${bad}`, { scale: bad }])
    }
    // -1 in the pattern matches any length, and the rank must agree
    expect(() => assertShape([3, 2], [-1, 2], 'x')).not.toThrow()
    expect(() => assertShape([], [], 'scalar')).not.toThrow()
    const rows: [number[], number[]][] = [
      [
        [3, 2],
        [2, 2],
      ],
      [[3, 2], [3]],
      [[3], [3, 2]],
      [
        [0, 2],
        [1, 2],
      ],
    ]
    for (const [shape, expected] of rows) {
      const e = failed(() => assertShape(shape, expected, 'x'))
      expect([e.code, e.message]).toEqual([
        'shape',
        `x must have shape [${expected.join(', ')}], got [${shape.join(', ')}]`,
      ])
      expect(e.details).toEqual({ shape, expected })
    }
    // isOmaiError narrows by code and ignores other errors
    expect([
      isOmaiError(broken),
      isOmaiError(broken, 'domain'),
      isOmaiError(broken, 'value'),
      isOmaiError(new Error('x')),
      isOmaiError('x'),
    ]).toEqual([true, true, false, false, false])
  })
})

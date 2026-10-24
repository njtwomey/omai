import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { rect } from './build.ts'
import { applyTransform, bounds, composeTransforms, invertTransform, rotate, scaleTo, translate } from './geometry.ts'
import type { Transform } from './types.ts'

const num = fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true })
const point = fc.tuple(num, num)
const transform: fc.Arbitrary<Transform> = fc.record({
  translate: fc.tuple(num, num),
  rotate: fc.double({ min: -360, max: 360, noNaN: true }),
  scale: fc.double({ min: 0.1, max: 5, noNaN: true }),
})

const near = (a: readonly number[], b: readonly number[], tol = 1e-7): void => {
  expect(a.length).toBe(b.length)
  a.forEach((x, i) => {
    expect(Math.abs(x - (b[i] as number))).toBeLessThan(tol * Math.max(1, Math.abs(x)))
  })
}

describe('transform properties', () => {
  it('composing transforms is applying one after the other', () => {
    fc.assert(
      fc.property(transform, transform, point, (a, b, p) => {
        near(applyTransform(composeTransforms(a, b), p), applyTransform(a, applyTransform(b, p)), 1e-6)
      }),
    )
  })

  it('composition is associative', () => {
    fc.assert(
      fc.property(transform, transform, transform, point, (a, b, c, p) => {
        const left = composeTransforms(composeTransforms(a, b), c)
        const right = composeTransforms(a, composeTransforms(b, c))
        near(applyTransform(left, p), applyTransform(right, p), 1e-6)
      }),
    )
  })

  it('the inverse undoes a transform', () => {
    fc.assert(
      fc.property(transform, point, (t, p) => {
        near(applyTransform(invertTransform(t), applyTransform(t, p)), p, 1e-6)
      }),
    )
  })

  it('translate shifts bounds and rotate keeps the centre of rotation fixed', () => {
    fc.assert(
      fc.property(num, num, num, num, (x, y, dx, dy) => {
        const r = rect(x, y, 3, 2)
        const b = bounds([translate(r, dx, dy)])
        near([b.x, b.y, b.width, b.height], [x + dx, y + dy, 3, 2], 1e-9)
        const g = rotate(r, 37, [x, y])
        near(applyTransform(g, [x, y]), [x, y], 1e-9)
      }),
    )
  })

  it('translating a group twice composes into one group', () => {
    fc.assert(
      fc.property(num, num, num, num, (a, b, c, d) => {
        const g = translate(translate(rect(0, 0, 1, 1), a, b), c, d)
        expect(g.children[0]?.kind).toBe('rect')
        near(applyTransform(g, [0, 0]), [a + c, b + d], 1e-9)
      }),
    )
  })

  it('scaleTo never exceeds the requested size and keeps the corner', () => {
    fc.assert(
      fc.property(
        num,
        num,
        fc.double({ min: 0.5, max: 20, noNaN: true }),
        fc.double({ min: 0.5, max: 20, noNaN: true }),
        fc.double({ min: 1, max: 30, noNaN: true }),
        (x, y, w, h, target) => {
          const b = bounds([scaleTo(rect(x, y, w, h), { width: target, height: target })])
          expect(b.width).toBeLessThanOrEqual(target * (1 + 1e-9))
          expect(b.height).toBeLessThanOrEqual(target * (1 + 1e-9))
          expect(Math.max(b.width, b.height)).toBeCloseTo(target, 6)
          near([b.x, b.y], [x, y], 1e-9)
        },
      ),
    )
  })
})

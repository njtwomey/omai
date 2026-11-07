import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Vec3 } from '../base/index.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import {
  box,
  intersectBox,
  intersectPlane,
  intersectSphere,
  intersectTriangle,
  plane,
  sphere,
  triangle,
} from './index.ts'

const arr = (x: unknown): number[][] => toArray(decode(x) as Tensor) as number[][]
const vec = (x: unknown): Vec3 => toArray(decode(x) as Tensor) as unknown as Vec3
const table = (rows: readonly (readonly number[])[]): Tensor =>
  decode({
    dtype: 'float64',
    shape: [rows.length, rows[0]?.length ?? 0],
    data: rows.flat().map((v) => (Number.isNaN(v) ? 'nan' : v)),
  }) as Tensor

describe('intersections against closed forms', () => {
  for (const c of loadFixture('graphics/intersect').cases) {
    it(c.name, () => {
      const origins = arr(c.inputs[0])
      const dirs = arr(c.inputs[1])
      const rays = origins.map((o, i) => ({ origin: o as unknown as Vec3, direction: dirs[i] as unknown as Vec3 }))
      const nan4 = [Number.NaN, Number.NaN, Number.NaN, Number.NaN]
      let rows: number[][]
      switch (c.fn) {
        case 'sphere': {
          const s = sphere(vec(c.inputs[2]), c.params['radius'] as number)
          rows = rays.map((r) => {
            const h = intersectSphere(r, s)
            return h === null ? nan4 : [h.t, ...h.normal]
          })
          break
        }
        case 'plane': {
          const s = plane(vec(c.inputs[2]), vec(c.inputs[3]))
          rows = rays.map((r) => [intersectPlane(r, s)?.t ?? Number.NaN])
          break
        }
        case 'triangle': {
          const s = triangle(vec(c.inputs[2]), vec(c.inputs[3]), vec(c.inputs[4]))
          rows = rays.map((r) => {
            const h = intersectTriangle(r, s)
            return h === null ? [Number.NaN, Number.NaN, Number.NaN] : [h.t, h.uv[0], h.uv[1]]
          })
          break
        }
        case 'box': {
          const s = box(vec(c.inputs[2]), vec(c.inputs[3]))
          rows = rays.map((r) => {
            const h = intersectBox(r, s)
            return h === null ? nan4 : [h.t, ...h.normal]
          })
          break
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
      expectMatches(table(rows), c.output, c.rtol, c.atol, c.name)
    })
  }
})

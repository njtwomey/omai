import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Vec3 } from '../base/index.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import {
  cameraFrustum,
  lookAt,
  orthographic,
  orthographicCamera,
  perspective,
  pinholeCamera,
  pixelRay,
  projectToPixel,
} from './index.ts'

const vec = (x: unknown): Vec3 => toArray(decode(x) as Tensor) as unknown as Vec3

describe('cameras against numpy', () => {
  for (const c of loadFixture('graphics/camera').cases) {
    it(c.name, () => {
      const p = c.params
      switch (c.fn) {
        case 'lookAt':
          expectMatches(
            matrix(lookAt(vec(c.inputs[0]), vec(c.inputs[1]), vec(c.inputs[2]))),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        case 'perspective': {
          const m = perspective(p['fov'] as number, p['aspect'] as number, p['near'] as number, p['far'] as number)
          expectMatches(matrix(m), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'orthographic': {
          const [l, r, b, t, n, f] = p['box'] as number[]
          expectMatches(
            matrix(orthographic(l ?? 0, r ?? 0, b ?? 0, t ?? 0, n ?? 0, f ?? 0)),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        }
        case 'view': {
          const [eye, target, up] = [vec(c.inputs[0]), vec(c.inputs[1]), vec(c.inputs[2])]
          const width = p['width'] as number
          const height = p['height'] as number
          const common = { eye, target, up, near: p['near'] as number, far: p['far'] as number }
          const cam =
            p['kind'] === 'pinhole'
              ? pinholeCamera({ ...common, fov: p['fov'] as number })
              : orthographicCamera({ ...common, height: p['size'] as number })
          const points = toArray(decode(c.inputs[3]) as Tensor) as number[][]
          const projected = points.map((q) => {
            const r = projectToPixel(cam, q as unknown as Vec3, width, height)
            return [r?.x ?? Number.NaN, r?.y ?? Number.NaN, r?.ndcZ ?? Number.NaN]
          })
          expectMatches(matrix(projected), c.output, c.rtol, c.atol, c.name)
          const coords = toArray(decode(c.inputs[4]) as Tensor) as number[][]
          for (const [i, [x = 0, y = 0]] of coords.entries()) {
            const ray = pixelRay(cam, width, height, x, y)
            const o = (p['origins'] as number[][])[i] ?? []
            const d = (p['directions'] as number[][])[i] ?? []
            for (let k = 0; k < 3; k++) {
              expect(ray.origin[k]).toBeCloseTo(o[k] ?? Number.NaN, 9)
              expect(ray.direction[k]).toBeCloseTo(d[k] ?? Number.NaN, 9)
            }
          }
          const planes = cameraFrustum(cam, width / height)
          for (const [i, want] of (p['planes'] as number[][]).entries()) {
            for (let k = 0; k < 4; k++) expect(planes[i]?.[k]).toBeCloseTo(want[k] ?? Number.NaN, 8)
          }
          return
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

function matrix(rows: readonly (readonly number[])[]): Tensor {
  return decode({ dtype: 'float64', shape: [rows.length, rows[0]?.length ?? 0], data: rows.flat() }) as Tensor
}

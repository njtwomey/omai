import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { cubature, productGaussRule } from '../numerics/quadrature/index.ts'
import { cos, mul, sin, sliceAxis, squeeze, stack, type Tensor, type Value } from '../tensor/index.ts'
import {
  blinnPhong,
  cookTorrance,
  fresnelDielectric,
  fresnelSchlick,
  ggxDistribution,
  metallicRoughness,
  phong,
  phongBrdf,
  schlick,
  smithG1,
} from './index.ts'

const input = (x: unknown): Tensor => decode(x) as Tensor

// The integral over the upper hemisphere of f(directions [m, 3]) cos(theta), by a product Gauss-Legendre rule.
function directionalAlbedo(f: (d: Value) => Value): Value {
  const rule = productGaussRule(
    [
      [0, Math.PI / 2],
      [0, 2 * Math.PI],
    ],
    96,
  )
  return cubature((pts: Value) => {
    const theta = squeeze(sliceAxis(pts, -1, 0, 1), -1)
    const phi = squeeze(sliceAxis(pts, -1, 1, 2), -1)
    const d = stack([mul(sin(theta), cos(phi)), mul(sin(theta), sin(phi)), cos(theta)], -1)
    return mul(mul(f(d), cos(theta)), sin(theta))
  }, rule)
}

describe('shading against closed forms and scipy quadrature', () => {
  for (const c of loadFixture('graphics/shading').cases) {
    it(c.name, () => {
      const p = c.params
      const x = c.inputs.map(input)
      const n = [0, 0, 1]
      let got: Value
      switch (c.fn) {
        case 'fresnel':
          got = fresnelDielectric(x[0] as Tensor, p['n1'] as number, p['n2'] as number)
          break
        case 'schlickDielectric':
          got = schlick(x[0] as Tensor, p['n1'] as number, p['n2'] as number)
          break
        case 'schlick':
          got = fresnelSchlick(x[0] as Tensor, x[1] as Tensor)
          break
        case 'ggx':
          got = ggxDistribution(x[0] as Tensor, x[1] as Tensor)
          break
        case 'smith':
          got = smithG1(x[0] as Tensor, x[1] as Tensor)
          break
        case 'phong':
          got = phong(x[0] as Tensor, x[1] as Tensor, x[2] as Tensor, p['shininess'] as number)
          break
        case 'blinnPhong':
          got = blinnPhong(x[0] as Tensor, x[1] as Tensor, x[2] as Tensor, p['shininess'] as number)
          break
        case 'cookTorrance':
          got = cookTorrance(x[0] as Tensor, x[1] as Tensor, x[2] as Tensor, p['roughness'] as number, x[3] as Tensor)
          break
        case 'metallicRoughness':
          got = metallicRoughness(
            x[0] as Tensor,
            x[1] as Tensor,
            x[2] as Tensor,
            x[3] as Tensor,
            p['roughness'] as number,
            p['metallic'] as number,
          )
          break
        case 'albedoPhong':
          got = directionalAlbedo((d) => phongBrdf(n, x[0] as Tensor, d, p['shininess'] as number))
          break
        case 'albedoCookTorrance':
          got = directionalAlbedo((d) => squeeze(cookTorrance(n, x[0] as Tensor, d, p['roughness'] as number, 1), -1))
          break
        case 'ggxNormalisation':
          got = directionalAlbedo((d) => ggxDistribution(squeeze(sliceAxis(d, -1, 2, 3), -1), p['alpha'] as number))
          break
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

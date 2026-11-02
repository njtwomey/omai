import { describe, expect, it } from 'vitest'
import { gradCheck, valueAndGrad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { mul, sum, tensor, toFlat, treeLeaves, type Value } from '../tensor/index.ts'
import { NerfField, nerfDensityGrid, nerfModel, nerfParams, renderRays } from './index.ts'

const small = {
  width: 8,
  depth: 2,
  positionEncoding: { kind: 'frequency', octaves: 3, includeInput: true },
  directionEncoding: { kind: 'frequency', octaves: 2, includeInput: true },
} as const

describe('NerfField', () => {
  it('maps position and direction to a non-negative density and a colour in (0, 1)', () => {
    const net = new NerfField({ ...small, key: key(0) })
    const x = tensor([
      [0.1, 0.2, 0.3],
      [-1, 0.5, 0.2],
    ])
    const d = tensor([
      [0, 0, 1],
      [0, 1, 0],
    ])
    const out = net.forward([x, d] as never)
    expect(out.density.shape).toEqual([2])
    expect(out.colour.shape).toEqual([2, 3])
    expect(toFlat(out.density).every((v) => v > 0)).toBe(true)
    expect(toFlat(out.colour).every((v) => v > 0 && v < 1)).toBe(true)
    expect(net.outputShape([5, 3])).toEqual([5, 4])
  })

  it('has a density that depends on position alone, and a colour that depends on the direction', () => {
    const net = new NerfField({ ...small, key: key(1) })
    const x = tensor([[0.3, -0.2, 0.5]])
    const a = net.forward([x, tensor([[0, 0, 1]])] as never)
    const b = net.forward([x, tensor([[1, 0, 0]])] as never)
    expect(toFlat(a.density)).toEqual(toFlat(b.density))
    expect(toFlat(a.colour)).not.toEqual(toFlat(b.colour))
  })

  it('uses the relu density of the paper when asked, and a hash grid for the position when asked', () => {
    const relu = new NerfField({ ...small, densityActivation: 'relu', key: key(2) })
    const out = relu.forward([tensor([[0.3, -0.2, 0.5]]), tensor([[0, 0, 1]])] as never)
    expect(toFlat(out.density)[0]).toBeGreaterThanOrEqual(0)
    const hash = new NerfField({
      ...small,
      positionEncoding: { kind: 'hash', levels: 3, log2TableSize: 6, bound: 1.5 },
      key: key(2),
    })
    expect(Object.keys(hash.params)).toContain('positionEncoding')
    expect(hash.forward([tensor([[0.3, -0.2, 0.5]]), tensor([[0, 0, 1]])] as never).colour.shape).toEqual([1, 3])
  })
})

describe('renderRays', () => {
  const rays = { origins: [0, 0, 4, 0.5, 0, 4], directions: [0, 0, -1, 0, 0, -1] }

  it('renders one colour per ray with a coarse pass, and a fine pass over the merged samples with a fine network', () => {
    const coarse = nerfModel({ ...small, key: key(0) })
    const one = renderRays(coarse, nerfParams(coarse), rays, { near: 2, far: 6, samples: 6 })
    expect(one.coarse.colour.shape).toEqual([2, 3])
    expect(one.coarse.weights.shape).toEqual([2, 6])
    expect(one.fine).toBeUndefined()
    const both = nerfModel({ ...small, fine: true, key: key(0) })
    const two = renderRays(both, nerfParams(both), rays, { near: 2, far: 6, samples: 6, fineSamples: 4, key: key(3) })
    expect(two.fine?.colour.shape).toEqual([2, 3])
    expect(two.fine?.t.shape).toEqual([2, 10])
    const t = toFlat(two.fine?.t as Value)
    expect(t.slice(0, 10).every((v, i) => i === 0 || v >= (t[i - 1] as number))).toBe(true)
  })

  it('is deterministic without a key and differs with one, and composites white on request', () => {
    const model = nerfModel({ ...small, key: key(0) })
    const a = renderRays(model, nerfParams(model), rays, { near: 2, far: 6, samples: 6 })
    const b = renderRays(model, nerfParams(model), rays, { near: 2, far: 6, samples: 6 })
    expect(toFlat(a.coarse.colour)).toEqual(toFlat(b.coarse.colour))
    const c = renderRays(model, nerfParams(model), rays, { near: 2, far: 6, samples: 6, key: key(1) })
    expect(toFlat(c.coarse.colour)).not.toEqual(toFlat(a.coarse.colour))
    const white = renderRays(model, nerfParams(model), rays, { near: 2, far: 6, samples: 6, whiteBackground: true })
    const alpha = toFlat(white.coarse.alpha)
    for (const [i, v] of toFlat(white.coarse.colour).entries())
      expect(v).toBeCloseTo((toFlat(a.coarse.colour)[i] as number) + 1 - (alpha[Math.floor(i / 3)] as number), 12)
  })

  it('is differentiable in the parameters of both networks', () => {
    const model = nerfModel({ ...small, fine: true, key: key(0) })
    const target = tensor([
      [0.2, 0.4, 0.6],
      [0.6, 0.4, 0.2],
    ])
    const objective = (p: ReturnType<typeof nerfParams>): Value => {
      const r = renderRays(model, p, rays, { near: 2, far: 6, samples: 5, fineSamples: 3 })
      return sum(mul(r.fine?.colour as Value, target))
    }
    const { grad } = valueAndGrad(objective)(nerfParams(model))
    const leaves = treeLeaves(grad)
    expect(leaves.length).toBeGreaterThan(10)
    expect(leaves.every((l) => toFlat(l).every(Number.isFinite))).toBe(true)
    expect(leaves.some((l) => toFlat(l).some((v) => v !== 0))).toBe(true)
    // And in a parameter, against finite differences.
    const w = model.coarse.params as Record<string, Record<string, Value>>
    const head = w['colourHead']?.['bias'] as Value
    const f = (b: Value): Value => {
      const p = nerfParams(model)
      const next = { ...p, coarse: { ...p.coarse, colourHead: { ...(p.coarse['colourHead'] as object), bias: b } } }
      return sum(renderRays(model, next, rays, { near: 2, far: 6, samples: 5 }).coarse.colour)
    }
    expect(gradCheck(f, head).passed).toBe(true)
  })
})

describe('nerfDensityGrid', () => {
  it('evaluates the field at the cell centres of a box, x fastest, with the density that renderRays uses', () => {
    const model = nerfModel({ ...small, key: key(3) })
    const params = nerfParams(model)
    const g = nerfDensityGrid(model, params, {
      bounds: [
        [0, 0, 0],
        [2, 4, 6],
      ],
      resolution: 2,
    })
    expect(g.density.length).toBe(8)
    expect(g.colour.length).toBe(24)
    // Cell (1, 0, 1) is index 1 + 2 (0 + 2 * 1) = 5, at the point (1.5, 1, 4.5).
    const out = model.coarse.apply(params.coarse, [tensor([[1.5, 1, 4.5]]), tensor([[0, 0, -1]])] as never, {
      train: false,
    })
    expect(g.density[5]).toBeCloseTo(toFlat(out.density)[0] as number, 12)
    expect(toFlat(out.colour).every((v, c) => Math.abs(v - (g.colour[15 + c] as number)) < 1e-12)).toBe(true)
    expect(g.maxDensity).toBe(Math.max(...g.density))
    expect(g.density.every((v) => v >= 0)).toBe(true)
  })

  it('uses the fine network when there is one', () => {
    const model = nerfModel({ ...small, fine: true, key: key(4) })
    const params = nerfParams(model)
    const g = nerfDensityGrid(model, params, { resolution: 2 })
    const fine = (model.fine as NerfField).apply(
      params.fine as never,
      [tensor([[-0.5, -0.5, -0.5]]), tensor([[0, 0, -1]])] as never,
      { train: false },
    )
    expect(g.density[0]).toBeCloseTo(toFlat(fine.density)[0] as number, 12)
  })
})

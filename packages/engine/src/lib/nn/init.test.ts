import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { matmul, toArray, toFlat, transpose } from '../tensor/index.ts'
import { conv1d } from './functional/conv.ts'
import * as I from './init.ts'

const round = (xs: number[]): number[] => xs.map((v) => Math.round(v * 1e9) / 1e9 + 0)
const std = (xs: number[]): number => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length)
}

describe('initialisers', () => {
  it('orthogonal rows or columns are orthonormal times the gain', () => {
    const wide = I.orthogonal(key(0), [2, 5], 2)
    expect(round(toFlat(matmul(wide, transpose(wide))))).toEqual([4, 0, 0, 4])
    const tall = I.orthogonal(key(1), [5, 3])
    expect(round(toFlat(matmul(transpose(tall), tall)))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1])
    expect(I.orthogonal(key(2), [4, 2, 3]).shape).toEqual([4, 2, 3])
  })
  it('variance-scaling initialisers have the intended spread', () => {
    const shape = [200, 100]
    expect(std(toFlat(I.lecunNormal(key(0), shape)))).toBeCloseTo(0.1, 2)
    expect(std(toFlat(I.lecunUniform(key(1), shape)))).toBeCloseTo(0.1, 2)
    expect(std(toFlat(I.kaimingNormal(key(2), shape)))).toBeCloseTo(Math.SQRT2 / 10, 2)
    expect(std(toFlat(I.kaimingUniform(key(3), shape, { mode: 'fanOut', nonlinearity: 'tanh' })))).toBeCloseTo(
      5 / 3 / Math.sqrt(200),
      2,
    )
    expect(std(toFlat(I.xavierNormal(key(4), shape)))).toBeCloseTo(Math.sqrt(2 / 300), 2)
  })
  it('truncated normal stays within its bounds', () => {
    const v = toFlat(I.truncatedNormalInit(key(0), [500], { mean: 1, std: 2, a: 0, b: 3 }))
    expect(v.every((x) => x >= 0 && x <= 3)).toBe(true)
  })
  it('Dirac weights make a convolution the identity', () => {
    const w = I.diracInit([2, 2, 3])
    const x = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]
    expect(toArray(conv1d(x, w, { padding: 1 }))).toEqual(x)
    expect(toFlat(I.diracInit([4, 2, 1], 2)).filter((v) => v === 1).length).toBe(4)
  })
  it('sparse, constant, eye and gains', () => {
    const s = toArray(I.sparseInit(key(0), [4, 3], 0.5)) as number[][]
    for (let c = 0; c < 3; c++) expect(s.filter((r) => r[c] === 0).length).toBe(2)
    expect(toFlat(I.constantInit([2], 3))).toEqual([3, 3])
    expect(toFlat(I.onesInit([2]))).toEqual([1, 1])
    expect(toArray(I.eyeInit([3, 2]))).toEqual([
      [1, 0],
      [0, 1],
      [0, 0],
    ])
    expect(I.calculateGain('leakyRelu', 0.2)).toBeCloseTo(Math.sqrt(2 / 1.04), 12)
    expect(I.calculateGain('conv2d')).toBe(1)
    expect(() => I.fans([3])).toThrow()
  })
})

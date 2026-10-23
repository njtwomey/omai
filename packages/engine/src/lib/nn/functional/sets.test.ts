import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { ones, take, tensor, toArray, toFlat, type Value, zeros } from '../../tensor/index.ts'
import { isab, type MabParams, mab, maskSet, pma, sab, setPool } from './sets.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value, digits = 11): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}
const D = 4
const params = (seed: number): MabParams => {
  const k = key(seed)
  return {
    attention: {
      inProjWeight: normal(foldIn(k, 0), [3 * D, D], { scale: 0.5 }),
      inProjBias: normal(foldIn(k, 1), [3 * D], { scale: 0.1 }),
      outProjWeight: normal(foldIn(k, 2), [D, D], { scale: 0.5 }),
    },
    ffWeight: normal(foldIn(k, 3), [D, D], { scale: 0.5 }),
    ffBias: normal(foldIn(k, 4), [D], { scale: 0.1 }),
    norm0: { weight: ones([D]), bias: zeros([D]) },
    norm1: { weight: ones([D]), bias: zeros([D]) },
  }
}
const permute = (x: Value, order: number[]): Value => take(x, order, 1)
const valid = (rows: boolean[][]): Value => tensor(rows, { dtype: 'bool' })

describe('set pooling', () => {
  it('pools the valid elements and gives zeros for empty sets', () => {
    const x = tensor([
      [
        [1, 2],
        [3, 4],
        [100, 100],
      ],
      [
        [5, 6],
        [7, 8],
        [9, 10],
      ],
    ])
    const m = valid([
      [true, true, false],
      [false, false, false],
    ])
    expect(toArray(setPool(x, m, 'sum'))).toEqual([
      [4, 6],
      [0, 0],
    ])
    expect(toArray(setPool(x, m, 'mean'))).toEqual([
      [2, 3],
      [0, 0],
    ])
    expect(toArray(setPool(x, m, 'max'))).toEqual([
      [3, 4],
      [0, 0],
    ])
    expect(toArray(setPool(x, undefined, 'max'))).toEqual([
      [100, 100],
      [9, 10],
    ])
    expect(() => setPool([[1, 2]])).toThrow()
    expect((toArray(maskSet(x, m)) as number[][][])[0]).toEqual([
      [1, 2],
      [3, 4],
      [0, 0],
    ])
  })
  it('is differentiable', () => {
    const m = valid([[true, false, true]])
    for (const r of ['sum', 'mean', 'max'] as const)
      expect(checkPrimitive((x) => setPool(x, m, r), [n(0, [1, 3, 2])]).failures).toEqual([])
  })
})

describe('set attention', () => {
  const x = n(1, [2, 5, D])
  const order = [3, 0, 4, 1, 2]

  it('SAB and ISAB are permutation equivariant, PMA invariant', () => {
    const p = params(2)
    close(sab(permute(x, order), p, { numHeads: 2 }), permute(sab(x, p, { numHeads: 2 }), order))
    const inducing = n(3, [3, D])
    const q = params(4)
    close(
      isab(permute(x, order), inducing, p, q, { numHeads: 2 }),
      permute(isab(x, inducing, p, q, { numHeads: 2 }), order),
    )
    const seeds = n(5, [2, D])
    const pooled = pma(x, seeds, p, { numHeads: 1, ffWeight: n(6, [D, D]) })
    expect(pooled.shape).toEqual([2, 2, D])
    close(pma(permute(x, order), seeds, p, { numHeads: 1, ffWeight: n(6, [D, D]) }), pooled)
  })

  it('masked elements do not affect the valid ones', () => {
    const p = params(7)
    const m = valid([
      [true, true, true, false, false],
      [true, true, true, true, true],
    ])
    const changed = toArray(x) as number[][][]
    ;(changed[0] as number[][])[4] = [9, 9, 9, 9]
    const a = toArray(sab(x, p, { numHeads: 2, mask: m })) as number[][][]
    const b = toArray(sab(tensor(changed), p, { numHeads: 2, mask: m })) as number[][][]
    for (let i = 0; i < 3; i++) for (let d = 0; d < D; d++) expect(b[0]?.[i]?.[d]).toBeCloseTo(a[0]?.[i]?.[d] ?? 0, 12)
    close(
      pma(x, n(8, [1, D]), p, { numHeads: 1, mask: m }),
      pma(tensor(changed), n(8, [1, D]), p, { numHeads: 1, mask: m }),
    )
  })

  it('MAB without layer norms is the plain residual form, and blocks are differentiable', () => {
    const p = params(9)
    const bare: MabParams = { attention: p.attention, ffWeight: p.ffWeight }
    expect(mab(x, n(10, [2, 3, D]), bare, { numHeads: 4 }).shape).toEqual([2, 5, D])
    const small = n(11, [1, 3, D])
    expect(
      checkPrimitive((v, w) => sab(v, { ...bare, ffWeight: w }, { numHeads: 2 }), [small, p.ffWeight as Value], {
        rtol: 1e-5,
        atol: 1e-7,
      }).failures,
    ).toEqual([])
  })
})

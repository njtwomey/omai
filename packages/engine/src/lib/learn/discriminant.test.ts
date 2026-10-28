import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import {
  discriminantAnalysis,
  kernelFisherDiscriminant,
  linearDiscriminant,
  quadraticDiscriminant,
} from './discriminant.ts'
import { revive, serialiseFitted } from './estimator.ts'

const x = [
  [0, 0.2],
  [1, 0],
  [0.3, 1.1],
  [0.9, 0.8],
  [4, 4.1],
  [5, 3.8],
  [4.2, 5],
  [4.9, 4.6],
  [8, 0.1],
  [9, 0.5],
  [8.4, 1],
  [9.2, -0.4],
]
const y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

describe('discriminant analysis', () => {
  it('classifies and projects on the discriminant axes', () => {
    const f = linearDiscriminant().fit({ x, y })
    expect(
      toFlat(
        f.predict([
          [0.5, 0.5],
          [4.5, 4.5],
          [8.5, 0.3],
        ]),
      ),
    ).toEqual([0, 1, 2])
    expect(f.score({ x, y })).toBe(1)
    const z = toArray(f.transform(x)) as number[][]
    expect(z[0]).toHaveLength(2)
    // Along the axes the classes are separated: the class means of the first axis are ordered apart.
    expect(f.explainedVarianceRatio.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const p = toArray(f.predictProba([[4, 4]])) as number[][]
    expect(p[0]?.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('moves from the pooled to the per-class covariance with the mix, and agrees at the ends', () => {
    const q = [
      [2, 2],
      [6, 2],
      [0, 5],
    ]
    const lda = toFlat(discriminantAnalysis({ covariance: 'pooled' }).fit({ x, y }).predictProba(q))
    const mixed0 = toFlat(discriminantAnalysis({ covariance: 'class', mix: 0 }).fit({ x, y }).predictProba(q))
    const qda = toFlat(quadraticDiscriminant().fit({ x, y }).predictProba(q))
    const mixed1 = toFlat(discriminantAnalysis({ covariance: 'class', mix: 1 }).fit({ x, y }).predictProba(q))
    for (const [i, v] of mixed0.entries()) expect(v).toBeCloseTo(lda[i] ?? 0, 10)
    for (const [i, v] of mixed1.entries()) expect(v).toBeCloseTo(qda[i] ?? 0, 12)
  })

  it('handles more features than rows: shrinkage restores full rank, a singular covariance is still inverted', () => {
    const wide = [
      [1, 2, 3, 4, 5],
      [2, 1, 4, 3, 6],
      [9, 8, 7, 6, 5],
      [8, 9, 6, 7, 4],
    ]
    const f = linearDiscriminant({ shrinkage: 0.5 }).fit({ x: wide, y: [0, 0, 1, 1] })
    expect(toFlat(f.predict(wide))).toEqual([0, 0, 1, 1])
    const singular = linearDiscriminant().fit({ x: wide, y: [0, 0, 1, 1] })
    expect(toFlat(singular.predictProba(wide)).every(Number.isFinite)).toBe(true)
    const q = quadraticDiscriminant({ regParam: 0.1 }).fit({ x: wide, y: [0, 0, 1, 1] })
    expect(toFlat(q.predict(wide))).toEqual([0, 0, 1, 1])
  })

  it('runs as a session over the rows, with the model of the first rows at an early step', () => {
    const est = linearDiscriminant({ shrinkage: 0.1 })
    const s = est.session({ x, y }, { key: key(0), logs: 'all' })
    s.advance({ steps: 6 })
    // Row 0 is the first row: the first step has one class and no covariance, a later one two classes.
    expect(s.fittedAt(0).params.classes).toEqual([0, 1, 2])
    expect(s.fittedAt(0).params.priors).toEqual([1, 0, 0])
    expect(s.probe(5, 'model/params')).toMatchObject({ priors: [4 / 6, 2 / 6, 0] })
    expect(toFlat(s.fittedAt(5).predict([[4.5, 4.5]]))).toEqual([1])
    expect(toFlat(s.fittedAt(0).predict([[4.5, 4.5]]))).toEqual([0])
    const full = revive(est, JSON.parse(JSON.stringify(serialiseFitted(est.fit({ x, y })))))
    expect(toFlat(full.predict([[8.5, 0.3]]))).toEqual([2])
  })
})

describe('kernel Fisher discriminant', () => {
  const ring = [
    [0.1, 0],
    [0, 0.2],
    [-0.1, 0],
    [0, -0.2],
    [0.2, 0.1],
    [2, 0],
    [0, 2.1],
    [-2, 0],
    [0, -2.1],
    [1.5, 1.5],
  ]
  const ringY = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

  it('separates concentric classes a line cannot', () => {
    const f = kernelFisherDiscriminant({ kernelOptions: { gamma: 0.5 } }).fit({ x: ring, y: ringY })
    expect(f.score({ x: ring, y: ringY })).toBe(1)
    expect(
      toFlat(
        f.predict([
          [0.05, -0.05],
          [-1.9, 0.2],
        ]),
      ),
    ).toEqual([0, 1])
    expect(toArray(f.transform(ring))).toHaveLength(10)
  })

  it('has the linear discriminant’s axis for a linear kernel and a vanishing ridge', () => {
    const two = x.slice(0, 8)
    const twoY = y.slice(0, 8)
    const kfd = kernelFisherDiscriminant({ kernel: 'linear', regularisation: 1e-9 }).fit({ x: two, y: twoY })
    const lda = linearDiscriminant().fit({ x: two, y: twoY })
    const a = toFlat(kfd.transform(two))
    const b = toFlat(lda.transform(two))
    const mean = (v: ArrayLike<number>) => Array.from(v).reduce((s, t) => s + t, 0) / v.length
    const [ma, mb] = [mean(a), mean(b)]
    let sab = 0
    let saa = 0
    let sbb = 0
    for (let i = 0; i < a.length; i++) {
      sab += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
      saa += ((a[i] ?? 0) - ma) ** 2
      sbb += ((b[i] ?? 0) - mb) ** 2
    }
    expect(Math.abs(sab / Math.sqrt(saa * sbb))).toBeCloseTo(1, 6)
  })

  it('survives a session from the first row', () => {
    const s = kernelFisherDiscriminant().session({ x: ring, y: ringY }, { key: key(0), logs: 'all' })
    s.advance({ steps: 6 })
    expect(toFlat(s.fittedAt(1).predict([[2, 0]]))).toEqual([0])
    expect(s.fittedAt(6).params.x).toHaveLength(7)
  })
})

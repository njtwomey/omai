import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { NMF, nmf } from './nmf.ts'

const x = Array.from({ length: 20 }, (_, i) => [i % 3, (i * 7) % 5, 1 + (i % 2), (i % 4) / 2, 2])

describe('NMF', () => {
  it('fits a non-negative factorisation whose error falls', () => {
    const f = nmf({ nComponents: 3, maxIter: 100 }).fit({ x })
    expect(f.params.components.flat().every((v) => v >= 0)).toBe(true)
    const [rec] = recordsOf(f.exposition, 'components')
    expect(rec?.scores?.length).toBe(20)
    const full = new NMF({ nComponents: 3, betaLoss: 'kullback-leibler', init: 'random', maxIter: 100 }).fit(
      x,
      undefined,
      {
        key: key(0),
      },
    )
    const w = toArray(full.transform(x)) as number[][]
    expect(w.flat().every((v) => v >= 0)).toBe(true)
    expect(full.inverseTransform(w).shape).toEqual([20, 5])
  })
})

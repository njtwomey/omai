import fc from 'fast-check'
import { expect, describe as group, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { type DType, tensor, toFlat } from '../tensor/index.ts'
import {
  batches,
  batchSampler,
  bucketBy,
  collate,
  corpus,
  dataLoader,
  epochBatches,
  epochs,
  padSequences,
  parallel,
  randomSampler,
  rowCount,
  sequenceLabelling,
  sequenceLengths,
  sequences,
  sequentialSampler,
  subsetRandomSampler,
  weightedRandomSampler,
  xy,
} from './index.ts'

/** The error code a call fails with, or `undefined` when it returns. */
function codeOf(f: () => unknown): string | undefined {
  try {
    f()
  } catch (e) {
    return isOmaiError(e) ? e.code : `not an OmaiError: ${String(e)}`
  }
  return undefined
}

const ROWS = (n: number): ReturnType<typeof xy> =>
  xy(
    Array.from({ length: n }, (_, i) => [i]),
    Array.from({ length: n }, (_, i) => i),
  )
const idsOf = (bs: readonly { readonly y: { readonly data: ArrayLike<number> } }[]): number[][] =>
  bs.map((b) => Array.from(b.y.data))

group('sequence lengths', () => {
  it('count the tokens of every row of every sequence-shaped kind', () => {
    const rows: [string, Parameters<typeof sequenceLengths>[0], number[]][] = [
      [
        'sequence: the real tokens of the mask',
        sequences([[1, 2, 3], [4], [5, 6]], [[1], [2], [3]], ['<pad>', 'a']),
        [3, 1, 2],
      ],
      ['corpus tokens', corpus({ tokens: [[1], [], [1, 2, 3]], vocabulary: ['a', 'b', 'c', 'd'] }), [1, 0, 3]],
      [
        'corpus documents split on any run of white space',
        corpus({ documents: ['a  b\tc', '  ', 'one', ' x y '] }),
        [3, 0, 1, 2],
      ],
      ['sequence labelling', sequenceLabelling([['a', 'b'], ['c']], [[0, 1], [0]], ['X', 'Y']), [2, 1]],
      ['parallel: the words of the source', parallel(['the cat  sat', 'hi', ''], ['x', 'y', 'z']), [3, 1, 0]],
    ]
    for (const [name, data, want] of rows) expect(sequenceLengths(data), name).toEqual(want)
    expect(
      codeOf(() => sequenceLengths(ROWS(3))),
      'a regression set has no sequence rows',
    ).toBe('value')
  })
})

group('padding and collating', () => {
  const SEQS = [[1, 2, 3, 4], [5, 6], [], [7]]
  it('lay sequences out in a rectangle whose mask and lengths say what is real, for every pad and truncate side', () => {
    for (const padding of ['pre', 'post'] as const)
      for (const truncating of ['pre', 'post'] as const)
        for (const maxLength of [undefined, 3, 6]) {
          const at = `padding ${padding}, truncating ${truncating}, maxLength ${maxLength}`
          const p = padSequences(SEQS, {
            padding,
            truncating,
            padValue: -1,
            ...(maxLength === undefined ? {} : { maxLength }),
          })
          const t = maxLength ?? 4
          expect(p.values.shape, at).toEqual([4, t])
          expect(p.mask.shape, at).toEqual([4, t])
          SEQS.forEach((s, i) => {
            const keep = Math.min(s.length, t)
            expect(p.lengths.data[i], `${at}: length of row ${i}`).toBe(keep)
            const row = Array.from(p.values.data.slice(i * t, (i + 1) * t))
            const real = Array.from(p.mask.data.slice(i * t, (i + 1) * t))
            // The mask marks the kept tokens, which sit at the end for `pre` padding and the start for `post`.
            expect(real, `${at}: mask of row ${i}`).toEqual(
              Array.from({ length: t }, (_, j) => ((padding === 'pre' ? j >= t - keep : j < keep) ? 1 : 0)),
            )
            const kept = truncating === 'pre' ? s.slice(s.length - keep) : s.slice(0, keep)
            expect(
              row.filter((_, j) => real[j] === 1),
              `${at}: tokens of row ${i}`,
            ).toEqual(kept)
            expect(
              row.filter((_, j) => real[j] === 0).every((v) => v === -1),
              `${at}: pad value`,
            ).toBe(true)
          })
        }
    // Trailing dimensions travel with the time axis, per lane; dtypes follow the request, else the values.
    const steps = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
    ])
    const p = padSequences([steps, tensor([[4, 40]])], { padding: 'pre' })
    expect([p.values.shape, p.values.dtype]).toEqual([[2, 3, 2], 'float64'])
    expect(Array.from(p.values.data)).toEqual([1, 10, 2, 20, 3, 30, 0, 0, 0, 0, 4, 40])
    const dtypes: [string, DType | undefined, DType][] = [
      ['integers', undefined, 'int32'],
      ['a request for float32', 'float32', 'float32'],
      ['a request for float64', 'float64', 'float64'],
    ]
    for (const [name, dtype, want] of dtypes)
      expect(padSequences([[1, 2]], dtype === undefined ? {} : { dtype }).values.dtype, name).toBe(want)
    expect(padSequences([[0.5]]).values.dtype).toBe('float64')
    expect(padSequences([]).values.shape).toEqual([0, 0])
    // maxLength must be a count of steps.
    for (const maxLength of [Number.NaN, -1, 2.5, Number.POSITIVE_INFINITY])
      expect(
        codeOf(() => padSequences(SEQS, { maxLength })),
        `maxLength ${maxLength}`,
      ).toBe('value')
  })
  it('stack fields into batch axes, and refuse items that disagree', () => {
    const c = collate([
      {
        x: [1, 2],
        t: tensor([
          [1, 2],
          [3, 4],
        ]),
        n: 1.5,
      },
      {
        x: [3, 4],
        t: tensor([
          [5, 6],
          [7, 8],
        ]),
        n: -2,
      },
      {
        x: [5, 6],
        t: tensor([
          [9, 10],
          [11, 12],
        ]),
        n: 0,
      },
    ])
    expect(Object.fromEntries(Object.entries(c).map(([k, v]) => [k, v.shape]))).toEqual({
      x: [3, 2],
      t: [3, 2, 2],
      n: [3],
    })
    expect(toFlat(c['x'] as never)).toEqual([1, 2, 3, 4, 5, 6])
    expect(toFlat(c['n'] as never)).toEqual([1.5, -2, 0])
    const rows: [string, () => unknown][] = [
      ['no items', () => collate([])],
      ['an item without a field', () => collate([{ a: 1, b: 2 }, { a: 1 }])],
      ['arrays of different lengths', () => collate([{ a: [1, 2] }, { a: [1] }])],
      ['tensors of different shapes', () => collate([{ a: tensor([1, 2]) }, { a: tensor([1, 2, 3]) }])],
    ]
    for (const [name, f] of rows) expect(['value', 'shape'], name).toContain(codeOf(f))
  })
})

group('batches, loaders and bucketing', () => {
  it('visit every row once per epoch for any size, key or sampler, and agree with the loader on the count', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 30 }),
        fc.integer({ min: 1, max: 12 }),
        fc.boolean(),
        fc.nat(5),
        (n, size, dropLast, seed) => {
          const data = ROWS(n)
          const samplers = {
            sequential: { sampler: sequentialSampler(n) },
            key: { key: key(seed) },
            subset: {
              sampler: subsetRandomSampler(
                Array.from({ length: n }, (_, i) => i),
                key(seed),
              ),
            },
            none: {},
          }
          for (const [name, extra] of Object.entries(samplers)) {
            const options = { batchSize: size, dropLast, ...extra }
            const bs = batches(data, options)
            const ids = idsOf(bs).flat()
            const kept = dropLast ? n - (n % size) : n
            expect(bs.length, name).toBe(dropLast ? Math.floor(n / size) : Math.ceil(n / size))
            expect(ids.length, name).toBe(kept)
            expect(new Set(ids).size, `${name}: no row twice`).toBe(kept)
            expect(
              bs.every((b, i) => rowCount(b) === size || (!dropLast && i === bs.length - 1)),
              name,
            ).toBe(true)
            if (dropLast === false)
              expect(
                [...ids].sort((a, b) => a - b),
                name,
              ).toEqual(Array.from({ length: n }, (_, i) => i))
            expect(dataLoader(data, options).length, `${name}: the loader's length`).toBe(bs.length)
            expect(dataLoader(data, options).epoch(0).length, `${name}: the loader's epoch`).toBe(bs.length)
          }
          // A sampler overrides the key, and a sampler is asked for the epoch.
          const bySampler = epochBatches(data, 3, {
            batchSize: size,
            key: key(seed),
            sampler: randomSampler(n, key(seed + 1)),
          })
          expect(idsOf(bySampler)).toEqual(
            idsOf(batches(data, { batchSize: size, sampler: randomSampler(n, key(seed + 1)), epoch: 3 })),
          )
          expect(epochs(data, 0, { batchSize: size })).toEqual([])
        },
      ),
      { numRuns: 40 },
    )
  })
  it('refuse sizes and counts that are not whole numbers, and unordered buckets', () => {
    const data = ROWS(6)
    const lengths = [1, 2, 3, 4, 5, 6]
    const rows: [string, () => unknown][] = [
      ['batch size 0', () => batchSampler(sequentialSampler(4), 0)],
      ['batch size NaN', () => batchSampler(sequentialSampler(4), Number.NaN)],
      ['batch size -1', () => batchSampler(sequentialSampler(4), -1)],
      ['batch size 2.5', () => batchSampler(sequentialSampler(4), 2.5)],
      ['batches with batch size 2.5', () => batches(data, { batchSize: 2.5 })],
      ['randomSampler numSamples NaN', () => randomSampler(5, key(0), { numSamples: Number.NaN, replacement: true })],
      ['randomSampler numSamples 2.5', () => randomSampler(5, key(0), { numSamples: 2.5, replacement: true })],
      ['randomSampler numSamples -1', () => randomSampler(5, key(0), { numSamples: -1, replacement: true })],
      ['weightedRandomSampler with a NaN weight', () => weightedRandomSampler([1, Number.NaN], 2, key(0))(0)],
      ['weightedRandomSampler with a negative weight', () => weightedRandomSampler([1, -1], 2, key(0))(0)],
      ['bucketBy unsorted boundaries', () => bucketBy(data, { boundaries: [4, 2], batchSize: 2, lengths })],
      ['bucketBy NaN boundary', () => bucketBy(data, { boundaries: [Number.NaN], batchSize: 2, lengths })],
      ['bucketBy batch size 0', () => bucketBy(data, { boundaries: [3], batchSize: 0, lengths })],
      ['bucketBy batch size 1.5', () => bucketBy(data, { boundaries: [3], batchSize: [1.5, 2], lengths })],
      ['bucketBy lengths for other rows', () => bucketBy(data, { boundaries: [3], batchSize: 2, lengths: [1, 2] })],
    ]
    const accepted = rows.filter(([, f]) => codeOf(f) !== 'value').map(([name]) => name)
    expect(accepted).toEqual([])
  })
  it('bucket rows by length: each batch within one bucket, rows once, partial batches dropped on request', () => {
    const lengths = [1, 9, 2, 8, 3, 7, 10, 1, 5]
    const data = ROWS(lengths.length)
    const bounds = [4, 8]
    const bucketOf = (len: number): number => bounds.filter((b) => len >= b).length
    for (const dropLast of [false, true]) {
      const bs = idsOf(bucketBy(data, { boundaries: bounds, batchSize: [2, 3, 2], lengths, dropLast, key: key(2) }))
      for (const b of bs) expect(new Set(b.map((i) => bucketOf(lengths[i] ?? 0))).size).toBe(1)
      const rows = bs.flat()
      expect(new Set(rows).size).toBe(rows.length)
      // Buckets hold 4, 2 and 3 rows: sizes 2, 3, 2 leave a partial batch in the middle bucket and the last.
      expect(rows.length).toBe(dropLast ? 6 : 9)
    }
    expect(idsOf(bucketBy(data, { boundaries: [], batchSize: 4, lengths }))).toEqual([[0, 1, 2, 3], [4, 5, 6, 7], [8]])
  })
})

import { expect, describe as group, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  concatRows,
  corpus,
  type Dataset,
  labelsOf,
  rowCount,
  stratifiedSplit,
  table,
  take,
  trainTestSplit,
  trainValTestSplit,
  xy,
  xyClasses,
} from './index.ts'

// Row identities: each dataset carries the row number i in a field of its own, so a part's rows can be read back.
const N = 60
const label = (i: number): number => (i < 40 ? 0 : i < 58 ? 1 : i === 58 ? 2 : 3) // 40, 18, 1 and 1 rows
const ROWS = Array.from({ length: N }, (_, i) => i)
const CLASSIFICATION = xyClasses(
  ROWS.map((i) => [i]),
  ROWS.map(label),
)
const CORPUS = corpus({
  tokens: ROWS.map((i) => [i]),
  vocabulary: ROWS.map(String),
  labels: ROWS.map(label),
})
const TABLE = table(
  {
    id: ROWS,
    kind: ROWS.map((i) => (i % 2 === 0 ? 'even' : 'odd')),
    flag: ROWS.map((i) => i % 3 === 0),
    note: ROWS.map((i) => `row ${i}`),
  },
  { text: ['note'] },
)

/** The row numbers of a dataset, read from its identity field. */
function idsOf(d: Dataset): number[] {
  if (d.kind === 'classification') return toFlat(d.x)
  if (d.kind === 'corpus') return (d.tokens ?? []).map((t) => t[0] ?? Number.NaN)
  if (d.kind === 'table') {
    const c = d.columns['id']
    return c?.kind === 'numeric' ? toFlat(c.values) : []
  }
  throw new Error(`no identity for ${d.kind}`)
}

type Parts = Readonly<Record<string, Dataset>>
const sorted = (xs: readonly number[]): number[] => [...xs].sort((a, b) => a - b)

/** The splitters under test: name, run (a different key gives a different split), whether the parts are stratified. */
const SPLITTERS: readonly [string, (d: Dataset, k: number) => Parts, boolean][] = [
  ['trainTestSplit, in order', (d) => trainTestSplit(d, { testSize: 0.25 }), false],
  ['trainTestSplit, shuffled', (d, k) => trainTestSplit(d, { testSize: 0.2, key: key(k) }), false],
  ['trainTestSplit, stratified', (d, k) => trainTestSplit(d, { testSize: 0.3, key: key(k), stratify: true }), true],
  ['stratifiedSplit', (d, k) => stratifiedSplit(d, { testSize: 17, key: key(k) }), true],
  ['trainValTestSplit, shuffled', (d, k) => trainValTestSplit(d, { valSize: 0.15, testSize: 12, key: key(k) }), false],
  [
    'trainValTestSplit, stratified',
    (d, k) => trainValTestSplit(d, { valSize: 10, testSize: 0.2, key: key(k), stratify: true }),
    true,
  ],
]

/** The error code a call fails with, or `undefined` when it returns. */
function codeOf(f: () => unknown): string | undefined {
  try {
    f()
  } catch (e) {
    return isOmaiError(e) ? e.code : `not an OmaiError: ${String(e)}`
  }
  return undefined
}

group('the split family', () => {
  it('splits every row into exactly one part, reproducibly, with class shares kept within a row when stratified', () => {
    for (const data of [CLASSIFICATION, CORPUS, TABLE])
      for (const [name, split, stratified] of SPLITTERS) {
        if (stratified && labelsOf(data) === undefined) continue
        const at = `${data.kind}, ${name}`
        const parts = split(data, 5)
        const ids = Object.values(parts).map(idsOf)
        // Disjoint, and together all the rows, each once.
        expect(sorted(ids.flat()), `${at}: every row once`).toEqual(ROWS)
        expect(
          ids.map((p) => p.length),
          `${at}: the sizes`,
        ).toEqual(Object.values(parts).map(rowCount))
        // The same key is the same split; another key moves rows (an in-order split has no key).
        expect(Object.values(split(data, 5)).map(idsOf), `${at}: same key`).toEqual(ids)
        if (name !== 'trainTestSplit, in order')
          expect(Object.values(split(data, 6)).map(idsOf), `${at}: other key`).not.toEqual(ids)
        // A part carries the dataset's own fields along with its rows (a text column follows its ids).
        if (data.kind === 'table') {
          for (const part of Object.values(parts)) {
            const note = (part as typeof TABLE).columns['note']
            expect(note?.kind === 'text' ? note.values : [], `${at}: text column`).toEqual(
              idsOf(part).map((i) => `row ${i}`),
            )
          }
        }
        if (!stratified) continue
        const labels = labelsOf(data) ?? []
        for (const part of Object.values(parts)) {
          const mine = labelsOf(part) ?? []
          for (const c of [0, 1, 2, 3]) {
            const share = (labels.filter((l) => l === c).length * mine.length) / N
            expect(
              Math.abs(mine.filter((l) => l === c).length - share),
              `${at}: class ${c} of a part of ${mine.length}`,
            ).toBeLessThan(2)
          }
        }
      }
  })
})

group('degenerate arguments of the row operations', () => {
  it('refuses what leaves a part empty or a split without what it needs, and returns the empty and single cases', () => {
    const tiny = xyClasses([[0], [1]], [0, 1])
    const rows: [string, () => unknown][] = [
      ['stratify without a key', () => trainTestSplit(CLASSIFICATION, { stratify: true })],
      [
        'stratify without class labels',
        () => trainTestSplit(xy([[0], [1], [2]], [0, 1, 2]), { stratify: true, key: key(0) }),
      ],
      ['stratifiedSplit without class labels', () => stratifiedSplit(xy([[0], [1]], [0, 1]), { key: key(0) })],
      ['test size 0', () => trainTestSplit(CLASSIFICATION, { testSize: 0 })],
      ['a fraction that rounds up to every row', () => trainTestSplit(tiny, { testSize: 0.999 })],
      ['one row alone', () => trainTestSplit(take(tiny, [0]))],
      [
        'three-way sizes that leave no training rows',
        () => trainValTestSplit(CLASSIFICATION, { valSize: 30, testSize: 30 }),
      ],
      ['three-way validation size 0', () => trainValTestSplit(CLASSIFICATION, { valSize: 0 })],
      ['three-way of a tiny dataset', () => trainValTestSplit(tiny)],
      ['no datasets to stack', () => concatRows([])],
      ['stacking two kinds', () => concatRows([CLASSIFICATION, xy([[0]], [0])] as Dataset[])],
      [
        'stacking categorical columns with other levels',
        () => concatRows([table({ a: ['x', 'y'] }), table({ a: ['y', 'z'] })]),
      ],
      [
        'stacking tables with different columns',
        () => concatRows([table({ a: [1], b: [2] }), table({ a: [1], c: [2] })]),
      ],
    ]
    for (const [name, f] of rows) expect(codeOf(f), name).toBe('value')

    // A dataset with one class splits, stratified, as it does unstratified: all rows in one class.
    const one = xyClasses(
      ROWS.map((i) => [i]),
      new Array(N).fill(0),
    )
    const s = stratifiedSplit(one, { testSize: 0.25, key: key(1) })
    expect([rowCount(s.train), rowCount(s.test)]).toEqual([45, 15])
    // No rows, and a row taken twice: every per-row field of every kind follows (a text column included).
    for (const d of [CLASSIFICATION, CORPUS, TABLE]) {
      expect(rowCount(take(d, [])), d.kind).toBe(0)
      expect(idsOf(take(d, [5, 5, 0])), d.kind).toEqual([5, 5, 0])
    }
    // Taking then stacking is the identity on rows, whatever the column kinds.
    const stacked = concatRows([take(TABLE, ROWS.slice(0, 25)), take(TABLE, ROWS.slice(25))])
    expect(stacked).toEqual(TABLE)
  })
})

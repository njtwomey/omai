import { expect, describe as group, it } from 'vitest'
import { isMdmlError, quantileOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { add, mul, reshape, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  asGenerated,
  balancedWeights,
  batches,
  batchSampler,
  bayesErrorEstimate,
  bayesPredict,
  bootstrap,
  bucketBy,
  ClassificationDataSchema,
  clusterAgreement,
  collate,
  comparisons,
  completeTruth,
  concatRows,
  corpus,
  type DatasetGenerator,
  dataLoader,
  describe,
  describeGenerator,
  describeRecipe,
  encodeRecipe,
  epochBatches,
  epochs,
  events,
  excessRisk,
  fromTensors,
  generate,
  graphData,
  graphRecord,
  images,
  indexedDbCache,
  interactions,
  labelsOf,
  load,
  makeFromRecipe,
  memoryCache,
  padSequences,
  pairwise,
  parallel,
  parseRecipe,
  perRowFields,
  posteriorFromJoint,
  queryData,
  RecipeSchema,
  type RegressionData,
  RegressionDataSchema,
  randomSampler,
  rowCount,
  sequenceLabelling,
  sequenceLengths,
  sequences,
  sequentialSampler,
  sha256Hex,
  shuffleRows,
  stratifiedIndices,
  stratifiedSplit,
  subsetRandomSampler,
  TableDataSchema,
  table,
  tableToXy,
  take,
  timeSeries,
  toTensors,
  trainTestSplit,
  trainValTestSplit,
  trajectories,
  triples,
  weightedRandomSampler,
  xy,
  xyClasses,
  xyMultilabel,
  xyMultiOutput,
} from './index.ts'

const line: DatasetGenerator<RegressionData> = {
  name: 'line',
  kind: 'regression',
  description: 'y = 2x + 1',
  knobs: dict({ n: int(1, 1000, { default: 5 }), slope: real(-10, 10, { default: 2 }) }),
  generate: (_k, knobs) => {
    const n = knobs['n'] as number
    const x = Array.from({ length: n }, (_, i) => [i])
    return {
      data: xy(
        x,
        x.map(([v]) => (knobs['slope'] as number) * (v ?? 0) + 1),
      ),
      truth: { regressionFunction: (v) => add(mul(v, knobs['slope'] as number), 1), noiseScale: 0 },
    }
  },
}

group('schemas', () => {
  it('build every kind and count rows through its per-row fields', () => {
    const datasets = [
      xy([[1], [2]], [1, 2]),
      xyClasses([[1], [2]], [0, 1]),
      xyMultilabel(
        [[1], [2]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
      xyMultiOutput(
        [[1], [2]],
        [
          [1, 0],
          [0, 2],
        ],
      ),
      table({ a: [1, 2], b: ['x', 'y'] }),
      interactions({ users: [0, 1], items: [1, 1], values: [5, 4], times: [0, 1] }),
      queryData([[1], [2]], [1, 0], [0, 0]),
      corpus({ documents: ['a b', 'c'] }),
      sequenceLabelling([['a'], ['b', 'c']], [[0], [1, 0]], ['X', 'Y']),
      parallel(['a', 'b'], ['x', 'y']),
      sequences([[1], [2, 3]], [[1], [3, 2]], ['<pad>', 'a', 'b', 'c']),
      timeSeries([1, 2]),
      events({ times: [0.1, 0.4], marks: [0, 1] }),
      pairwise([0, 1], [1, 2]),
      graphData([graphRecord([[0, 1]]), graphRecord([[0, 1]])], { graphLabels: [0, 1] }),
      triples([0, 1], [0, 0], [1, 2]),
      images([[[0]], [[1]]], { labels: [0, 1] }),
      trajectories({ observations: [[0], [1]], actions: [0, 1], rewards: [1, 0], nActions: 2 }),
    ]
    for (const d of datasets) {
      expect(rowCount(d), d.kind).toBe(2)
      expect(d.meta.task).toBe(d.kind)
      const two = take(d, [1, 0, 1])
      expect(rowCount(two), d.kind).toBe(3)
      expect(rowCount(concatRows([d, d])), d.kind).toBe(4)
    }
  })

  it('validate with zod schemas', () => {
    expect(RegressionDataSchema.safeParse(xy([[1]], [1])).success).toBe(true)
    expect(ClassificationDataSchema.safeParse(xy([[1]], [1])).success).toBe(false)
    expect(TableDataSchema.safeParse(table({ a: [true, false] })).success).toBe(true)
  })

  it('reject inconsistent fields', () => {
    expect(() => xy([[1], [2]], [1])).toThrow(/targets/)
    expect(() => interactions({ users: [0], items: [0, 1], values: [1] })).toThrow(/items/)
    expect(() => graphRecord([[0, 3]], { nNodes: 2 })).toThrow(/outside/)
    expect(() => sequenceLabelling([['a', 'b']], [[0]], ['X'])).toThrow(/tags/)
    expect(() => corpus({ tokens: [[5]], vocabulary: ['a'] })).toThrow(/vocabulary/)
    expect(() => events({ times: [5], window: [0, 1] })).toThrow(/window/)
    expect(() => timeSeries([1, 2], { times: [1, 0] })).toThrow(/decrease/)
    expect(() => comparisons([[0, 1]], [[0]])).toThrow(/shape|\[n, k\]/)
  })

  it('fill defaults', () => {
    expect(perRowFields(interactions({ users: [0], items: [0], values: [1] }))).toEqual(['users', 'items', 'values'])
    expect(timeSeries([[1, 2]]).channelNames).toEqual(['c0', 'c1'])
    expect(events({ locations: [[0, 0]] }).nSequences).toBe(1)
    expect(pairwise([0], [1], { ties: [true] }).ranks.data).toEqual(new Int32Array([0, 0]))
  })
})

group('tables', () => {
  const t = table(
    { age: [30, null, 50], city: ['Cork', 'Bray', null], ok: [true, false, true], label: ['a', 'b', 'a'] },
    { target: 'label' },
  )
  it('encode missing values as NaN and -1', () => {
    const city = t.columns['city']
    expect(city?.kind === 'categorical' ? Array.from(city.codes.data) : []).toEqual([1, 0, -1])
    const d = tableToXy(t)
    expect(d.kind).toBe('classification')
    expect(d.featureNames).toEqual(['age', 'city=Bray', 'city=Cork', 'ok'])
    expect(toFlat(d.x).map((v) => (Number.isNaN(v) ? 'nan' : v))).toEqual([
      30,
      0,
      1,
      1,
      'nan',
      1,
      0,
      0,
      50,
      'nan',
      'nan',
      1,
    ])
    expect(Object.keys(toTensors(t))).toEqual(['x', 'y'])
  })
  it('describe numeric columns and class counts', () => {
    const s = describe(t)
    expect(s.features[0]?.missing).toBe(1)
    expect(s.classCounts).toEqual({ a: 2, b: 1 })
  })
})

group('rows and splits', () => {
  const d = xyClasses(
    Array.from({ length: 40 }, (_, i) => [i]),
    Array.from({ length: 40 }, (_, i) => (i < 30 ? 0 : 1)),
  )
  it('split three ways with every row once', () => {
    const p = trainValTestSplit(d, { key: key(1) })
    expect([p.train, p.val, p.test].map((x) => rowCount(x))).toEqual([28, 6, 6])
    const all = [p.train, p.val, p.test].flatMap((x) => toFlat(x.x)).sort((a, b) => a - b)
    expect(all).toEqual(Array.from({ length: 40 }, (_, i) => i))
  })
  it('stratify by class', () => {
    const s = trainTestSplit(d, { testSize: 0.2, key: key(3), stratify: true })
    expect(Array.from(s.test.y.data).filter((c) => c === 1).length).toBe(2)
    const t = stratifiedSplit(d, { testSize: 10, key: key(4) })
    expect([2, 3]).toContain(Array.from(t.test.y.data).filter((c) => c === 1).length)
    expect(labelsOf(t.test)?.length).toBe(10)
  })
  it('bootstrap with an out-of-bag part near e^-1', () => {
    const b = bootstrap(d, key(7))
    const inBag = new Set(b.indices)
    expect(rowCount(b.outOfBag)).toBe(40 - inBag.size)
    const big = xy(
      Array.from({ length: 2000 }, (_, i) => [i]),
      new Array(2000).fill(0),
    )
    expect(rowCount(bootstrap(big, key(8)).outOfBag) / 2000).toBeCloseTo(Math.exp(-1), 1)
  })
  it('shuffle as a permutation and reject bad indices', () => {
    expect(toFlat(shuffleRows(d, key(0)).x).sort((a, b) => a - b)).toEqual(toFlat(d.x))
    expect(() => take(d, [40])).toThrow(/outside/)
  })
  it('round-trip tensors', () => {
    const r = fromTensors(d, { x: mul(d.x, 2) })
    expect(toFlat(r.x)[1]).toBe(2)
    expect(() => fromTensors(d, { classes: tensor([1]) })).toThrow(/per-row/)
  })
})

/** The error code a call fails with, or `undefined` when it returns. */
function codeOf(f: () => unknown): string | undefined {
  try {
    f()
  } catch (e) {
    return isMdmlError(e) ? e.code : `not an MdmlError: ${String(e)}`
  }
  return undefined
}

group('invalid arguments at the data boundary', () => {
  const d = xyClasses(
    Array.from({ length: 8 }, (_, i) => [i]),
    [0, 0, 0, 0, 0, 0, 1, 1],
  )
  const k = key(0)
  it('refuse sizes and indices that are not whole numbers in range, in every row operation', () => {
    const rows: [string, () => unknown][] = [
      ['testSize NaN', () => trainTestSplit(d, { testSize: Number.NaN })],
      ['testSize Infinity', () => trainTestSplit(d, { testSize: Number.POSITIVE_INFINITY })],
      ['testSize negative', () => trainTestSplit(d, { testSize: -0.2 })],
      ['testSize 2.5 rows', () => trainTestSplit(d, { testSize: 2.5 })],
      ['testSize larger than the rows', () => trainTestSplit(d, { testSize: 9 })],
      ['valSize NaN', () => trainValTestSplit(d, { valSize: Number.NaN })],
      ['stratified testSize NaN', () => stratifiedSplit(d, { testSize: Number.NaN, key: k })],
      ['stratifiedIndices more test rows than rows', () => stratifiedIndices([0, 1], 5, k)],
      ['stratifiedIndices negative test rows', () => stratifiedIndices([0, 1], -1, k)],
      ['stratifiedIndices fractional test rows', () => stratifiedIndices([0, 1], 0.5, k)],
      ['stratifiedIndices NaN label', () => stratifiedIndices([0, Number.NaN], 1, k)],
      ['take NaN', () => take(d, [Number.NaN])],
      ['take a fraction', () => take(d, [1.7])],
      ['take Infinity', () => take(d, [Number.POSITIVE_INFINITY])],
      ['take a float tensor with a fraction', () => take(d, tensor([0.5]))],
      ['take negative', () => take(d, [-1])],
      ['take past the end', () => take(d, [8])],
      ['bootstrap negative size', () => bootstrap(d, k, -1)],
      ['bootstrap fractional size', () => bootstrap(d, k, 2.5)],
      ['bootstrap size 0', () => bootstrap(d, k, 0)],
      ['bootstrap of no rows', () => bootstrap(take(d, []), k)],
    ]
    for (const [name, f] of rows) expect(codeOf(f), name).toBe('value')
    // Valid arguments still work: a fraction, a count, an exact count of every row but one.
    expect(rowCount(trainTestSplit(d, { testSize: 0.25 }).test)).toBe(2)
    expect(rowCount(trainTestSplit(d, { testSize: 3 }).test)).toBe(3)
    expect(rowCount(take(d, tensor([0, 7])))).toBe(2)
  })
  it('refuse labels, columns and stacked parts that would silently mean something else', () => {
    const x = [[0], [1]]
    const rows: [string, () => unknown][] = [
      ['NaN label', () => xyClasses(x, [Number.NaN, 0])],
      ['fractional label', () => xyClasses(x, [1.5, 0])],
      ['label above int32', () => xyClasses(x, [3e9, 0])],
      [
        'NaN multilabel indicator',
        () =>
          xyMultilabel(x, [
            [1, Number.NaN],
            [0, 1],
          ]),
      ],
      [
        'NaN multioutput class',
        () =>
          xyMultiOutput(x, [
            [0, Number.NaN],
            [1, 0],
          ]),
      ],
      ['numbers then a string', () => table({ a: [1, 'x', 3] as unknown as number[] })],
      ['a string then numbers', () => table({ a: ['x', 2, 3] as unknown as string[] })],
      ['a boolean then a number', () => table({ a: [true, 2] as unknown as boolean[] })],
      [
        'classes that differ',
        () =>
          concatRows([xyClasses(x, [0, 1], { classes: ['a', 'b'] }), xyClasses(x, [0, 1], { classes: ['x', 'y'] })]),
      ],
      [
        'feature names that differ',
        () => concatRows([xy(x, [0, 1], { featureNames: ['a'] }), xy(x, [0, 1], { featureNames: ['b'] })]),
      ],
      [
        'sensitive attributes that differ',
        () =>
          concatRows([
            xy(x, [0, 1], { sensitive: { s: ['a', 'b'] } }),
            xy(x, [0, 1], { sensitive: { s: ['b', 'a'] } }),
          ]),
      ],
    ]
    for (const [name, f] of rows) expect(codeOf(f), name).toBe('value')
    // Well-typed input is unchanged: whole floats are labels, nulls are missing, negative labels mean unlabelled.
    expect(Array.from(xyClasses(x, tensor([1, 0])).y.data)).toEqual([1, 0])
    expect(Array.from(xyClasses(x, [-1, 0]).y.data)).toEqual([-1, 0])
    expect(table({ a: [1, null, 3], b: ['u', null, 'v'] }).columns['a']?.kind).toBe('numeric')
    expect(
      concatRows([xyClasses(x, [0, 1], { classes: ['a', 'b'] }), xyClasses(x, [1, 0], { classes: ['a', 'b'] })])
        .classes,
    ).toEqual(['a', 'b'])
  })
  it('choose an integer dtype from the range of the values, and a bool one from every item', () => {
    // 3e9 and epoch milliseconds do not fit 32 bits: they stay exact in float64 instead of wrapping.
    const big = [3e9, 1]
    const epoch = [1.7e12, 5]
    expect(Array.from(padSequences([big]).values.data)).toEqual(big)
    expect(padSequences([big]).values.dtype).toBe('float64')
    expect(padSequences([[2 ** 31 - 1, -(2 ** 31 - 1)]]).values.dtype).toBe('int32')
    expect(padSequences([[1, 2]], { dtype: 'int32' }).values.dtype).toBe('int32')
    const stacked = collate([{ a: epoch[0] ?? 0 }, { a: epoch[1] ?? 0 }])['a']
    expect([stacked?.dtype, Array.from(stacked?.data ?? [])]).toEqual(['float64', epoch])
    expect(collate([{ a: 1 }, { a: 2 }])['a']?.dtype).toBe('int32')
    expect(collate([{ a: false }, { a: true }])['a']?.dtype).toBe('bool')
    expect(codeOf(() => collate([{ a: true }, { a: 2 }]))).toBe('value')
    expect(codeOf(() => collate([{ a: 2 }, { a: true }]))).toBe('value')
  })
})

group('batching and samplers', () => {
  const d = xy(
    Array.from({ length: 10 }, (_, i) => [i]),
    Array.from({ length: 10 }, (_, i) => i),
  )
  it('cover every row once per epoch, reshuffled across epochs', () => {
    const e = epochs(d, 2, { batchSize: 4, key: key(2) })
    expect(e.map((b) => b.map((x) => rowCount(x)))).toEqual([
      [4, 4, 2],
      [4, 4, 2],
    ])
    const order = (bs: RegressionData[]) => bs.flatMap((b) => toFlat(b.y))
    expect([...order(e[0] ?? [])].sort((a, b) => a - b)).toEqual(Array.from({ length: 10 }, (_, i) => i))
    expect(order(e[0] ?? [])).not.toEqual(order(e[1] ?? []))
    expect(order(epochBatches(d, 1, { batchSize: 4, key: key(2) }))).toEqual(order(e[1] ?? []))
  })
  it('loaders follow their sampler', () => {
    const loader = dataLoader(d, { batchSize: 3, sampler: subsetRandomSampler([0, 1, 2, 3], key(0)) })
    expect(loader.length).toBe(2)
    expect(
      loader
        .epoch(0)
        .flatMap((b) => toFlat(b.y))
        .sort(),
    ).toEqual([0, 1, 2, 3])
    expect(batches(d, { batchSize: 4, dropLast: true }).length).toBe(2)
  })
  it('samplers', () => {
    expect(Array.from(sequentialSampler(3)(5))).toEqual([0, 1, 2])
    const r = randomSampler(5, key(1), { replacement: true, numSamples: 20 })(0)
    expect(r.length).toBe(20)
    expect(() => randomSampler(2, key(0), { numSamples: 3 })).toThrow(/replacement/)
    // A message that names the sampler, not the draw underneath it.
    expect(() => randomSampler(0, key(0), { replacement: true, numSamples: 2 })).toThrow(/randomSampler/)
    expect(() => weightedRandomSampler([0, 0], 2, key(0))).toThrow(/weightedRandomSampler: the weights have no positive/)
    const w = weightedRandomSampler([1, 0, 3], 4000, key(3))(0)
    const share = Array.from(w).filter((v) => v === 2).length / 4000
    expect(Math.abs(share - 0.75)).toBeLessThan(4 * Math.sqrt((0.75 * 0.25) / 4000))
    const noRep = weightedRandomSampler([1, 1, 0, 5], 3, key(0), { replacement: false })(0)
    expect(new Set(noRep).size).toBe(3)
    expect(Array.from(noRep)).not.toContain(2)
    expect(batchSampler(sequentialSampler(5), 2, true)(0).length).toBe(2)
    expect(toFlat(balancedWeights([0, 1, 1, 1])).reduce((a, b) => a + b, 0)).toBeCloseTo(4, 12)
    expect(toFlat(balancedWeights(tensor([0, 1, 1, 1], { dtype: 'int32' })))).toEqual(
      toFlat(balancedWeights([0, 1, 1, 1])),
    )
    const weights = balancedWeights([0, 0, 0, 1])
    expect(weightedRandomSampler(weights, 5, key(0))(0).length).toBe(5)
    expect(() => balancedWeights([[0, 1]])).toThrow(expect.objectContaining({ code: 'shape' }))
  })
  it('collate and pad', () => {
    const c = collate([
      { x: tensor([1, 2]), y: 0, f: 0.5, m: true },
      { x: tensor([3, 4]), y: 1, f: 1.5, m: false },
    ])
    expect(c['x']?.shape).toEqual([2, 2])
    expect(c['y']?.dtype).toBe('int32')
    expect(c['f']?.dtype).toBe('float64')
    expect(c['m']?.dtype).toBe('bool')
    const p = padSequences([[1, 2, 3, 4], [5]], { maxLength: 3, truncating: 'pre' })
    expect(Array.from(p.values.data)).toEqual([2, 3, 4, 5, 0, 0])
    expect(Array.from(p.mask.data)).toEqual([1, 1, 1, 1, 0, 0])
    expect(p.values.dtype).toBe('int32')
  })
  it('bucket rows by length', () => {
    const c = corpus({ tokens: [[1], [1, 2, 3], [2], [3, 3, 3], [1, 1]], vocabulary: ['a', 'b', 'c', 'd'] })
    expect(sequenceLengths(c)).toEqual([1, 3, 1, 3, 2])
    const b = bucketBy(c, { boundaries: [2, 3], batchSize: [2, 1, 2], key: key(0) })
    for (const batch of b) {
      const lens = (batch.tokens ?? []).map((t) => t.length)
      expect(new Set(lens).size).toBe(1)
    }
    expect(b.reduce((s, x) => s + rowCount(x), 0)).toBe(5)
    expect(() => bucketBy(c, { boundaries: [2], batchSize: [1] })).toThrow(/batch sizes/)
  })
})

group('generators and truth', () => {
  it('generate with defaults, validate knobs and stamp the recipe and key', () => {
    const g = generate(line, { key: key(9) })
    expect(rowCount(g.data)).toBe(5)
    expect(g.data.meta.recipe?.generator).toBe('line')
    expect(g.data.meta.key).toEqual([9, 0])
    expect(() => generate(line, { knobs: { n: 0 } })).toThrow(/outside/)
    expect(() => generate(line, { knobs: { m: 3 } })).toThrow(/unknown/)
    expect(describeGenerator(line).knobs[1]).toMatchObject({ name: 'slope', low: -10, high: 10, default: 2 })
  })
  it('derive posteriors and Bayes errors', () => {
    // Two unit Gaussians at -1 and +1 with equal priors: Bayes error Phi(-1).
    const logJoint = (x: Tensor | import('../tensor/index.ts').Value, y: import('../tensor/index.ts').Value) => {
      const m = add(mul(y, 2), -1)
      const d = add(reshape(x, [-1]), mul(m, -1))
      return mul(mul(d, d), -0.5)
    }
    const post = posteriorFromJoint(logJoint, 2)
    const xs = tensor(Array.from({ length: 4001 }, (_, i) => [-4 + (8 * i) / 4000]))
    const p = toFlat(post(tensor([[0], [1]])))
    expect(p[0]).toBeCloseTo(0.5, 12)
    expect(p[3]).toBeCloseTo(1 / (1 + Math.exp(-2)), 12)
    const t = completeTruth({ logProb: logJoint as never }, { nClasses: 2, x: xs })
    expect(t.posterior).toBeDefined()
    expect(toFlat(bayesPredict(t, tensor([[-2], [2]])))).toEqual([0, 1])
    expect(bayesErrorEstimate(post, tensor([[0]])).value).toBe(0.5)
    expect(clusterAgreement({ clusters: tensor([0, 1, 1], { dtype: 'int32' }) }, [5, 7, 7])).toBe(1)
    // The best relabelling, not the greedy one: 0 -> 1 and 1 -> 0 match four of seven rows (greedy gave 3 / 7).
    const planted = { clusters: tensor([0, 0, 0, 0, 0, 1, 1], { dtype: 'int32' }) }
    expect(clusterAgreement(planted, [0, 0, 0, 1, 1, 0, 0])).toBe(4 / 7)
    expect(clusterAgreement({ clusters: tensor([], { dtype: 'int32' }) }, [])).toBe(1)
    expect(() => clusterAgreement(planted, [0, 1])).toThrow(expect.objectContaining({ code: 'value' }))
    expect(excessRisk({ regressionFunction: (x) => x }, (x) => x, tensor([1]))).toBe(0)
  })

  it('derive the density, posterior and regression function from a generating program', () => {
    const logJoint = (x: Value, y?: Value): Value =>
      y === undefined ? mul(reshape(x, [-1]), 0) : mul(reshape(x, [-1]), y)
    const program = {
      kind: 'program' as const,
      model: () => undefined,
      features: 'x',
      target: 'y',
      classes: 2,
      logProb: logJoint,
      regressionFunction: (x: Value) => mul(reshape(x, [-1]), 3),
      sample: () => ({ x: tensor([[0]]) }),
    }
    const t = completeTruth({ program })
    expect(toFlat(t.logProb?.(tensor([[1], [2]]), tensor([1, 1])) ?? tensor([]))).toEqual([1, 2])
    expect(toFlat(t.regressionFunction?.(tensor([[1]])) ?? tensor([]))).toEqual([3])
    // The classes come from the program, so Bayes' rule gives the posterior: p(1 | x) = sigmoid(x).
    expect(toFlat(t.posterior?.(tensor([[0]])) ?? tensor([]))).toEqual([0.5, 0.5])
  })
})

group('recipes', () => {
  it('make the same data from the same recipe and record it', () => {
    const recipe = {
      generator: 'line',
      knobs: { n: 6 },
      seed: 4,
      modifiers: [{ name: 'outliers', knobs: { fraction: 0.5 } }],
    }
    const a = makeFromRecipe(recipe, { generators: { line } })
    const b = makeFromRecipe(recipe, { generators: { line } })
    expect(toFlat((a.data as RegressionData).x)).toEqual(toFlat((b.data as RegressionData).x))
    expect(a.data.meta.recipe?.modifiers?.[0]?.name).toBe('outliers')
    expect(() => makeFromRecipe({ generator: 'nope' }, { generators: { line } })).toThrow(/unknown generator/)
    expect(() =>
      makeFromRecipe({ generator: 'line', modifiers: [{ name: 'nope' }] }, { generators: { line } }),
    ).toThrow(/unknown modifier/)
    expect(describeRecipe(recipe)).toBe('line (n=6), then outliers (fraction=0.5); seed 4')
  })
  it('encode and parse', () => {
    const recipe = { generator: 'line', knobs: { n: 6, name: 'é' }, seed: 1 }
    expect(parseRecipe(encodeRecipe(recipe))).toEqual(recipe)
    expect(parseRecipe(JSON.stringify(recipe))).toEqual(recipe)
    expect(() => parseRecipe('{"seed": 1}')).toThrow(/parseRecipe/)
    expect(() => parseRecipe('!!')).toThrow(/parseRecipe/)
    expect(RecipeSchema.safeParse({ generator: 'x', seed: -1 }).success).toBe(false)
  })
})

group('loading', () => {
  const bytes = new TextEncoder().encode('1,2\n3,4')
  const parse = (b: Uint8Array): RegressionData => {
    const rows = new TextDecoder()
      .decode(b)
      .split('\n')
      .map((r) => r.split(',').map(Number))
    return xy(
      rows.map((r) => [r[0] ?? 0]),
      rows.map((r) => r[1] ?? 0),
    )
  }
  it('verifies checksums and uses the cache', async () => {
    const sha = await sha256Hex(bytes)
    let calls = 0
    const fetcher = async () => {
      calls++
      return { ok: true, status: 200, arrayBuffer: async () => bytes.slice().buffer }
    }
    const cache = memoryCache()
    const ref = { kind: 'remote' as const, name: 'pairs', url: 'https://example.org/pairs.csv', sha256: sha, parse }
    expect(toFlat((await load(ref, { fetch: fetcher, cache })).y)).toEqual([2, 4])
    await load(ref, { fetch: fetcher, cache })
    expect(calls).toBe(1)
    await expect(load({ ...ref, sha256: '0'.repeat(64) }, { fetch: fetcher })).rejects.toThrow(/checksum/)
    await expect(
      load(ref, { fetch: async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) }) }),
    ).rejects.toThrow(/404/)
    expect(() => indexedDbCache()).toThrow(/IndexedDB/)
  })
})

group('base helper', () => {
  it('quantileOf interpolates like numpy', () => {
    expect(quantileOf([1, 2, 3, 4], 0.25)).toBe(1.75)
    expect(Number.isNaN(quantileOf([], 0.5))).toBe(true)
  })
})

void asGenerated

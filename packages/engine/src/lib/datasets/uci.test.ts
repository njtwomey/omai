import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { type Dataset, rowCount, type TableColumn, tableToXy } from '../data/index.ts'
import { UCI_INFO, UCI_LICENCE } from './_uci.ts'
import { datasetRegistry } from './catalogue.ts'
import * as classificationUci from './classification-uci.ts'
import { datasetCollections } from './collections.ts'
import { findDataset, loadDataset, whereDatasets } from './registry.ts'
import * as regressionUci from './regression-uci.ts'
import * as tablesUci from './tables-uci.ts'
import { uciEntries } from './uci.ts'

// Every expected number below was counted from the file the repository publishes (rows, columns, class counts, means,
// missing marks), not from the vendored payload. The two subsamples (spambase, mushroom, adult) are checked against the
// stratified share of the full file: spam 1813 of 4601, edible 4208 of 8124, at most 50K 24720 of 32561.
// [name, rows, features, class counts, mean of the first feature, checksum of the payload]
const CLASSIFICATION: readonly (readonly [string, number, number, readonly number[], number, string])[] = [
  ['balanceScale', 625, 4, [288, 49, 288], 3.0, '8ad91f107259c170066dff0d3136cebf7b0a677a9df8985e5432207926186927'],
  [
    'banknoteAuthentication',
    1372,
    4,
    [762, 610],
    0.43373525706997085,
    'cd9d0b13464ef7fdd744982dbb2ec02996418fe7c804b20f9b435464f941b0e6',
  ],
  [
    'ecoli',
    336,
    7,
    [143, 77, 2, 2, 35, 20, 5, 52],
    0.5000595238095238,
    'd03699a9cfc4ec1cde86edb4a7f4acceb0954516da8049d1bdc0b9511171adfb',
  ],
  [
    'glassIdentification',
    214,
    9,
    [70, 76, 17, 13, 9, 29],
    1.5183654205607477,
    '01d29ff798e1140f03ae8da3bf7be5dc26c7385ce716e38b0724c3f403c96cf3',
  ],
  [
    'haberman',
    306,
    3,
    [225, 81],
    52.45751633986928,
    '4959429084d85fd2b198e50801088d18505ccfe7ea5374c3551ef245a8f3e4d3',
  ],
  [
    'ionosphere',
    351,
    34,
    [126, 225],
    0.8917378917378918,
    'e4511d10783ba7a2ad53945fc23d824f48bf4afc4b0c91cfcd71d735773a76ec',
  ],
  ['lenses', 24, 4, [4, 5, 15], 2.0, 'bca8398b40d2e04e9a5b9fa42c19d5ec30b02227eec0e78426d77339090b77c9'],
  [
    'seeds',
    210,
    7,
    [70, 70, 70],
    14.84752380952381,
    '926e7193a4786642dfd33020507b97a1c3056549e58658469ade58f9284b82d6',
  ],
  [
    'sonar',
    208,
    60,
    [97, 111],
    0.029163942307692307,
    '47c7c950d44a53c063ee34500ace83cc331fa9375ae74c0c9f43e71d6983a69c',
  ],
  [
    'vehicleSilhouettes',
    846,
    18,
    [218, 212, 217, 199],
    93.67848699763593,
    'b2738de7a901036c8adb54037d1a6f31321b175eef311d10532efa1fa3e6f84d',
  ],
  [
    'wholesaleCustomers',
    440,
    7,
    [298, 142],
    2.543181818181818,
    'de2bb42ed973e3108e7ed02bc1ee58fc6afe0190618d952a2029990950d80115',
  ],
  [
    'yeast',
    1484,
    8,
    [463, 5, 35, 44, 51, 163, 244, 429, 20, 30],
    0.500121293800539,
    '8ee9a58d1bd5c6a7eec161af5cb45779a257f6b705f0dae912f9e85b92b7f479',
  ],
  [
    'spambase',
    1000,
    57,
    [606, 394],
    0.10559999999999999,
    '33e1bf4c9461aa2d487ff59942300c0a644ab983dbf70187f74db821623bb0c1',
  ],
]

// [name, rows, features, mean of each target, checksum of the payload]
const REGRESSION: readonly (readonly [string, number, number, readonly number[], string])[] = [
  [
    'airfoilSelfNoise',
    1503,
    5,
    [124.83594278110445],
    '51424faa4879b0cc6f4891c6c41a53caa62f8eab948afa00709a489e02c66e28',
  ],
  [
    'concreteStrength',
    1030,
    8,
    [35.817961165048544],
    'c11618a466e1bce3c9a4d1852a13ce1b45025b6db6cb8ccd48edfd1415375cc5',
  ],
  [
    'energyEfficiency',
    768,
    8,
    [22.307200520833334, 24.58776041666667],
    '203de2fdb9b22e3b90c7f95b2b57173adf291daa1a62e76cbc75080470ac24c1',
  ],
  [
    'wineQualityRed',
    1599,
    11,
    [5.6360225140712945],
    'e12b8e3f9e637e211990f938c7cffd43e9972c407600c8db42807155e3fdde8b',
  ],
  [
    'wineQualityWhite',
    4898,
    11,
    [5.87790935075541],
    'c05bd9e65a2a614adb4a5785658f91c53d7c5dbc901e4dd4994d66c59678f4f1',
  ],
  [
    'yachtHydrodynamics',
    308,
    6,
    [10.495357142857143],
    '283f7593278d66c87fa8cc028427b7df85bef74ca3fb9ade3afec0c22c4ed34e',
  ],
]

// [name, rows, columns, class counts of the target (or null), mean of the target (or null), missing values, checksum]
const TABLES: readonly (readonly [string, number, number, readonly number[] | null, number | null, number, string])[] =
  [
    [
      'abalone',
      4177,
      9,
      null,
      9.933684462532918,
      0,
      'a8bd30deb711587fd50e7f555284884900d569b3adaffeb7f1ee0e21587cf6e6',
    ],
    [
      'adultIncome',
      2500,
      15,
      [1898, 602],
      null,
      330,
      '5d6dd69ef6efe8f413f181c4346cd68cc34816968b902455af519fab991b4c78',
    ],
    [
      'autoMpg',
      398,
      9,
      null,
      23.514572864321607,
      6,
      '9ba63500dde66f35a0a61b97173695371fde3a9cb38bd1e9e32fd577826ec7b0',
    ],
    [
      'bikeSharingDaily',
      731,
      15,
      null,
      4504.3488372093025,
      0,
      '062b3856b817ab61263c9c72e3a8d1b9f17af87521bc9fe8a0bf9e786ffbb0d9',
    ],
    [
      'carEvaluation',
      1728,
      7,
      [1210, 384, 69, 65],
      null,
      0,
      '3efebb9c67f9daae2424bd5d2b85e11ff17b75e299c995d8dc7abb65d9761193',
    ],
    [
      'congressionalVoting',
      435,
      17,
      [267, 168],
      null,
      392,
      '262d47739f86c8e60ac8fcd4cfb32488f93681e1be58364b8b49b6e076e2309d',
    ],
    [
      'dermatology',
      366,
      35,
      [112, 61, 72, 49, 52, 20],
      null,
      8,
      'efd05b022f06af0f631cd6d997d9acf9a900a008d2663f7ebcda17e8ac677ef5',
    ],
    [
      'forestFires',
      517,
      13,
      null,
      12.847292069632495,
      0,
      'c65b5aefbc15d72f6601319eda2a954d5c55d9db99bac11213ad0757896bec2b',
    ],
    [
      'heartDisease',
      303,
      14,
      [164, 55, 36, 35, 13],
      null,
      6,
      '13317556b71c89ac64ecb055cf34f00774bea56b2fb787fe7b08b50ee72f1fb7',
    ],
    ['hepatitis', 155, 20, [32, 123], null, 167, '05796f632e3321ec8f994ce926472269ed13230962e8673ffc5c58b3be64d427'],
    ['mushroom', 2000, 23, [1036, 964], null, 606, '8a855a1a9ae9d4d85213e8e1481547953af3bb23d9ac61ac61442e34cc40d016'],
    ['servo', 167, 5, null, 1.3897083879041916, 0, '8dd14e5459fb9f0b612fe14dd21dcd9d978b04b79f8bb3ce112ba39598e65844'],
    [
      'studentMath',
      395,
      33,
      null,
      10.415189873417722,
      0,
      '6fb0763dfac154f1ceb2dfe1b9d2563b52d9bc2cb82211be0bc764c2e91cbdcd',
    ],
    ['ticTacToe', 958, 10, [332, 626], null, 0, '289c3df55587a906fc49f807eb2f5653820dfb57fa3fb0dbd665b490ebb71a81'],
    [
      'zoo',
      101,
      18,
      [41, 20, 5, 13, 4, 8, 10],
      null,
      0,
      'a5d4f5289c75a47c80ae3e50f62a1522f01e238e47a5694037a1801d72ab0a0d',
    ],
  ]

const classification = classificationUci as unknown as Record<string, () => Dataset & { kind: 'classification' }>
const regression = regressionUci as unknown as Record<string, () => Dataset & { kind: 'regression' }>
const tables = tablesUci as unknown as Record<string, () => Dataset & { kind: 'table' }>

const missingOf = (column: TableColumn): number => {
  if (column.kind === 'numeric') return column.values.data.filter((v) => Number.isNaN(v)).length
  if (column.kind === 'categorical') return column.codes.data.filter((v) => v < 0).length
  return 0
}

describe('the UCI classification datasets', () => {
  it.each(CLASSIFICATION)(
    '%s has the rows, classes and checksum of the published file',
    (name, n, d, counts, mean0, sha) => {
      const data = classification[name]?.()
      if (data === undefined || data.kind !== 'classification') throw new Error(name)
      expect(data.x.shape).toEqual([n, d])
      expect(data.featureNames.length).toBe(d)
      expect(data.classes.length).toBe(counts.length)
      for (const [j, count] of counts.entries())
        expect(data.y.data.filter((v) => v === j).length, data.classes[j]).toBe(count)
      let sum = 0
      for (let i = 0; i < n; i++) sum += data.x.data[i * d] ?? Number.NaN
      expect(sum / n).toBeCloseTo(mean0, 9)
      expect(data.x.data.every((v) => Number.isFinite(v))).toBe(true)
      expect(data.meta.sha256).toBe(sha)
      expect(data.meta.licence).toBe(UCI_LICENCE)
      expect(data.meta.name).toBe(name)
    },
  )

  it('keeps the share of spam in the subsample of spambase', () => {
    const [spam] = [classification['spambase']?.()]
    expect(Math.round((1000 * 1813) / 4601)).toBe(spam?.y.data.filter((v) => v === 1).length)
  })
})

describe('the UCI regression datasets', () => {
  it.each(REGRESSION)('%s has the rows, targets and checksum of the published file', (name, n, d, means, sha) => {
    const data = regression[name]?.()
    if (data === undefined || data.kind !== 'regression') throw new Error(name)
    expect(data.x.shape).toEqual([n, d])
    expect(data.targetNames.length).toBe(means.length)
    expect(data.y.shape).toEqual(means.length === 1 ? [n] : [n, means.length])
    for (const [j, mean] of means.entries()) {
      let sum = 0
      for (let i = 0; i < n; i++) sum += data.y.data[i * means.length + j] ?? Number.NaN
      expect(sum / n, data.targetNames[j]).toBeCloseTo(mean, 8)
    }
    expect(data.x.data.every((v) => Number.isFinite(v))).toBe(true)
    expect(data.meta.sha256).toBe(sha)
    expect(data.meta.licence).toBe(UCI_LICENCE)
  })
})

describe('the UCI tables', () => {
  it.each(TABLES)(
    '%s has the rows, columns, target, missing values and checksum of the published file',
    (name, n, columns, counts, mean, missing, sha) => {
      const data = tables[name]?.()
      if (data === undefined || data.kind !== 'table') throw new Error(name)
      expect(rowCount(data)).toBe(n)
      expect(Object.keys(data.columns).length).toBe(columns)
      const target = data.columns[data.target ?? '']
      if (target === undefined) throw new Error(`${name}: no target`)
      expect(missingOf(target)).toBe(0)
      if (counts !== null && target.kind === 'categorical') {
        expect(target.levels.length).toBe(counts.length)
        for (const [j, count] of counts.entries())
          expect(target.codes.data.filter((v) => v === j).length, target.levels[j]).toBe(count)
      } else if (mean !== null && target.kind === 'numeric') {
        expect(target.values.data.reduce((a, b) => a + b, 0) / n).toBeCloseTo(mean, 8)
      } else {
        throw new Error(`${name}: target of the wrong kind`)
      }
      expect(Object.values(data.columns).reduce((total, c) => total + missingOf(c), 0)).toBe(missing)
      expect(data.meta.sha256).toBe(sha)
      expect(data.meta.licence).toBe(UCI_LICENCE)
    },
  )

  it('keeps missing values as NaN when a table becomes features and a target', () => {
    const cars = tablesUci.autoMpg()
    const xy = tableToXy(cars)
    const [n, d] = xy.x.shape
    expect(n).toBe(398)
    expect(xy.x.data.filter((v) => Number.isNaN(v)).length).toBe(6)
    expect(d).toBe(Object.keys(cars.columns).length - 2) // the target and the text column are not features
  })

  it('keeps the shares of the stratified subsamples of mushroom and adult', () => {
    const mushroom = tablesUci.mushroom().columns['class']
    const income = tablesUci.adultIncome().columns['income']
    if (mushroom?.kind !== 'categorical' || income?.kind !== 'categorical') throw new Error('target')
    expect(mushroom.codes.data.filter((v) => v === 0).length).toBe(Math.round((2000 * 4208) / 8124))
    expect(income.codes.data.filter((v) => v === 0).length).toBe(Math.round((2500 * 24720) / 32561))
  })
})

describe('the UCI datasets in the registry', () => {
  const entries = uciEntries()
  const names = Object.keys(UCI_INFO)

  it('has 34 datasets, each with provenance, and as many registry entries', () => {
    expect(names.length).toBe(34)
    expect(entries.map((e) => e.name).sort()).toEqual([...names].sort())
    expect(CLASSIFICATION.length + REGRESSION.length + TABLES.length).toBe(34)
  })

  it.each(names)('%s is registered with its tags, licence and a DOI citation', (name) => {
    const entry = lookup(datasetRegistry, name)
    const info = UCI_INFO[name]
    expect(entry.form).toBe('vendored')
    expect(entry.kind).toBe(info?.kind)
    expect(entry.modality).toBe('tabular')
    expect(entry.real).toBe(true)
    expect(entry.size).toBe(info?.size)
    expect(entry.licence).toBe(UCI_LICENCE)
    expect(entry.source).toContain('UCI Machine Learning Repository')
    expect(info?.citation).toMatch(/https:\/\/doi\.org\/10\.24432\/[A-Z0-9]+\.$/)
    const data = loadDataset(entry).data
    expect(data.kind).toBe(entry.kind)
    expect(data.meta.url).toMatch(/^https:\/\/archive\.ics\.uci\.edu\/dataset\/\d+\//)
    expect(rowCount(data)).toBe(entry.size)
  })

  it('is found by kind, by text and by the real flag', () => {
    const found = (query: Parameters<typeof whereDatasets>[1]) =>
      new Set(whereDatasets(datasetRegistry, query).map((e) => e.name))
    expect(found({ kind: 'table', real: true }).has('heartDisease')).toBe(true)
    expect(found({ kind: 'regression', text: 'wine quality' })).toEqual(new Set(['wineQualityRed', 'wineQualityWhite']))
    expect(found({ kind: 'classification', text: 'subsample' })).toEqual(new Set(['spambase']))
    expect(found({ text: 'cleveland' })).toEqual(new Set(['heartDisease']))
    expect(findDataset(datasetRegistry, 'autoMpg')?.title).toBe('Auto MPG')
    for (const name of names) {
      const kind = UCI_INFO[name]?.kind
      if (kind === undefined) throw new Error(name)
      expect(found({ kind, real: true }).has(name), name).toBe(true)
    }
  })

  it('is in the collections the pickers offer', () => {
    const inCollection = (id: string): Set<string> => new Set(datasetCollections[id]?.entries.map((e) => e.name))
    const classes = inCollection('classificationTabular')
    const regs = inCollection('regressionTabular')
    const mixed = inCollection('mixedTables')
    for (const [name] of CLASSIFICATION) expect(classes.has(name), name).toBe(true)
    for (const [name] of REGRESSION) expect(regs.has(name), name).toBe(true)
    for (const [name] of TABLES) expect(mixed.has(name), name).toBe(true)
    for (const id of ['classificationTabular', 'regressionTabular', 'mixedTables', 'clusteringTabular']) {
      for (const e of datasetCollections[id]?.entries ?? []) {
        if ('load' in e) expect(rowCount(e.load()), e.name).toBeGreaterThan(0)
      }
    }
  })
})

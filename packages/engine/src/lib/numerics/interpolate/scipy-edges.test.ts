import { describe, expect, it } from 'vitest'
import { toArray, type TensorLike, type Value } from '../../tensor/index.ts'
import * as I from './index.ts'

// Interpolants on awkward data (two and three points, flat monotone stretches, periodic data, queries on, outside and
// between the knots, in value and derivatives) against scipy.interpolate 1.18 (one-off run, same data).
const NaN_ = Number.NaN
const x1 = [0, 0.5, 1.2, 2.0, 3.1, 3.5, 5.0]
const y1 = [0.3, -0.2, 1.1, 0.8, -0.5, 0.25, 0.9]
const q1 = [-1, 0, 0.25, 1.2, 2.5, 3.3, 5, 6.5]
const xm = [0, 1, 2, 3, 4]
const ym = [0, 0, 1, 1, 1.5]
const qm = [-0.5, 0.5, 1, 1.5, 2.5, 3.5, 4.5]
const grid = { x: [0, 1, 2.5, 4], y: [0, 2, 3] }
const gridValues = [
  [0.0, 0.5646424733950354, 0.7833269096274833],
  [0.8414709848078965, 1.2852878887557908, 1.374871516258843],
  [0.5984721441039564, 0.7558663767190048, 0.8158874694017402],
  [-0.7568024953079283, 0.14916613922367838, 0.7318331016613817],
]
const gridQueries = [
  [0.5, 1],
  [1, 2],
  [2, 0.5],
  [3.9, 2.9],
  [4, 3],
  [0, 0],
  [2.5, 3],
]
const sites = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
  [0.5, 0.5],
  [0.2, 0.8],
  [0.9, 0.3],
  [0.4, 0.1],
]
const siteValues = [
  0.0, 0.1411200080598672, 1.0, 1.1411200080598671, 1.2474949866040546, 1.2046424733950356, 0.5173798802338297,
  0.9420390859672264,
]
const targets = [
  [0.3, 0.3],
  [0.7, 0.6],
  [0.5, 0.05],
  [0.1, 0.9],
  [1.2, 1.2],
]

// Each row: the scipy values at the queries (q1, qm, the grid queries or the scattered targets, by the row's name).
const expected = new Map<string, readonly number[]>([
  [
    'cubic not-a-knot nu0',
    [
      13.585019424591664, 0.3, -0.2555881471145417, 1.1, -0.10844588192683677, -0.21401486047510657, 0.9000000000000004,
      -28.72176465953047,
    ],
  ],
  [
    'cubic not-a-knot nu1',
    [
      -25.43575548939357, -3.774765436505844, -0.8349698702052445, 1.3058045793680098, -1.6794536562752502,
      1.945215139804757, -5.707653272492518, -37.73763455423242,
    ],
  ],
  [
    'cubic not-a-knot nu2',
    [
      29.582436283035992, 13.739543822739464, 9.778820707665332, -5.271927129616374, 1.855718688484204,
      4.45074302375531, -13.45411762645799, -29.25252408252855,
    ],
  ],
  [
    'cubic natural nu0',
    [
      -0.9209529320701701, 0.3, -0.08880955825438572, 1.1, -0.1572362208230654, -0.16756043341073532,
      0.8999999999999999, 1.549999999999999,
    ],
  ],
  [
    'cubic natural nu1',
    [
      7.143494084257292, -1.7403176440233905, -1.1850794110058476, 1.5008238904367708, -1.7326352636530087,
      2.047428343296611, -0.32787382375706353, 1.9557476475141282,
    ],
  ],
  [
    'cubic natural nu2',
    [
      -17.767623456561367, 8.881784197001252e-16, 4.441905864140343, -5.9689871500235725, 2.156375507073051,
      2.128021670536751, 0.0, 3.0448286283615884,
    ],
  ],
  [
    'cubic clamped nu0',
    [
      -13.343726617665066, 0.3, 0.05371118106598888, 1.1, -0.16744090048757643, -0.1642615335413794, 0.9000000000000001,
      3.386988305419644,
    ],
  ],
  [
    'cubic clamped nu1',
    [
      35.049937647106844, 0.0, -1.4851552757360444, 1.6564472772965189, -1.7292834718334513, 2.0526429501369536,
      4.440892098500626e-16, 4.361647175699407,
    ],
  ],
  [
    'cubic clamped nu2',
    [
      -58.33739088243698, -11.762484411776711, -0.11875779411164444, -6.622608325403878, 2.228908939308277,
      1.9630766770689514, 0.8164392468531751, 4.999090320746033,
    ],
  ],
  [
    'cubic mixed nu0',
    [
      -16.915523288314986, 0.3, 0.09462996574671886, 1.1, -0.1643244314619295, -0.17030341411011884, 0.8999999999999999,
      -0.6999999999999993,
    ],
  ],
  [
    'cubic mixed nu1',
    [
      43.07472876883997, 0.5, -1.5714801370131246, 1.6990099407765826, -1.7189512849085329, 2.0383697741482827,
      -0.7256472964023089, -0.9987054071953789,
    ],
  ],
  ['cubic two points', [0.0, 1.5, 4.0]],
  ['cubic three points', [3.333333333333334, 0.3333333333333333, 0.3333333333333334, 4.999999999999999]],
  ['cubic periodic', [0.5, 1.0, -0.5159999999999999, 1.3389629629629631, 1.0, -1.125142857142857, -1.0]],
  [
    'cubic periodic nu1',
    [
      1.6666666666666665, -1.619047619047619, -2.05904761904762, 0.5955555555555552, -1.619047619047619,
      0.15809523809523734, -1.0476190476190477,
    ],
  ],
  [
    'pchip',
    [
      6.014285714285716, 0.3, -0.08690476190476193, 1.1, 0.15594536137567613, -0.16502232464352642, 0.9000000000000001,
      0.051339478611551304,
    ],
  ],
  [
    'pchip nu1',
    [
      -10.000000000000004, -2.1904761904761907, -0.9523809523809523, 0.0, -1.648147534175832, 2.6123883767823703, 0.0,
      -1.197767535647415,
    ],
  ],
  ['pchip flat', [0.0, 0.0, 0.0, 0.5, 1.0, 1.15625, 1.84375]],
  ['pchip flat nu1', [0.0, 0.0, 0.0, 1.5, 0.0, 0.5625, 0.5625]],
  ['akima', [NaN_, 0.3, -0.13953634085213035, 1.1, 0.1424836987045502, -0.174015197840625, 0.9, NaN_]],
  [
    'makima',
    [NaN_, 0.3, -0.11562500000000005, 1.1, 0.12947601123167327, -0.17209364563477184, 0.8999999999999998, NaN_],
  ],
  ['akima flat', [NaN_, -0.125, 0.0, 0.5208333333333333, 1.0, 1.1979166666666667, NaN_]],
  [
    'interp1d linear',
    [1.2999999999999998, 0.3, 0.04999999999999999, 1.1, 0.2090909090909092, -0.12500000000000042, 0.9, 1.55],
  ],
  ['interp1d nearest', [NaN_, 0.3, 0.3, 1.1, 0.8, -0.5, 0.9, NaN_]],
  ['interp1d previous', [NaN_, 0.3, 0.3, 1.1, 0.8, -0.5, 0.9, NaN_]],
  ['interp1d next', [NaN_, 0.3, -0.2, 1.1, -0.5, 0.25, 0.9, NaN_]],
  ['interp1d zero', [NaN_, 0.3, 0.3, 1.1, 0.8, -0.5, 0.9, NaN_]],
  [
    'interp1d slinear',
    [1.2999999999999998, 0.3, 0.04999999999999999, 1.1, 0.2090909090909092, -0.12500000000000042, 0.9, 1.55],
  ],
  [
    'interp1d quadratic',
    [
      6.07898811592512, 0.3, -0.14912450483021336, 1.1, -0.07090041673568977, -0.1657082790993742, 0.9,
      -2.80496251749171,
    ],
  ],
  [
    'interp1d cubic',
    [
      13.585019424591666, 0.3, -0.2555881471145417, 1.1, -0.10844588192683718, -0.2140148604751064, 0.9,
      -28.721764659530454,
    ],
  ],
  [
    'interp spline k1',
    [1.2999999999999998, 0.3, 0.04999999999999999, 1.1, 0.2090909090909092, -0.12500000000000042, 0.9, 1.55],
  ],
  [
    'interp spline k2',
    [
      6.07898811592512, 0.3, -0.14912450483021336, 1.1, -0.07090041673568977, -0.1657082790993742, 0.9,
      -2.80496251749171,
    ],
  ],
  [
    'interp spline k3',
    [
      13.585019424591666, 0.3, -0.2555881471145417, 1.1, -0.10844588192683718, -0.2140148604751064, 0.9,
      -28.721764659530454,
    ],
  ],
  [
    'interp spline k4',
    [
      20.81101218300588, 0.3, -0.3137769022950902, 1.1000000000000003, -0.140564850851775, -0.2238966555029532, 0.9,
      -68.12851665719369,
    ],
  ],
  [
    'interp spline k5',
    [
      24.692237247000776, 0.3, -0.3357252212134653, 1.1, -0.1210694426419665, -0.23618887378710177, 0.9,
      -114.50677762802917,
    ],
  ],
  [
    'interp spline k3 nu1',
    [
      -25.43575548939358, -3.7747654365058443, -0.8349698702052448, 1.3058045793680102, -1.679453656275251,
      1.9452151398047577, -5.707653272492514, -37.73763455423239,
    ],
  ],
  [
    'lsq spline',
    [
      10.970942983741153, 0.29103689570784946, -0.1929539321809348, 1.0490889827366663, -0.18234462868079687,
      -0.2106947142374225, 0.8997110048652046, -27.213181403335458,
    ],
  ],
  [
    'grid linear',
    [
      0.6728503367396808, 1.2852878887557908, 0.7426888717701023, 0.6826543357320016, 0.7318331016613817, 0.0,
      0.8158874694017402,
    ],
  ],
  [
    'grid nearest',
    [0.0, 1.2852878887557908, 0.5984721441039564, 0.7318331016613817, 0.7318331016613817, 0.0, 0.8158874694017402],
  ],
  [
    'grid slinear',
    [
      0.6728503367396808, 1.285287888755791, 0.7426888717701026, 0.6826543357320016, 0.7318331016613816, 0.0,
      0.8158874694017403,
    ],
  ],
  [
    'rbf thin_plate_spline',
    [1.0581157108266628, 1.1134565067706934, 0.8399830884401437, 1.101990353084807, 1.2046394633972763],
  ],
  ['rbf linear', [0.947363892126315, 1.093052931940312, 0.7734276148757615, 1.0975272559490419, 1.0817407448676128]],
  ['rbf cubic', [1.059026169494412, 1.124967527334936, 0.8857893249213343, 1.1098043921858012, 1.1082696907230485]],
  ['rbf gaussian', [1.0256205385483876, 1.1066055187065265, 0.9328215824748218, 1.1124056092264132, 0.992294793893442]],
  [
    'rbf multiquadric',
    [1.0659045316822482, 1.107420523426775, 0.8973897129976258, 1.1066774281842022, 1.1424363914075286],
  ],
  [
    'rbf inverse_multiquadric',
    [1.0497020407932034, 1.1192037169029585, 0.9018551458576818, 1.1081611658884514, 1.038762341136259],
  ],
  [
    'rbf tps smoothing',
    [0.9657935319809878, 1.1105975970592616, 0.7107286024374188, 1.1061991556762, 1.208733736334274],
  ],
  ['griddata linear', [0.7484969919624327, 1.0974424745972338, 0.5133555454015734, 1.102321236697518, -9.0]],
  ['griddata nearest', [0.9420390859672264, 1.2474949866040546, 0.9420390859672264, 1.0, 1.1411200080598671]],
])

const arr = (v: Value): number[] => toArray(v) as number[]
const ok = (got: number, want: number): boolean =>
  Number.isNaN(want) ? Number.isNaN(got) : Math.abs(got - want) <= 1e-9 * Math.max(1, Math.abs(want))

describe('interpolation against scipy.interpolate', () => {
  it('cubic splines, PCHIP, Akima, interp1d and B-splines match scipy at, between and outside the knots, in value and derivatives', () => {
    const failures: string[] = []
    const compare = (name: string, got: Value): void => {
      const want = expected.get(name) ?? []
      const g = arr(got)
      if (g.length !== want.length) failures.push(`${name}: ${g.length} values, scipy ${want.length}`)
      want.forEach((w, i) => {
        if (!ok(g[i] ?? Number.NaN, w)) failures.push(`${name}[${i}]: ${g[i]}, scipy ${w}`)
      })
    }
    for (const bc of ['not-a-knot', 'natural', 'clamped'] as const) {
      const s = I.cubicSpline(x1, y1, bc)
      for (const nu of [0, 1, 2]) compare(`cubic ${bc} nu${nu}`, s.evaluate(q1, nu))
    }
    const mixed = I.cubicSpline(x1, y1, [
      [1, 0.5],
      [2, -1],
    ])
    for (const nu of [0, 1]) compare(`cubic mixed nu${nu}`, mixed.evaluate(q1, nu))
    compare('cubic two points', I.cubicSpline([0, 2], [1, 3]).evaluate([-1, 0.5, 3]))
    compare('cubic three points', I.cubicSpline([0, 1, 3], [1, 0, 2]).evaluate([-1, 0.5, 2, 4]))
    const periodic = I.cubicSpline([0, 1, 2.5, 4], [1, -1, 0.5, 1], 'periodic')
    compare('cubic periodic', periodic.evaluate([-1.5, 0, 0.7, 3.2, 4, 5.3, 9]))
    compare('cubic periodic nu1', periodic.evaluate([-1.5, 0, 0.7, 3.2, 4, 5.3, 9], 1))
    const pchip = I.pchip(x1, y1)
    compare('pchip', pchip.evaluate(q1))
    compare('pchip nu1', pchip.evaluate(q1, 1))
    // Flat stretches stay flat (no overshoot): the derivative is exactly 0 where the data repeat.
    const flat = I.pchip(xm, ym)
    compare('pchip flat', flat.evaluate(qm))
    compare('pchip flat nu1', flat.evaluate(qm, 1))
    compare('akima', I.akima(x1, y1, 'akima', false).evaluate(q1))
    compare('makima', I.akima(x1, y1, 'makima', false).evaluate(q1))
    compare('akima flat', I.akima(xm, ym, 'akima', false).evaluate(qm))
    const outside = (kind: I.Interp1dOptions['kind']): I.Interp1dOptions => ({
      kind,
      fillValue: ['linear', 'slinear', 'quadratic', 'cubic'].includes(kind ?? '') ? 'extrapolate' : Number.NaN,
    })
    for (const kind of ['linear', 'nearest', 'previous', 'next', 'zero', 'slinear', 'quadratic', 'cubic'] as const) {
      compare(`interp1d ${kind}`, I.interp1d(x1, y1, outside(kind))(q1))
    }
    for (const k of [1, 2, 3, 4, 5]) compare(`interp spline k${k}`, I.makeInterpSpline(x1, y1, k).evaluate(q1))
    compare('interp spline k3 nu1', I.makeInterpSpline(x1, y1, 3).evaluate(q1, 1))
    compare('lsq spline', I.makeLsqSpline(x1, y1, [0, 0, 0, 0, 1.5, 3, 5, 5, 5, 5], 3).evaluate(q1))
    expect(failures).toEqual([])
  })

  it('regular-grid, radial-basis and scattered-data interpolation match scipy, and a periodic or out-of-range query is handled', () => {
    const failures: string[] = []
    const compare = (name: string, got: Value, tol = 1e-9): void => {
      const want = expected.get(name) ?? []
      const g = arr(got)
      if (g.length !== want.length) failures.push(`${name}: ${g.length} values, scipy ${want.length}`)
      want.forEach((w, i) => {
        if (!(Math.abs((g[i] ?? Number.NaN) - w) <= tol * Math.max(1, Math.abs(w))))
          failures.push(`${name}[${i}]: ${g[i]}, scipy ${w}`)
      })
    }
    for (const method of ['linear', 'nearest', 'slinear'] as const) {
      compare(`grid ${method}`, I.regularGridInterpolator([grid.x, grid.y], gridValues, { method })(gridQueries))
    }
    const rbf = (name: string, options: I.RbfOptions): void =>
      compare(`rbf ${name}`, I.rbfInterpolator(sites, siteValues, options)(targets), 1e-7)
    rbf('thin_plate_spline', {})
    rbf('linear', { kernel: 'linear' })
    rbf('cubic', { kernel: 'cubic' })
    rbf('gaussian', { kernel: 'gaussian', epsilon: 2 })
    rbf('multiquadric', { kernel: 'multiquadric', epsilon: 1.5 })
    rbf('inverse_multiquadric', { kernel: 'inverse_multiquadric', epsilon: 1.5 })
    rbf('tps smoothing', { smoothing: 0.1 })
    compare('griddata linear', I.griddata(sites, siteValues, targets, 'linear', -9), 1e-9)
    compare('griddata nearest', I.griddata(sites, siteValues, targets, 'nearest'), 1e-12)
    // Outside the grid is an error unless a fill value is given; a kernel that is not scale invariant needs epsilon.
    expect(() => I.regularGridInterpolator([grid.x, grid.y], gridValues)([[5, 1]])).toThrow(
      expect.objectContaining({ code: 'value' }),
    )
    expect(
      arr(
        I.regularGridInterpolator([grid.x, grid.y], gridValues, { boundsError: false })([
          [5, 1],
          [1, 2],
        ]),
      )[0],
    ).toBeNaN()
    expect(() => I.rbfInterpolator(sites, siteValues, { kernel: 'gaussian' })).toThrow(
      expect.objectContaining({ code: 'value' }),
    )
    expect(() => I.cubicSpline([0, 1, 2], [1, 2, 3], 'periodic')).toThrow(expect.objectContaining({ code: 'value' }))
    expect(() => I.cubicSpline([0], [1])).toThrow(expect.objectContaining({ code: 'value' }))
  })
})

import { describe, expect, it } from 'vitest'
import { maxOf } from '../../lib/base/index.ts'
import {
  describe as describeData,
  generate,
  load,
  memoryCache,
  rowCount,
  sha256Hex,
  tableToXy,
} from '../../lib/data/index.ts'
import { key } from '../../lib/random/index.ts'
import { sum, type Tensor, tensor, toFlat } from '../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import {
  anscombe,
  breastCancer,
  californiaHousing,
  digitImages,
  digits,
  growthChart,
  growthCurves,
  growthValue,
  iris,
  mtcars,
  mtcarsRegression,
  penguins,
  penguinsClassification,
  titanic,
  wine,
} from './index.ts'

describe('vendored classification data against their sources', () => {
  const loaders: Record<string, () => { x: Tensor; y: Tensor }> = {
    iris,
    wine,
    breastCancer,
    digits,
    penguinsClassification,
  }
  for (const c of loadFixture('methods/datasets/classification-real').cases) {
    it(c.name, () => {
      const d = loaders[c.fn]?.()
      if (d === undefined) throw new Error(`no loader ${c.fn}`)
      if (c.name === 'digits/pixel-sums') {
        expectMatches(sum(d.x, 0), c.output, 0, 0, c.name)
        return
      }
      if (c.name === 'penguins/means') {
        const n = d.x.shape[0] ?? 1
        expectMatches(tensor(toFlat(sum(d.x, 0)).map((v) => v / n)), c.output, c.rtol, 1e-12, c.name)
        return
      }
      expectMatches(c.params['part'] === 'x' ? d.x : d.y, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('vendored regression data against their sources', () => {
  for (const c of loadFixture('methods/datasets/regression-real').cases) {
    it(c.name, () => {
      if (c.fn === 'anscombe') {
        const d = anscombe(c.params['set'] as 1 | 2 | 3 | 4)
        const x = toFlat(d.x)
        const y = toFlat(d.y)
        const n = x.length
        const mx = x.reduce((a, b) => a + b, 0) / n
        const my = y.reduce((a, b) => a + b, 0) / n
        const sxx = x.reduce((a, v) => a + (v - mx) ** 2, 0)
        const syy = y.reduce((a, v) => a + (v - my) ** 2, 0)
        const sxy = x.reduce((a, v, i) => a + (v - mx) * ((y[i] ?? 0) - my), 0)
        const slope = sxy / sxx
        const stats = [mx, my, sxx / (n - 1), syy / (n - 1), sxy / Math.sqrt(sxx * syy), slope, my - slope * mx]
        expectMatches(tensor(stats), c.output, c.rtol, c.atol, c.name)
        return
      }
      if (c.fn === 'mtcars') {
        expectMatches(mtcarsRegression().y, c.output, 0, 0, c.name)
        return
      }
      const chart = growthChart(c.params['measure'] as 'weight' | 'stature', 'female')
      const ages = toFlat(decode(c.inputs[0]) as Tensor)
      const z = c.params['z'] as number
      expectMatches(tensor(ages.map((a) => growthValue(chart, a, z))), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('real datasets', () => {
  it('carry provenance and the expected shapes', () => {
    expect(iris().meta.sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(digitImages().images.shape).toEqual([1797, 8, 8, 1])
    expect(maxOf(toFlat(digits().x))).toBe(16)
    expect(rowCount(penguins())).toBe(344)
    expect(penguins().meta.licence).toBe('CC0-1.0')
    expect(describeData(penguins()).classCounts).toEqual({ Adelie: 152, Chinstrap: 68, Gentoo: 124 })
    const t = titanic()
    expect(describeData(t).classCounts).toEqual({ no: 1490, yes: 711 })
    expect(tableToXy(t).featureNames).toEqual([
      'class=1st',
      'class=2nd',
      'class=3rd',
      'class=Crew',
      'sex=female',
      'sex=male',
      'age=child',
      'age=adult',
    ])
    expect(mtcars().columns['model']?.kind).toBe('text')
  })

  it('growth curves follow the chart quantiles', () => {
    const { data, truth } = generate(growthCurves, { key: key(0), knobs: { n: 4000, measure: 'stature', sex: 'male' } })
    const q90 = toFlat(truth.quantile?.(data.x, 0.9) as Tensor)
    const below = toFlat(data.y).filter((v, i) => v <= (q90[i] ?? 0)).length / 4000
    expect(Math.abs(below - 0.9)).toBeLessThan(4 * Math.sqrt(0.09 / 4000))
    const chart = growthChart('stature', 'male')
    const mean = toFlat(truth.regressionFunction?.(tensor([[120]])) as Tensor)[0] ?? 0
    expect(Math.abs(mean - growthValue(chart, 120, 0))).toBeLessThan(0.5)
  })

  it('California housing parses the StatLib archive into sklearn form', async () => {
    const csv =
      '-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0\n-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014,358500.0\n'
    const body = new TextEncoder().encode(csv)
    const header = new Uint8Array(512)
    header.set(new TextEncoder().encode('CaliforniaHousing/cal_housing.data'), 0)
    header.set(new TextEncoder().encode(`${body.length.toString(8).padStart(11, '0')}\0`), 124)
    header[156] = 48
    const tar = new Uint8Array(512 + Math.ceil(body.length / 512) * 512 + 1024)
    tar.set(header, 0)
    tar.set(body, 512)
    const gz = new Uint8Array(
      await new Response(new Blob([tar]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer(),
    )
    const ref = { ...californiaHousing, sha256: await sha256Hex(gz) }
    const fetcher = async () => ({ ok: true, status: 200, arrayBuffer: async () => gz.slice().buffer })
    const d = await load(ref, { fetch: fetcher, cache: memoryCache() })
    expect(d.featureNames).toEqual([
      'MedInc',
      'HouseAge',
      'AveRooms',
      'AveBedrms',
      'Population',
      'AveOccup',
      'Latitude',
      'Longitude',
    ])
    expect(toFlat(d.y)).toEqual([4.526, 3.585])
    expect(toFlat(d.x).slice(0, 4)).toEqual([8.3252, 41, 880 / 126, 129 / 126])
  })
})

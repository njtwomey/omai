import { describe, expect, it } from 'vitest'
import { decode, expectMatches, type GoldenCase, loadFixture } from '../../testing/golden.ts'
import { fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { halton, key, poissonDisk, scrambledSobol, sobol } from './index.ts'

const fixture = loadFixture('random/sequences')
const num = (c: GoldenCase, name: string): number => Number(c.params[name])

/** Hickernell's centred L2 discrepancy (squared), the default of `scipy.stats.qmc.discrepancy`. */
function centredDiscrepancy(x: Tensor): number {
  const [n = 0, d = 0] = x.shape
  const v = x.data
  let single = 0
  for (let i = 0; i < n; i++) {
    let p = 1
    for (let k = 0; k < d; k++) {
      const a = Math.abs((v[i * d + k] ?? 0) - 0.5)
      p *= 1 + 0.5 * a - 0.5 * a * a
    }
    single += p
  }
  let pairs = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let p = 1
      for (let k = 0; k < d; k++) {
        const xi = v[i * d + k] ?? 0
        const xj = v[j * d + k] ?? 0
        p *= 1 + 0.5 * Math.abs(xi - 0.5) + 0.5 * Math.abs(xj - 0.5) - 0.5 * Math.abs(xi - xj)
      }
      pairs += p
    }
  }
  return (13 / 12) ** d - (2 / n) * single + pairs / (n * n)
}

function weighted(rows: number[][]): Tensor<'float64'> {
  const out = rows.map((row) => {
    let a = 0
    let b = 0
    row.forEach((x, j) => {
      a += x * (j + 1)
      b += x * (((j * j) % 97) + 1)
    })
    return [a, b]
  })
  return fromStorage(Float64Array.from(out.flat()), [rows.length, 2], 'float64')
}

function rowsOf(x: Tensor): number[][] {
  const [n = 0, d = 0] = x.shape
  return Array.from({ length: n }, (_, i) => Array.from(x.data.slice(i * d, (i + 1) * d)))
}

type Check = (c: GoldenCase) => void

const CHECKS: Readonly<Record<string, Check>> = {
  sobol: (c) => {
    const x = sobol(num(c, 'n'), num(c, 'd'), { skip: num(c, 'skip'), bits: num(c, 'bits') })
    expectMatches(x, c.output, c.rtol, c.atol, c.name)
  },
  sobolColumns: (c) => {
    const x = sobol(num(c, 'n'), num(c, 'd'), { skip: num(c, 'skip') })
    const columns = c.params['columns'] as number[]
    const rows = rowsOf(x).map((row) => columns.map((j) => row[j] ?? Number.NaN))
    const got = fromStorage(Float64Array.from(rows.flat()), [rows.length, columns.length], 'float64')
    expectMatches(got, c.output, c.rtol, c.atol, c.name)
  },
  sobolDirections: (c) => {
    // Point 2^(j+1) - 1 has Gray code 2^j, so it is the j-th direction numbers over 2^bits.
    const d = num(c, 'd')
    const bits = num(c, 'bits')
    const v = Array.from({ length: bits }, (_, j) =>
      Array.from(sobol(1, d, { skip: 2 ** (j + 1) - 1, bits }).data, (x) => x * 2 ** bits),
    )
    const sums = weighted(v)
    // The fixture is [2, bits]: weights by rows; ours is [bits, 2].
    const got = fromStorage(Float64Array.from([...column(sums, 0), ...column(sums, 1)]), [2, bits], 'float64')
    expectMatches(got, c.output, c.rtol, c.atol, c.name)
  },
  sobolFingerprint: (c) => {
    const x = sobol(num(c, 'n'), num(c, 'd'), { skip: num(c, 'skip') })
    expectMatches(weighted(rowsOf(x)), c.output, c.rtol, c.atol, c.name)
  },
  halton: (c) => {
    const x = halton(num(c, 'n'), num(c, 'd'), { skip: num(c, 'skip') })
    expectMatches(x, c.output, c.rtol, c.atol, c.name)
  },
  centeredDiscrepancy: (c) => {
    // Checks the discrepancy used below against scipy's on the same points.
    expectMatches(centredDiscrepancy(decode(c.inputs[0]) as Tensor), c.output, c.rtol, c.atol, c.name)
  },
  scrambledDiscrepancy: (c) => {
    const [scipyScrambled, scipyUniform] = toFlat(decode(c.output) as Tensor)
    const n = num(c, 'n')
    const d = num(c, 'd')
    let total = 0
    for (let s = 0; s < 20; s++) total += centredDiscrepancy(scrambledSobol(key(s), n, d))
    const ours = total / 20
    // Same class as scipy's LMS+shift (mean over 20 scrambles), and far below independent uniforms.
    expect(Math.abs(ours - (scipyScrambled ?? 0))).toBeLessThanOrEqual(c.rtol * (scipyScrambled ?? 0))
    expect(ours).toBeLessThan((scipyUniform ?? 0) / 4)
  },
  poissonDiskCount: (c) => {
    const want = toFlat(decode(c.output) as Tensor)[0] ?? 0
    let total = 0
    for (let s = 0; s < 5; s++) total += poissonDisk(key(s), num(c, 'd'), num(c, 'radius')).shape[0] ?? 0
    expect(Math.abs(total / 5 - want)).toBeLessThanOrEqual(c.rtol * want)
  },
}

function column(x: Tensor, j: number): number[] {
  const [n = 0, w = 0] = x.shape
  return Array.from({ length: n }, (_, i) => x.data[i * w + j] ?? Number.NaN)
}

describe('sequences against scipy.stats.qmc', () => {
  for (const c of fixture.cases) {
    it(c.name, () => {
      const check = CHECKS[c.fn]
      if (check === undefined) throw new Error(`no check for ${c.fn}`)
      check(c)
    })
  }
})

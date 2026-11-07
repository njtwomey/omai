import { describe, expect, it } from 'vitest'
import { toArray, type ComplexValue } from '../../tensor/index.ts'
import * as F from './index.ts'

// FFTs at the lengths where the algorithm changes (1, 2, odd, prime, power of two, 4093, 65537) against numpy.fft rows
// (numpy 2.5, one-off run) and against a direct DFT sum for chosen bins, with round trips and Parseval's identity.
const signal = (n: number): number[] => Array.from({ length: n }, (_, j) => ((j * 7919) % 1009) / 1009 - 0.5)
const parts = (z: ComplexValue): [number[], number[]] => [toArray(z.re) as number[], toArray(z.im) as number[]]

/** The DFT bin k of a real signal by its definition, with the angle taken from (j k mod n) so it is exact for any n. */
function bin(x: readonly number[], k: number): [number, number] {
  const n = x.length
  let re = 0
  let im = 0
  for (let j = 0; j < n; j++) {
    const angle = (2 * Math.PI * ((j * k) % n)) / n
    re += (x[j] ?? 0) * Math.cos(angle)
    im -= (x[j] ?? 0) * Math.sin(angle)
  }
  return [re, im]
}

// numpy.fft.fft(signal(n)) in full for the smallest lengths: [re, im] per bin.
const numpyFull: readonly (readonly [number, readonly (readonly [number, number])[]])[] = [
  [1, [[-0.5, 0]]],
  [
    2,
    [
      [-0.15163528245787905, 0],
      [-0.848364717542121, 0],
    ],
  ],
  [
    3,
    [
      [0.04509415262636274, 0],
      [-0.7725470763131814, -0.1313200067185522],
      [-0.7725470763131814, 0.1313200067185522],
    ],
  ],
  [
    5,
    [
      [-0.016352824578790892, 0],
      [-0.6209117938553023, -0.5217701532014112],
      [-0.6209117938553024, -0.12317322478601264],
      [-0.6209117938553024, 0.12317322478601264],
      [-0.6209117938553023, 0.5217701532014112],
    ],
  ],
]

// numpy.fft.fft(signal(n)) at chosen bins and the sum of |X|^2, for the large prime and power-of-two lengths.
const numpyBins: readonly (readonly [number, number, readonly (readonly [number, number, number])[]])[] = [
  [
    4093,
    1396262.1108617592,
    [
      [0, -1.5099108027750292, 0],
      [1, 0.5141315676553514, -0.030298006752654278],
      [1364, -0.6784260233665855, 1.0505016639067901],
      [2046, -1.276998124132881, 0.2305105297342808],
      [4092, 0.514131567655347, 0.03029800675266534],
    ],
  ],
  [
    65537,
    357925675.75265014,
    [
      [0, -31.823092170465856, 0],
      [1, 0.6531403730793671, -0.0003895757725796711],
      [21845, -0.1384276421776885, -0.5035692565128478],
      [32768, -0.6392359859639882, 0.002676229978759472],
      [65536, 0.6531403730793379, 0.00038957577254722045],
    ],
  ],
  [
    4096,
    1398542.123889946,
    [
      [0, -2.394449950445989, 0],
      [1, -0.369083074959748, -0.031613605472363915],
      [1365, -0.44967391647315413, 0.9231426241793266],
      [2048, -1.4509415262636272, 0],
      [4095, -0.369083074959747, 0.03161360547236505],
    ],
  ],
]

describe('fft at awkward lengths', () => {
  it('matches numpy.fft on the smallest lengths and at chosen bins of 4093, 4096 and 65537, and a direct DFT sum on many lengths', () => {
    const failures: string[] = []
    const near = (label: string, got: number, want: number, atol: number): void => {
      if (!(Math.abs(got - want) <= atol)) failures.push(`${label}: ${got}, expected ${want}`)
    }
    for (const [n, expected] of numpyFull) {
      const [re, im] = parts(F.fft(signal(n)))
      expected.forEach(([r, i], k) => {
        near(`fft n=${n} re[${k}]`, re[k] ?? Number.NaN, r, 1e-14)
        near(`fft n=${n} im[${k}]`, im[k] ?? Number.NaN, i, 1e-14)
      })
    }
    for (const [n, power, bins] of numpyBins) {
      const x = signal(n)
      const [re, im] = parts(F.fft(x))
      for (const [k, r, i] of bins) {
        near(`fft n=${n} re[${k}]`, re[k] ?? Number.NaN, r, 1e-10)
        near(`fft n=${n} im[${k}]`, im[k] ?? Number.NaN, i, 1e-10)
      }
      // Parseval: sum |X_k|^2 = n sum |x_j|^2, which numpy's sum above also satisfies.
      const energy = re.reduce((s, v, k) => s + v * v + (im[k] ?? 0) ** 2, 0)
      if (Math.abs(energy / power - 1) > 1e-12) failures.push(`fft n=${n}: sum |X|^2 ${energy}, numpy ${power}`)
    }
    // Every length against the definition, for the bins 0, 1, n / 3, n - 1, and the round trips through ifft, rfft and irfft.
    for (const n of [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 64, 97, 100, 127, 128, 255, 256, 1000, 1024, 4093,
    ]) {
      const x = signal(n)
      const [re, im] = parts(F.fft(x))
      for (const k of [0, 1, Math.floor(n / 3), n - 1]) {
        if (k >= n) continue
        const [r, i] = bin(x, k)
        near(`fft n=${n} re[${k}]`, re[k] ?? Number.NaN, r, 1e-11)
        near(`fft n=${n} im[${k}]`, im[k] ?? Number.NaN, i, 1e-11)
      }
      const [back] = parts(F.ifft(F.fft(x)))
      back.forEach((v, j) => near(`ifft(fft) n=${n}[${j}]`, v, x[j] ?? 0, 1e-12))
      // The real transform is the first n / 2 + 1 bins, and irfft with the length n recovers odd and even lengths.
      const half = Math.floor(n / 2) + 1
      const [rr, ri] = parts(F.rfft(x))
      if (rr.length !== half) failures.push(`rfft n=${n}: length ${rr.length}, expected ${half}`)
      for (let k = 0; k < Math.min(half, rr.length); k++) {
        near(`rfft n=${n} re[${k}]`, rr[k] ?? Number.NaN, re[k] ?? 0, 1e-11)
        near(`rfft n=${n} im[${k}]`, ri[k] ?? Number.NaN, im[k] ?? 0, 1e-11)
      }
      ;(toArray(F.irfft(F.rfft(x), n)) as number[]).forEach((v, j) =>
        near(`irfft(rfft) n=${n}[${j}]`, v, x[j] ?? 0, 1e-12),
      )
    }
    expect(failures).toEqual([])
  })
})

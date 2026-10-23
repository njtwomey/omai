import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { add, item, mul, stack, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as F from './index.ts'

const pair = (z: F.ComplexValue): Value => stack([z.re, z.im])

describe('Fourier transforms against numpy, scipy.fft and scipy.signal', () => {
  for (const c of loadFixture('numerics/fourier/fourier').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const norm = p['norm'] as F.FftNorm
      const type = p['type'] as F.TrigType
      const z = (): F.ComplexValue => ({ re: t[0] as Tensor, im: t[1] as Tensor })
      const actual: Value =
        c.fn === 'fft'
          ? pair(F.fft(z(), p['n'] as number | undefined, -1, norm))
          : c.fn === 'ifft'
            ? pair(F.ifft(z(), undefined, -1, norm))
            : c.fn === 'fftAxis0'
              ? pair(F.fft(t[0] as Tensor, undefined, 0))
              : c.fn === 'rfft'
                ? pair(F.rfft(t[0] as Tensor))
                : c.fn === 'irfft'
                  ? F.irfft(z(), p['n'] as number)
                  : c.fn === 'hfft'
                    ? F.hfft(t[0] as Tensor, p['n'] as number)
                    : c.fn === 'fftn'
                      ? pair(F.fftn(t[0] as Tensor))
                      : c.fn === 'fftnShape'
                        ? pair(F.fftn(t[0] as Tensor, [5, 8]))
                        : c.fn === 'fftfreq'
                          ? F.fftfreq(9, 0.1)
                          : c.fn === 'rfftfreq'
                            ? F.rfftfreq(10, 0.1)
                            : c.fn === 'fftshift'
                              ? F.fftshift(t[0] as Tensor)
                              : c.fn === 'ifftshift'
                                ? F.ifftshift(t[0] as Tensor)
                                : c.fn === 'nextFastLen'
                                  ? tensor([1, 7, 97, 1000, 1025, 4097].map((v) => F.nextFastLen(v)))
                                  : c.fn === 'nextFastLenReal'
                                    ? tensor([1, 7, 97, 1000, 1025, 4097].map((v) => F.nextFastLen(v, true)))
                                    : c.fn === 'dct'
                                      ? F.dct(t[0] as Tensor, type, { norm })
                                      : c.fn === 'idct'
                                        ? F.idct(t[0] as Tensor, type, { norm })
                                        : c.fn === 'dst'
                                          ? F.dst(t[0] as Tensor, type, { norm })
                                          : c.fn === 'idst'
                                            ? F.idst(t[0] as Tensor, type, { norm })
                                            : c.fn === 'dctAxis0'
                                              ? F.dct(t[0] as Tensor, 2, { axis: 0 })
                                              : c.fn === 'czt'
                                                ? pair(
                                                    F.czt(
                                                      t[0] as Tensor,
                                                      p['m'] as number,
                                                      [p['w_re'] as number, p['w_im'] as number],
                                                      [p['a_re'] as number, p['a_im'] as number],
                                                    ),
                                                  )
                                                : c.fn === 'zoomFft'
                                                  ? pair(F.zoomFft(t[0] as Tensor, [0.2, 0.6], 16, 2))
                                                  : c.fn === 'nudft1'
                                                    ? pair(
                                                        F.nudft1(
                                                          t[0] as Tensor,
                                                          { re: t[1] as Tensor, im: t[2] as Tensor },
                                                          p['m'] as number,
                                                        ),
                                                      )
                                                    : pair(
                                                        F.nudft2(t[0] as Tensor, {
                                                          re: t[1] as Tensor,
                                                          im: t[2] as Tensor,
                                                        }),
                                                      )
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('Fourier properties', () => {
  it('the FFT primitive passes the conformance check for radix-2 and Bluestein lengths', () => {
    for (const n of [8, 7]) {
      const re = Array.from({ length: n }, (_, i) => Math.sin(i + 1))
      const im = Array.from({ length: n }, (_, i) => Math.cos(2 * i))
      const report = checkPrimitive(
        (a, b) => add(F.fft({ re: a, im: b }).re, mul(F.fft({ re: a, im: b }).im, 0.5)),
        [re, im],
      )
      expect(report.failures).toEqual([])
    }
    const report = checkPrimitive((x) => F.irfft(F.rfft(x)), [[0.3, -1, 2, 0.5, 0.7]])
    expect(report.failures).toEqual([])
    expect(checkPrimitive((x) => F.dct(x, 4, { norm: 'ortho' }), [[1, 2, -1, 0.5]]).failures).toEqual([])
  })

  it("Parseval's theorem holds and the gradient of the spectral energy is 2 N x", () => {
    const x = [0.5, -1, 2, 0.25, 3, -0.75]
    const energy = (v: Value): Value => {
      const z = F.fft(v)
      return sum(add(mul(z.re, z.re), mul(z.im, z.im)))
    }
    expect(item(energy(tensor(x)))).toBeCloseTo(6 * x.reduce((s, v) => s + v * v, 0), 10)
    toFlat(grad(energy)(tensor(x)) as Value).forEach((g, i) => {
      expect(g).toBeCloseTo(12 * (x[i] ?? 0), 10)
    })
  })

  it('round trips: ifft(fft), irfft(rfft), idct(dct), idst(dst), imdct(mdct)', () => {
    const x = [1, -2, 0.5, 4, 3, -1, 0.25]
    const back = F.ifft(F.fft(x))
    for (const [i, v] of toFlat(back.re).entries()) expect(v).toBeCloseTo(x[i] ?? 0, 12)
    for (const [i, v] of toFlat(F.irfft(F.rfft(x), 7)).entries()) expect(v).toBeCloseTo(x[i] ?? 0, 12)
    for (const type of [1, 2, 3, 4] as const) {
      for (const [i, v] of toFlat(F.idct(F.dct(x, type), type)).entries()) expect(v).toBeCloseTo(x[i] ?? 0, 11)
      for (const [i, v] of toFlat(F.idst(F.dst(x, type, { norm: 'ortho' }), type, { norm: 'ortho' })).entries())
        expect(v).toBeCloseTo(x[i] ?? 0, 11)
    }
    const signal = Array.from({ length: 24 }, (_, i) => Math.sin(i / 3) + 0.1 * i)
    for (const [i, v] of toFlat(F.imdct(F.mdct(signal, 6))).entries()) expect(v).toBeCloseTo(signal[i] ?? 0, 11)
  })

  it('the NUFFT matches the direct NUDFT', () => {
    const x = Array.from({ length: 30 }, (_, i) => Math.sin(i * 7.3) * 3)
    const c = {
      re: Array.from({ length: 30 }, (_, i) => Math.cos(i)),
      im: Array.from({ length: 30 }, (_, i) => Math.sin(2 * i)),
    }
    for (const sign of [1, -1] as const) {
      const fast = F.nufft1(x, c, 20, { sign })
      const slow = F.nudft1(x, c, 20, sign)
      for (const [i, v] of toFlat(fast.re).entries()) expect(v).toBeCloseTo(toFlat(slow.re)[i] ?? 0, 9)
      const f = {
        re: Array.from({ length: 20 }, (_, i) => 1 / (1 + i)),
        im: Array.from({ length: 20 }, (_, i) => Math.cos(i)),
      }
      const fast2 = F.nufft2(x, f, { sign })
      const slow2 = F.nudft2(x, f, sign)
      for (const [i, v] of toFlat(fast2.im).entries()) expect(v).toBeCloseTo(toFlat(slow2.im)[i] ?? 0, 9)
    }
    expect(F.dft([1, 2, 3, 4]).re).toEqual([10, -2, -2, -2])
    expect(toFlat(F.dftMatrix(4, 'sqrtn').re)[0]).toBe(0.5)
  })
})

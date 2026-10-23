import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as C from './index.ts'

describe('convolution against numpy, scipy.signal and torch', () => {
  for (const c of loadFixture('numerics/convolution/convolution').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const mode = p['mode'] as C.ConvolutionMode
      const patch = {
        stride: p['stride'] as number[],
        dilation: p['dilation'] as number[],
        padding: p['padding'] as number[],
      }
      const actual: Value =
        c.fn === 'convolve'
          ? C.convolve(t[0] as Tensor, t[1] as Tensor, mode)
          : c.fn === 'correlate'
            ? C.correlate(t[0] as Tensor, t[1] as Tensor, mode)
            : c.fn === 'fftConvolve'
              ? C.fftConvolve(t[0] as Tensor, t[1] as Tensor, mode)
              : c.fn === 'oaConvolve'
                ? C.oaConvolve(t[0] as Tensor, t[1] as Tensor, mode)
                : c.fn === 'lags'
                  ? tensor(C.correlationLags(p['n1'] as number, p['n2'] as number, mode))
                  : c.fn === 'pad'
                    ? C.padArray(
                        t[0] as Tensor,
                        [
                          [2, 1],
                          [3, 5],
                        ],
                        p['mode'] as C.PadMode,
                      )
                    : c.fn === 'conv'
                      ? C.convNd(t[0] as Tensor, t[1] as Tensor, {
                          stride: p['stride'] as number | number[],
                          padding: p['padding'] as number | number[],
                          dilation: p['dilation'] as number | number[],
                          groups: p['groups'] as number,
                          ...(t[2] === undefined ? {} : { bias: t[2] }),
                        })
                      : c.fn === 'convSame'
                        ? C.convNd(t[0] as Tensor, t[1] as Tensor, {
                            padding: 'same',
                            dilation: 2,
                            groups: 2,
                            bias: t[2] as Tensor,
                          })
                        : c.fn === 'convReflect'
                          ? C.convNd(t[0] as Tensor, t[1] as Tensor, { padding: 2, paddingMode: 'reflect' })
                          : c.fn === 'convTranspose'
                            ? C.convTranspose(t[0] as Tensor, t[1] as Tensor, {
                                stride: p['stride'] as number | number[],
                                padding: p['padding'] as number | number[],
                                outputPadding: p['outputPadding'] as number,
                                dilation: p['dilation'] as number | number[],
                                groups: p['groups'] as number,
                                ...(t[2] === undefined ? {} : { bias: t[2] }),
                              })
                            : c.fn === 'unfold'
                              ? C.im2col(t[0] as Tensor, p['kernel'] as number[], patch)
                              : C.col2im(t[0] as Tensor, [6, 5], p['kernel'] as number[], patch)
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('convolution properties', () => {
  it('direct, FFT and overlap-add convolution agree and are differentiable', () => {
    const report = checkPrimitive(
      (a, b) => C.convolve(a, b, 'same'),
      [
        [1, 2, -1, 0.5, 3],
        [0.2, -0.4, 1],
      ],
    )
    expect(report.failures).toEqual([])
    expect(
      checkPrimitive(
        (a, b) => C.fftConvolve(a, b),
        [
          [1, 2, -1, 0.5],
          [0.3, 1],
        ],
      ).failures,
    ).toEqual([])
    const x = Array.from({ length: 50 }, (_, i) => Math.sin(i / 3))
    const k = [0.25, 0.5, 0.25]
    const direct = toFlat(C.convolve(x, k))
    for (const [i, v] of toFlat(C.oaConvolve(x, k)).entries()) expect(v).toBeCloseTo(direct[i] ?? 0, 11)
  })

  it('convNd and convTranspose are adjoint, and conv passes the conformance check', () => {
    const x = [
      [
        [1, -2, 0.5, 3, 1],
        [0, 1, 2, -1, 0.5],
      ],
    ]
    const w = [
      [
        [0.5, -1, 2],
        [1, 0.25, -0.5],
      ],
      [
        [1, 1, 1],
        [0, -1, 0],
      ],
    ]
    expect(checkPrimitive((a, b) => C.convNd(a, b, { stride: 2, padding: 1 }), [x, w]).failures).toEqual([])
    // <conv(x, w), y> = <x, convT(y, w)>.
    const yy = [
      [
        [0.3, -0.7, 1.1],
        [0.2, 0.4, -0.6],
      ],
    ]
    const conv = toFlat(C.convNd(x, w, { stride: 2, padding: 1 }))
    const back = toFlat(C.convTranspose(yy, w, { stride: 2, padding: 1 }))
    const lhs = conv.reduce((s, v, i) => s + v * (toFlat(tensor(yy))[i] ?? 0), 0)
    const rhs = back.reduce((s, v, i) => s + v * (toFlat(tensor(x))[i] ?? 0), 0)
    expect(lhs).toBeCloseTo(rhs, 12)
  })

  it('padding modes map the ndimage names', () => {
    expect(C.padModeOf('reflect')).toBe('symmetric')
    expect([-3, -1, 5, 6].map((i) => C.extendIndex(i, 5, 'reflect'))).toEqual([3, 1, 3, 2])
    expect([-1, 5].map((i) => C.extendIndex(i, 5, 'wrap'))).toEqual([4, 0])
  })
})

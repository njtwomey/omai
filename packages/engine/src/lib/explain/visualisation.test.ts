import { describe, expect, it } from 'vitest'
import { conv2d } from '../nn/index.ts'
import { key, normal } from '../random/index.ts'
import { relu, reshape, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { channelObjective, featureVisualisation, totalVariation, unitObjective } from './index.ts'

// A tiny CNN: one 3x3 convolution of a 1-channel image into 3 feature maps, then a ReLU.
const weights = normal(key(11), [3, 1, 3, 3])
const cnn = (x: Value): Value => relu(conv2d(x, weights, { padding: 'same' }))
const shape = [1, 8, 8] as const

const trace = (alg: ReturnType<typeof featureVisualisation>, steps: number): number[] => {
  const out: number[] = []
  let state = alg.init(key(0))
  out.push(state.activation)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, {
      t,
      key: key(t),
      log: () => undefined,
      wants: () => false,
      emit: () => undefined,
      taps: undefined as never,
      run: undefined as never,
    })
    out.push(state.activation)
  }
  return out
}

describe('feature visualisation', () => {
  it('raises the target channel monotonically under plain gradient ascent', () => {
    const alg = featureVisualisation(cnn, {
      shape,
      objective: channelObjective(1),
      parametrisation: 'pixel',
      optimiser: 'sgd',
      learningRate: 10,
      steps: 30,
    })
    const a = trace(alg, 30)
    for (let i = 1; i < a.length; i++) expect(a[i] as number).toBeGreaterThanOrEqual((a[i - 1] as number) - 1e-12)
    expect(a[30]).toBeGreaterThan((a[0] as number) * 1.5)
  })
  it('works in the Fourier basis with the standard transformations', () => {
    const alg = featureVisualisation(cnn, {
      shape,
      objective: channelObjective(2),
      jitter: 2,
      scale: [0.9, 1.1],
      rotate: 10,
      totalVariation: 0.1,
      steps: 40,
    })
    const r = runAlgorithm(alg, { maxSteps: 40, seed: 3 })
    expect(r.status).toBe('completed')
    const first = alg.init(key(3)).activation
    expect(r.state.activation).toBeGreaterThan(first)
    const img = toFlat(r.state.image)
    expect(img.length).toBe(64)
    for (const v of img) expect(v >= 0 && v <= 1).toBe(true)
  })
  it('the total-variation penalty gives a smoother image', () => {
    const base = { shape, objective: unitObjective(0), parametrisation: 'pixel' as const, steps: 40 }
    const model = (x: Value): Value => cnn(x)
    const rough = runAlgorithm(featureVisualisation(model, { ...base, objective: channelObjective(0) }), {
      maxSteps: 40,
      seed: 1,
    }).state
    const smooth = runAlgorithm(
      featureVisualisation(model, { ...base, objective: channelObjective(0), totalVariation: 40 }),
      { maxSteps: 40, seed: 1 },
    ).state
    const tv = (s: typeof rough): number => toFlat(totalVariation(reshape(s.image, [1, 1, 8, 8])))[0] as number
    expect(tv(smooth)).toBeLessThan(tv(rough))
  })
  it('frequency preconditioning favours low frequencies', () => {
    const wide = [1, 16, 16] as const
    // The starting images: noise in a spectrum scaled by 1/f^2 is smooth, noise in the pixels is white.
    const run = (parametrisation: 'fourier' | 'pixel'): number[] =>
      Array.from(
        toFlat(
          featureVisualisation(cnn, { shape: wide, objective: channelObjective(1), parametrisation, decay: 2 }).init(
            key(2),
          ).image,
        ),
      )
    // Neighbouring-pixel differences against the image's own spread: about 1.1 for white noise, small for smooth images.
    const roughness = (img: number[]): number => {
      const m = img.reduce((a, b) => a + b, 0) / img.length
      const sd = Math.sqrt(img.reduce((a, b) => a + (b - m) ** 2, 0) / img.length)
      let s = 0
      let n = 0
      for (let y = 0; y < 16; y++)
        for (let x = 0; x < 15; x++) {
          s += Math.abs((img[y * 16 + x + 1] as number) - (img[y * 16 + x] as number))
          n++
        }
      return s / n / sd
    }
    expect(roughness(run('fourier'))).toBeLessThan(0.6 * roughness(run('pixel')))
  })
  it('serialises and resumes', () => {
    const alg = featureVisualisation(cnn, { shape, objective: channelObjective(0), steps: 6 })
    const mid = runAlgorithm(alg, { maxSteps: 3, seed: 5 })
    const back = alg.deserialise?.(alg.serialise?.(mid.state) as never)
    expect(back?.step).toBe(3)
    expect(back?.activation).toBeCloseTo(mid.state.activation, 12)
    expect(toFlat(back?.image as never)).toEqual(toFlat(mid.state.image))
  })
  it('blur needs the pixel parametrisation', () => {
    expect(() => featureVisualisation(cnn, { shape, blur: { sigma: 1, every: 2 } })).toThrow(/pixel/)
  })
})

/**
 * README example "Bijections between distributions": a Normal pushed through `exp` is the log-normal (`transformed` applies
 * the log-Jacobian, so `logProb` matches the closed form), and a coupling layer bends a 2-D Gaussian into a banana. The
 * figure draws the base, the map, the transformed density against samples, and the banana.
 */
import { bijectors, normal } from 'mdml-engine/dists'
import { transformed } from 'mdml-engine/dists/compose'
import { key } from 'mdml-engine/random'
import { item, mul, square, sub, tensor, toArray, toFlat } from 'mdml-engine/tensor'
import { cartesian, contours, histogram, lines, points } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Bijections between distributions',
  summary:
    'A Normal through exp is a log-normal; a coupling layer turns a Gaussian into a banana. The density stays exact.',
  readme: true,
  alt: 'A Normal, the map exp, the log-normal it gives, and a coupling layer turning a Gaussian into a banana',
}

interface Result {
  readonly base: number[]
  readonly logNormal: { readonly draws: number[]; readonly y: number[]; readonly density: number[] }
  readonly banana: {
    readonly cloud: number[][]
    readonly x: number[]
    readonly y: number[]
    readonly density: number[][]
  }
}

export function run(): Outcome<Result> {
  // exp pushes a Normal onto the positive reals; logProb adds the log-Jacobian for you.
  const lognormal = transformed(normal(0, 1), bijectors.exp())
  const closedForm = (y: number): number => -Math.log(y) - 0.5 * Math.log(2 * Math.PI) - Math.log(y) ** 2 / 2
  const draws = toFlat(lognormal.sample(key(0), [4000]))

  // A coupling layer bends a 2-D Gaussian into a banana; the density is still exact.
  const bend = bijectors.affineCoupling(1, (xa) => ({ shift: sub(mul(0.8, square(xa)), 0.8), logScale: mul(xa, 0) }))
  const banana = transformed(normal([0, 0], [1, 1]), bend)

  const ys = Array.from({ length: 160 }, (_, i) => 0.02 + (i * 7.98) / 159)
  const gx = Array.from({ length: 60 }, (_, i) => -4 + (i * 8) / 59)
  const gy = Array.from({ length: 60 }, (_, i) => -2 + (i * 9) / 59)
  const notes = [
    `logProb at 0.5, 1, 2: ${[0.5, 1, 2].map((y) => `${item(lognormal.logProb(y)).toFixed(6)} (closed form ${closedForm(y).toFixed(6)})`)}`,
    `mean of 4000 draws ${(draws.reduce((a, b) => a + b, 0) / draws.length).toFixed(3)} (exp(1/2) = 1.649)`,
    `banana logProb at (0, -0.8) ${item(banana.logProb(tensor([0, -0.8]))).toFixed(4)}`,
  ]
  const result = {
    base: toFlat(normal(0, 1).sample(key(1), [4000])),
    logNormal: { draws, y: ys, density: ys.map((y) => Math.exp(item(lognormal.logProb(y)))) },
    banana: {
      cloud: toArray(banana.sample(key(2), [250])) as number[][],
      x: gx,
      y: gy,
      density: gy.map((b) => gx.map((a) => Math.exp(item(banana.logProb(tensor([a, b])))))),
    },
  }
  return { result, notes }
}

export function View({ result }: { result: Result }) {
  const grid = Array.from({ length: 121 }, (_, i) => -3.5 + (i * 7) / 120)
  const normalDensity = (x: number): number => Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI)
  const bars = 24
  const barsY = 32
  const left = cartesian(
    [
      histogram(result.base, { bins: bars, range: [-3.5, 3.5], colour: '#9fb4c9' }),
      lines([
        {
          x: grid,
          y: grid.map((x) => normalDensity(x) * result.base.length * (7 / bars)),
          colour: '#222222',
          width: 2,
        },
      ]),
    ],
    { x: { label: 'x' }, y: { label: 'count' }, legend: false },
  )
  const map = cartesian([lines([{ x: grid, y: grid.map(Math.exp), colour: '#d95f02', width: 2 }])], {
    x: { label: 'x' },
    y: { label: 'exp(x)', limits: [0, 8] },
  })
  const { logNormal, banana } = result
  const right = cartesian(
    [
      histogram(logNormal.draws, { bins: barsY, range: [0, 8], colour: '#9fb4c9' }),
      lines([
        {
          x: logNormal.y,
          y: logNormal.density.map((d) => d * logNormal.draws.length * (8 / barsY)),
          colour: '#222222',
          width: 2,
        },
      ]),
    ],
    { x: { label: 'y = exp(x)' }, y: { label: 'count' }, legend: false },
  )
  const bent = cartesian(
    [
      points([
        {
          x: banana.cloud.map((r) => r[0] ?? 0),
          y: banana.cloud.map((r) => r[1] ?? 0),
          colour: '#9fb4c9',
          size: 3,
          opacity: 0.5,
        },
      ]),
      contours(banana.density, { x: banana.x, y: banana.y, levels: 6, labels: false, colour: '#222222', width: 1.5 }),
    ],
    { x: { label: 'y1' }, y: { label: 'y2' } },
  )
  return (
    <Row
      alt={meta.alt}
      width={300}
      height={280}
      panels={[
        { title: 'x ~ Normal(0, 1)', still: left },
        { title: 'the map exp', still: map },
        { title: 'exp(x): the log-normal', still: right },
        { title: 'a coupling layer: a banana', still: bent },
      ]}
    />
  )
}

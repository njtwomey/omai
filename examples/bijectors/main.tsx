/**
 * README example "Bijections between distributions": a Normal pushed through `exp` is the log-normal (`transformed` applies
 * the log-Jacobian, so `logProb` matches the closed form), and a coupling layer bends a 2-D Gaussian into a banana. The
 * figure draws the base, the map, the transformed density against samples, and the banana. Every distribution is drawn
 * on the density scale (a histogram is normalised to a density, the curves are the engine's `logProb` exponentiated), so
 * what changes under the map is the density itself, and the banana is a heat map of its density with contours of the
 * same density.
 */
import { bijectors, normal } from 'mdml-engine/dists'
import { transformed } from 'mdml-engine/dists/compose'
import { key } from 'mdml-engine/random'
import { item, mul, square, sub, tensor, toArray, toFlat } from 'mdml-engine/tensor'
import { cartesian, histogram, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { mix, PAPER, tab } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { fieldLayers } from '../_shared/maps.ts'

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

/** The window of the banana panel (the heat map covers exactly this) and its grid. */
const BANANA = { x: [-3.5, 3.5], y: [-3.5, 6] } as const
const BANANA_GRID = 80

/** The histograms' colour: the first category, lightened towards the page so the density line stays readable over it. */
const tab0Tint = mix(tab(0), PAPER, 0.6)

export function run(): Outcome<Result> {
  // exp pushes a Normal onto the positive reals; logProb adds the log-Jacobian for you.
  const lognormal = transformed(normal(0, 1), bijectors.exp())
  const closedForm = (y: number): number => -Math.log(y) - 0.5 * Math.log(2 * Math.PI) - Math.log(y) ** 2 / 2
  const draws = toFlat(lognormal.sample(key(0), [4000]))

  // A coupling layer bends a 2-D Gaussian into a banana; the density is still exact.
  const bend = bijectors.affineCoupling(1, (xa) => ({ shift: sub(mul(0.8, square(xa)), 0.8), logScale: mul(xa, 0) }))
  const banana = transformed(normal([0, 0], [1, 1]), bend)

  const ys = Array.from({ length: 160 }, (_, i) => 0.02 + (i * 7.98) / 159)
  const gx = Array.from(
    { length: BANANA_GRID },
    (_, i) => BANANA.x[0] + (i * (BANANA.x[1] - BANANA.x[0])) / (BANANA_GRID - 1),
  )
  const gy = Array.from(
    { length: BANANA_GRID },
    (_, i) => BANANA.y[0] + (i * (BANANA.y[1] - BANANA.y[0])) / (BANANA_GRID - 1),
  )
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

/** The area under a curve by the trapezoid rule. */
const area = (x: readonly number[], y: readonly number[]): number =>
  x.reduce((acc, xi, i) => (i === 0 ? 0 : acc + ((xi - (x[i - 1] ?? 0)) * ((y[i] ?? 0) + (y[i - 1] ?? 0))) / 2), 0)

export function View({ result }: { result: Result }) {
  const grid = Array.from({ length: 121 }, (_, i) => -3.5 + (i * 7) / 120)
  const normalDensity = (x: number): number => Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI)
  const { logNormal, banana } = result
  // The log-normal density by the change of variables: p_Y(y) = p_X(log y) / |dy/dx| with dy/dx = y.
  const lognormalDensity = logNormal.y.map((y) => normalDensity(Math.log(y)) / y)
  // The histogram is a density (counts / (n * bin width)), so its area is the share of the draws inside the axes.
  const share = (draws: readonly number[], lo: number, hi: number): number =>
    draws.filter((d) => d >= lo && d <= hi).length / draws.length
  const two = (v: number): string => v.toFixed(2)
  const frame = figure({ width: 440, height: 340, padding: { top: 30, right: 14, bottom: 38, left: 50 } })
  const wide = figure({ width: 520, height: 340, padding: { top: 30, right: 14, bottom: 38, left: 50 } })
  const left = cartesian(
    [
      histogram(result.base, {
        bins: 28,
        range: [-3.5, 3.5],
        density: true,
        colour: tab0Tint,
        name: `draws, area ${two(share(result.base, -3.5, 3.5))}`,
      }),
      lines([
        {
          x: grid,
          y: grid.map(normalDensity),
          colour: 'fg',
          width: 2,
          name: `p(x), area ${two(area(grid, grid.map(normalDensity)))}`,
        },
      ]),
    ],
    {
      figure: frame,
      x: { label: 'x', limits: [-3.5, 3.5] },
      y: { label: 'density p(x)', limits: [0, 0.5] },
      legend: true,
    },
  )
  const map = cartesian([lines([{ x: grid, y: grid.map(Math.exp), colour: 'tab3', width: 2 }])], {
    figure: wide,
    x: { label: 'x', limits: [-3.5, 3.5] },
    y: { label: 'y = exp(x)', limits: [0, 8] },
    legend: false,
  })
  const right = cartesian(
    [
      histogram(logNormal.draws, {
        bins: 40,
        range: [0, 8],
        density: true,
        colour: tab0Tint,
        name: `draws, area ${two(share(logNormal.draws, 0, 8))}`,
      }),
      lines([
        {
          x: logNormal.y,
          y: lognormalDensity,
          colour: 'fg',
          width: 2,
          name: `p(y), area ${two(area(logNormal.y, lognormalDensity))}`,
        },
      ]),
    ],
    {
      figure: frame,
      x: { label: 'y = exp(x)', limits: [0, 8] },
      y: { label: 'density p(y)', limits: [0, 0.8] },
      legend: true,
    },
  )
  const window = { x: BANANA.x, y: BANANA.y }
  const bent = cartesian(
    [
      ...fieldLayers({
        window,
        field: banana.density,
        label: 'density',
        levels: 8,
        range: [0, Math.max(...banana.density.flat())],
      }),
      points([
        {
          x: banana.cloud.map((r) => r[0] ?? 0),
          y: banana.cloud.map((r) => r[1] ?? 0),
          name: '250 samples',
          colour: 'tab3',
          size: 3,
          opacity: 0.6,
          outline: false,
        },
      ]),
    ],
    {
      figure: figure({ width: 520, height: 340, padding: { top: 30, right: 40, bottom: 38, left: 50 } }),
      x: { label: 'y1', limits: [...window.x] },
      y: { label: 'y2', limits: [...window.y] },
      legend: true,
    },
  )
  return (
    <Figure width={960} height={732} title={meta.alt}>
      <Panel x={0} width={440} height={340} title="x ~ Normal(0, 1): the density p(x)" still={left} />
      <Panel x={440} width={520} height={340} title="the map y = exp(x), with dy/dx = exp(x)" still={map} />
      <g transform="translate(0 366)">
        <Panel x={0} width={440} height={340} title="y = exp(x): p(y) = p(x) / |dy/dx|" still={right} />
        <Panel x={440} width={520} height={340} title="a coupling layer: the density of the banana" still={bent} />
      </g>
    </Figure>
  )
}

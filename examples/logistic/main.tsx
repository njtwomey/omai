/**
 * README example "Bayesian logistic regression, approximated four ways": one probabilistic program, inferred by NUTS (the
 * reference), the Laplace approximation, mean-field variational inference and full-rank variational inference. The
 * dataset knows the true class boundary. The figure draws the weight posteriors and the predicted probability from the
 * posteriors the engines return; the tape is the full-rank variational run (its ELBO).
 */
import { hessian } from 'mdml-engine/autodiff'
import { generate } from 'mdml-engine/data'
import { blobs } from 'mdml-engine/datasets'
import { bernoulli, multivariateNormal } from 'mdml-engine/dists'
import { inv } from 'mdml-engine/numerics/linalg'
import { ndtr } from 'mdml-engine/numerics/special'
import { lbfgs } from 'mdml-engine/optim'
import {
  autoMultivariateNormal,
  laplaceApproximation,
  mcmc,
  observe,
  type Posterior,
  plate,
  potentialEnergy,
  sample,
  svi,
} from 'mdml-engine/probprog'
import { key } from 'mdml-engine/random'
import { eye, matmul, mul, neg, tensor, toArray, toFlat, type Value } from 'mdml-engine/tensor'
import { recorder, tracePolicy } from 'mdml-engine/trace'
import { cartesian, contours, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, classMap, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Bayesian logistic regression, approximated four ways',
  summary:
    'The weight posterior by NUTS, Laplace and two variational fits, and the predicted probability beside the true boundary.',
  readme: true,
  alt: 'The weight posterior of a Bayesian logistic regression by NUTS, Laplace and two variational fits, and the predicted probability with the true boundary',
}

interface Result {
  readonly rows: number[][]
  readonly labels: number[]
  readonly nuts: number[][]
  readonly ellipses: { readonly name: string; readonly colour: string; readonly sample: number[][] }[]
  /** On the grid of `axesOf(windowOf(rows), GRID)`: the probability of class 1 from the NUTS draws, and the true one. */
  readonly probability: number[][]
  readonly trueProbability: number[][]
}

const GRID = 60

const mean = (p: Posterior) =>
  p
    .summary()
    .map((r) => `${r.mean.toFixed(2)} +- ${r.sd.toFixed(2)}`)
    .join(' | ')

export function run(): Outcome<Result> {
  const { data, truth } = generate(blobs, { knobs: { n: 60, centers: 2, clusterStd: 2.2, boxSize: 3 }, key: key(4) })
  const rows = toArray(data.x) as number[][]
  const y = Array.from(toFlat(data.y))
  const X = tensor(rows.map(([a, b]) => [1, a ?? 0, b ?? 0])) // a column of ones for the intercept

  const prior = multivariateNormal([0, 0, 0], { covariance: mul(eye(3), 25) })
  const model = (labels: number[]) => {
    const w = sample('w', prior)
    plate('n', labels.length, () => observe('y', bernoulli({ logits: matmul(X, w) }), labels))
  }

  const nuts = mcmc(model, y, { key: key(3), numChains: 4, numWarmup: 500, numDraws: 1000 })
  const laplace = laplaceApproximation(model, y, { key: key(1) })
  const meanField = svi(model, y, { key: key(2), numSteps: 2000 }).posterior
  const tape = recorder() // the full-rank run's ELBO, every 50 steps
  const fullRank = svi(model, y, {
    key: key(2),
    numSteps: 2000,
    guide: autoMultivariateNormal(model),
    tracer: tape,
    policy: tracePolicy({ schedule: { every: 50 } }),
  }).posterior

  // The Laplace covariance by hand: the MAP of the model as an objective, and the inverse Hessian there.
  const um = potentialEnergy(model, y)
  const map = lbfgs((u) => neg(um.fn(u as Value)), tensor([0, 0, 0]), {})
  const covariance = toArray(inv(hessian((u) => neg(um.fn(u)))(map.x as Value))) as number[][]

  // A probit link is a one-line change.
  const probit = (labels: number[]) => {
    const w = sample('w', prior)
    plate('n', labels.length, () => observe('y', bernoulli({ probs: ndtr(matmul(X, w)) }), labels))
  }
  const notes = [
    `NUTS ${mean(nuts)}; R-hat ${nuts.summary().map((r) => r.rhat?.toFixed(3))}`,
    `MAP ${toFlat(map.x as Value).map((v) => v.toFixed(3))}; Laplace sd ${covariance.map((r, i) => Math.sqrt(r[i] ?? 0).toFixed(3))}`,
    `Laplace ${mean(laplace)}`,
    `mean-field ${mean(meanField)}`,
    `full-rank ${mean(fullRank)}`,
    `probit NUTS ${mean(mcmc(probit, y, { key: key(3), numChains: 4, numWarmup: 500, numDraws: 1000 }))}`,
    `probit Laplace ${mean(laplaceApproximation(probit, y, { key: key(1) }))}`,
  ]

  // What the figure draws: draws of each posterior, and the true class probability on a grid of the plane.
  const draws = (p: Posterior, k: number): number[][] => toArray(p.sample(key(k), 4000)['w'] as Value) as number[][]
  if (truth.posterior === undefined) throw new Error('blobs carries its class posterior')
  const cells = gridRows(axesOf(windowOf(rows), GRID))
  const classes = toArray(truth.posterior(tensor(cells))) as number[][]
  // The predicted probability: the NUTS draws pushed through the logistic function over the plane, averaged.
  const keep = draws(nuts, 5).filter((_, i) => i % 10 === 0)
  const probability = cells.map(([a = 0, b = 0]) => {
    const total = keep.reduce(
      (acc, w) => acc + 1 / (1 + Math.exp(-((w[0] ?? 0) + (w[1] ?? 0) * a + (w[2] ?? 0) * b))),
      0,
    )
    return total / keep.length
  })
  const result: Result = {
    rows,
    labels: y,
    nuts: draws(nuts, 5),
    ellipses: [
      { name: 'NUTS', colour: '#222222', sample: draws(nuts, 5) },
      { name: 'Laplace', colour: '#d62728', sample: draws(laplace, 6) },
      { name: 'mean-field VI', colour: '#9467bd', sample: draws(meanField, 7) },
      { name: 'full-rank VI', colour: '#2ca02c', sample: draws(fullRank, 8) },
    ],
    probability: gridOfValues(probability, GRID),
    trueProbability: gridOfValues(
      classes.map((r) => r[1] ?? 0),
      GRID,
    ),
  }
  return { result, tapes: { 'full-rank-vi': tape.run() }, notes }
}

/** The 95% ellipse of the two slope weights of some draws. */
function ellipse(draws: number[][]): { x: number[]; y: number[] } {
  const n = draws.length
  const m = [0, 1].map((j) => draws.reduce((a, r) => a + (r[j + 1] ?? 0), 0) / n)
  const c = (i: number, j: number): number =>
    draws.reduce((a, r) => a + ((r[i + 1] ?? 0) - (m[i] ?? 0)) * ((r[j + 1] ?? 0) - (m[j] ?? 0)), 0) / (n - 1)
  const l11 = Math.sqrt(c(0, 0))
  const l21 = c(0, 1) / l11
  const l22 = Math.sqrt(c(1, 1) - l21 * l21)
  const r = Math.sqrt(5.991)
  const t = Array.from({ length: 81 }, (_, i) => (i / 80) * 2 * Math.PI)
  return {
    x: t.map((u) => (m[0] ?? 0) + r * l11 * Math.cos(u)),
    y: t.map((u) => (m[1] ?? 0) + r * (l21 * Math.cos(u) + l22 * Math.sin(u))),
  }
}

export function View({ result }: { result: Result }) {
  const thin = result.nuts.filter((_, i) => i % 16 === 0)
  const curves = result.ellipses.map((e) => ({ ...ellipse(e.sample), name: e.name, colour: e.colour, width: 2 }))
  const window = windowOf(
    [...thin.map((r) => [r[1] ?? 0, r[2] ?? 0]), ...curves.flatMap((c) => c.x.map((x, i) => [x, c.y[i] ?? 0]))],
    0.04,
  )
  const left = cartesian(
    [
      points([
        {
          x: thin.map((r) => r[1] ?? 0),
          y: thin.map((r) => r[2] ?? 0),
          name: 'NUTS draws',
          colour: '#8a94a0',
          size: 3,
          opacity: 0.4,
          outline: false,
        },
      ]),
      lines(curves),
    ],
    {
      figure: figure({ width: 470, height: 400, padding: { top: 30, right: 12, bottom: 38, left: 46 } }),
      x: { label: 'weight of feature 1', limits: [...window.x] },
      y: { label: 'weight of feature 2', limits: [...window.y] },
      legend: true,
    },
  )

  const mapWindow = windowOf(result.rows)
  const axes = axesOf(mapWindow, GRID)
  const right = classMap({
    window: mapWindow,
    field: result.probability,
    centre: 0.5,
    range: [0, 1],
    label: 'P(class 1)',
    boundary: 0.5,
    dashed: [0.05, 0.25, 0.75, 0.95],
    extra: [
      contours(result.trueProbability, {
        x: axes.x,
        y: axes.y,
        levels: [0.5],
        labels: false,
        colour: 'fg',
        width: 2,
        dash: 'dashed',
        name: 'true boundary',
      }),
    ],
    rows: result.rows,
    labels: result.labels,
    classNames: ['class 0', 'class 1'],
    width: 530,
    height: 400,
    xLabel: 'feature 1',
    yLabel: 'feature 2',
  })
  return (
    <Figure width={1000} height={426} title={meta.alt}>
      <Panel x={0} width={470} height={400} title="posterior of the two slope weights: 95% regions" still={left} />
      <Panel
        x={470}
        width={530}
        height={400}
        title="P(class 1) from NUTS: contours at 5, 25, 50, 75 and 95%; true boundary dashed"
        still={right}
      />
    </Figure>
  )
}

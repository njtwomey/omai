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
import { cartesian, contours, heatmap, lines, points } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

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
  readonly grid: number[]
  readonly trueProbability: number[][]
}

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
  const grid = Array.from({ length: 41 }, (_, i) => -8 + (i * 16) / 40)
  if (truth.posterior === undefined) throw new Error('blobs carries its class posterior')
  const classes = toArray(truth.posterior(tensor(grid.flatMap((b) => grid.map((a) => [a, b]))))) as number[][]
  const result: Result = {
    rows,
    labels: y,
    nuts: draws(nuts, 5),
    ellipses: [
      { name: 'NUTS', colour: '#222222', sample: draws(nuts, 5) },
      { name: 'Laplace', colour: '#d95f02', sample: draws(laplace, 6) },
      { name: 'mean-field VI', colour: '#7570b3', sample: draws(meanField, 7) },
      { name: 'full-rank VI', colour: '#1b9e77', sample: draws(fullRank, 8) },
    ],
    grid,
    trueProbability: grid.map((_, i) => grid.map((__, j) => classes[i * grid.length + j]?.[1] ?? 0)),
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
        },
      ]),
      lines(result.ellipses.map((e) => ({ ...ellipse(e.sample), name: e.name, colour: e.colour, width: 2 }))),
    ],
    { x: { label: 'weight of feature 1' }, y: { label: 'weight of feature 2' }, legend: true },
  )

  // The predicted probability: the NUTS draws pushed through the logistic function over the plane.
  const keep = result.nuts.filter((_, i) => i % 10 === 0)
  const probability = result.grid.map((b) =>
    result.grid.map(
      (a) =>
        keep.reduce((acc, w) => acc + 1 / (1 + Math.exp(-((w[0] ?? 0) + (w[1] ?? 0) * a + (w[2] ?? 0) * b))), 0) /
        keep.length,
    ),
  )
  const right = cartesian(
    [
      heatmap(probability, {
        x: result.grid,
        y: result.grid,
        smooth: 'contours',
        levels: 10,
        ramp: 'rdbu',
        range: [0, 1],
        label: 'P(class 1)',
        opacity: 0.8,
      }),
      contours(result.trueProbability, {
        x: result.grid,
        y: result.grid,
        levels: [0.5],
        labels: false,
        colour: 'fg',
        width: 2.5,
        dash: 'dashed',
        name: 'true boundary',
      }),
      points([
        {
          x: result.rows.map((r) => r[0] ?? 0),
          y: result.rows.map((r) => r[1] ?? 0),
          groups: result.labels,
          groupNames: { 0: 'class 0', 1: 'class 1' },
          size: 6,
        },
      ]),
    ],
    { x: { label: 'feature 1' }, y: { label: 'feature 2' } },
  )
  return (
    <Row
      alt={meta.alt}
      width={420}
      height={360}
      panels={[
        { title: 'the weights: 95% regions', still: left },
        { title: 'predicted probability (NUTS) and the true boundary', still: right },
      ]}
    />
  )
}

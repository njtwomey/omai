/**
 * Example "A sampler on a banana": the No-U-Turn sampler (NUTS) on Haario's banana, a twisted Gaussian whose density is
 * known exactly. Four chains warm up (the step size is tuned) and then draw; the figure is made from the chains' own
 * records: the draws over the true density contours, and the trace of one coordinate. The convergence diagnostics (R-hat
 * and effective sample size) are computed from the same draws.
 */
import { banana, type Chains, nuts, parameterDraws, summary, thin } from 'mdml-engine/sampling'
import { key } from 'mdml-engine/random'
import { toFlat } from 'mdml-engine/tensor'
import { type Run, recorder, tracePolicy } from 'mdml-engine/trace'
import { cartesian, contours, lines, points } from 'mdml-render/charts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A sampler on a banana',
  summary:
    'NUTS on a twisted Gaussian: the draws over the true density, the trace of one coordinate, R-hat and effective sample size.',
  readme: false,
  alt: 'Draws of four NUTS chains over the contours of the banana density, and the trace of the first coordinate',
}

const target = banana({ curvature: 0.1, scale: 3 })

interface Result {
  readonly chains: Chains
  readonly rhat: number
  readonly ess: number
}

export function run(): Outcome<Result> {
  const tape = recorder()
  const chains = nuts(target, {
    key: key(0),
    numChains: 4,
    numWarmup: 500,
    numDraws: 500,
    tracer: tape,
    policy: tracePolicy({ metrics: ['logDensity', 'treeDepth'], schedule: { every: 10 } }),
  })
  const [first, second] = summary(chains)
  const notes = [
    `x[0]: mean ${first?.mean.toFixed(2)} sd ${first?.sd.toFixed(2)} (true 0 and 3), R-hat ${first?.rhat.toFixed(3)}, bulk ESS ${first?.essBulk.toFixed(0)}`,
    `x[1]: mean ${second?.mean.toFixed(2)} sd ${second?.sd.toFixed(2)} (true 0 and 1.62), R-hat ${second?.rhat.toFixed(3)}, bulk ESS ${second?.essBulk.toFixed(0)}`,
  ]
  const run: Run = tape.run()
  return {
    result: { chains, rhat: first?.rhat ?? Number.NaN, ess: first?.essBulk ?? Number.NaN },
    tapes: { run },
    notes,
  }
}

const GRID = 70
const X = [-10, 10] as const
const Y = [-4.5, 9] as const

/** The draws of one chain for one coordinate. */
const coordinate = (c: Chains, chain: number, dim: number): number[] =>
  toFlat(parameterDraws(c, dim)).slice(chain * c.numDraws, (chain + 1) * c.numDraws)

export function View({ result }: { result: Result }) {
  const { chains } = result
  const xs = Array.from({ length: GRID }, (_, i) => X[0] + ((X[1] - X[0]) * i) / (GRID - 1))
  const ys = Array.from({ length: GRID }, (_, j) => Y[0] + ((Y[1] - Y[0]) * j) / (GRID - 1))
  const density = ys.map((y) => xs.map((x) => Math.exp(target.logDensity([x, y]))))
  const peak = Math.max(...density.flat())
  const sparse = thin(chains, 10) // a tenth of the draws, so the scatter stays readable
  const chainIds = Array.from({ length: chains.numChains }, (_, i) => i)
  const surface = contours(density, {
    x: xs,
    y: ys,
    levels: [0.011, 0.135, 0.607].map((r) => r * peak), // where a Gaussian's density is 3, 2 and 1 sd out
    labels: false,
    opacity: 0.5,
  })
  const samples = points(
    chainIds.map((i) => ({
      x: coordinate(sparse, i, 0),
      y: coordinate(sparse, i, 1),
      name: `chain ${i + 1}`,
      size: 5,
    })),
  )
  const trace = lines(chainIds.map((i) => ({ y: coordinate(chains, i, 0), name: `chain ${i + 1}`, width: 1 })))
  return (
    <Row
      alt={meta.alt}
      width={470}
      height={340}
      panels={[
        {
          title: 'draws over the true density',
          still: cartesian([surface, samples], { x: { label: 'x[0]' }, y: { label: 'x[1]' }, legend: true }),
        },
        {
          title: `trace of x[0]: R-hat ${result.rhat.toFixed(3)}, effective sample size ${result.ess.toFixed(0)}`,
          still: cartesian([trace], { x: { label: 'draw' }, y: { label: 'x[0]' }, legend: true }),
        },
      ]}
    />
  )
}

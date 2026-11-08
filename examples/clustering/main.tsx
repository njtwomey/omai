/**
 * Example "A Gaussian mixture, step by step" (in the index, not in the README): expectation-maximisation for a mixture
 * of three Gaussians on overlapping blobs, recorded step by step. Everything drawn is read off the recording: the
 * log likelihood at each kept step, and the weights, means, covariances and responsibilities of the last kept step.
 * Left, the density the mixture gives the plane (a heat map and its contours, on the density scale) with each row
 * coloured by the component most responsible for it; right, the log likelihood climbing.
 */
import { generate } from 'omai-engine/data'
import { blobs } from 'omai-engine/datasets'
import { adjustedRandScore } from 'omai-engine/metrics'
import { componentLabels, gaussianEmAlgorithm, initialMixture, mixturePosterior } from 'omai-engine/mixture'
import { key } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { iterate, metricSeries, policies, type Run, recorder, runAlgorithm, tensorSeries } from 'omai-engine/trace'
import { cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, densityMap, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A Gaussian mixture, step by step',
  summary:
    'Expectation-maximisation for three overlapping Gaussians: the density, the log likelihood and the soft assignments.',
  readme: false,
  home: { label: 'Mixture', seconds: 9 },
  alt: 'A Gaussian mixture fitted by EM: the density it gives the plane with the rows coloured by component, and the log likelihood over the steps',
}

interface Result {
  readonly rows: number[][]
  /** The mixture after each kept step (the first is the starting point): its density on a grid over the data, and the component most responsible for each row. */
  readonly snapshots: readonly { readonly density: number[][]; readonly component: number[] }[]
  /** The mean log likelihood at each kept step. */
  readonly steps: number[]
  readonly logLikelihood: number[]
}

const GRID = 60

/** The values of a tensor channel at the `i`th step it was kept, cut into rows of `width`. */
function rowsAt(run: Run, name: string, i: number, width: number): number[][] {
  const flat = tensorSeries(run, name)[i]?.values ?? []
  return Array.from({ length: flat.length / width }, (_, j) => flat.slice(j * width, (j + 1) * width))
}

export function run(): Outcome<Result> {
  const { data } = generate(blobs, { knobs: { n: 100, centers: 3, clusterStd: 1.3, boxSize: 6 }, key: key(2) })
  const start = initialMixture(data.x, { nComponents: 3, covarianceType: 'full', init: 'random' }, key(0))
  const tape = recorder()
  // Every step is kept (it is a short run); the metrics and tensors share the schedule.
  const thin = {
    ...policies.teaching,
    roles: ['param', 'state'] as const,
    messages: false,
    frames: false,
    schedule: { every: 1 },
  }
  const fit = runAlgorithm(gaussianEmAlgorithm(data.x, start, { tol: 1e-5 }), {
    seed: 0,
    maxSteps: 100,
    tracer: tape,
    policy: thin,
  })
  const run = tape.run()

  const rows = toArray(data.x) as number[][]
  const grid = gridRows(axesOf(windowOf(rows), GRID))
  // The mixture at each kept step, as the recording holds it.
  const kept = tensorSeries(run, 'means').length
  const snapshots = Array.from({ length: kept }, (_, i) => {
    const means = rowsAt(run, 'means', i, 2)
    const k = means.length
    const mixture = {
      weights: rowsAt(run, 'weights', i, k)[0] ?? [],
      means,
      covariances: rowsAt(run, 'covariances', i, 4).map((r) => [
        [r[0] ?? 0, r[1] ?? 0],
        [r[2] ?? 0, r[3] ?? 0],
      ]),
    }
    const resp = rowsAt(run, 'responsibilities', i, k)
    return {
      density: gridOfValues(
        mixturePosterior(grid, mixture).logNorm.map((v) => Math.exp(v)),
        GRID,
      ),
      component: componentLabels(resp),
      resp,
    }
  })
  const logged = metricSeries(run, 'logLikelihood')
  const ari = adjustedRandScore(data.y, componentLabels(fit.state.resp)).estimate.value
  const confidence = (snapshots.at(-1)?.resp ?? []).map((r) => Math.max(...r))
  const notes = [
    `${fit.steps} steps, ${fit.status}; mean log likelihood ${logged.values[0]?.toFixed(3)} at the first kept step, ${fit.state.logLikelihood.toFixed(3)} at the end`,
    `adjusted Rand index against the true blobs ${ari.toFixed(3)}; ${confidence.filter((c) => c < 0.9).length} of ${rows.length} rows have no component above 0.9; ${kept} kept steps`,
  ]
  return {
    result: {
      rows,
      snapshots: snapshots.map(({ density, component }) => ({ density, component })),
      steps: logged.steps,
      logLikelihood: logged.values,
    },
    tapes: { run },
    notes,
  }
}

/**
 * Fitted live, as it is watched: each step is one step of the same EM algorithm `run` uses (taken from `iterate`, which
 * yields every state), and the mixture it holds then is read off as the snapshot to draw.
 */
export function live(): Live<Result> {
  const { data } = generate(blobs, { knobs: { n: 100, centers: 3, clusterStd: 1.3, boxSize: 6 }, key: key(2) })
  const start = initialMixture(data.x, { nComponents: 3, covarianceType: 'full', init: 'random' }, key(0))
  const rows = toArray(data.x) as number[][]
  const grid = gridRows(axesOf(windowOf(rows), GRID))
  const states = iterate(gaussianEmAlgorithm(data.x, start, { tol: 1e-5 }), { seed: 0, maxSteps: 100 })
  const snapshots: { density: number[][]; component: number[] }[] = []
  const steps: number[] = []
  const logLikelihood: number[] = []
  const run: Live<Result> = {
    step: () => {
      const next = states.next()
      if (next.done === true) return false
      const { t, state, status } = next.value
      snapshots.push({
        density: gridOfValues(
          mixturePosterior(grid, state).logNorm.map((v) => Math.exp(v)),
          GRID,
        ),
        component: componentLabels(state.resp),
      })
      steps.push(t)
      logLikelihood.push(state.logLikelihood)
      return status === undefined
    },
    result: () => ({ rows, snapshots, steps, logLikelihood }),
  }
  run.step() // the first step, so there is always something to draw
  return run
}

/** Played in the browser: one frame per kept step of the EM run. */
export const frames = (result: Result): number => result.snapshots.length - 1

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const last = result.snapshots.length - 1
  const i = at === undefined ? last : Math.min(at, last)
  const now = result.snapshots[i]
  if (now === undefined) throw new Error('clustering: no step to draw')
  const window = windowOf(result.rows)
  const left = densityMap({
    window,
    field: now.density,
    label: 'density',
    rows: result.rows,
    groups: now.component,
    names: ['component 0', 'component 1', 'component 2'],
    width: 400,
    height: 340,
    xLabel: 'x0',
    yLabel: 'x1',
    mode,
  })
  const small = figure({ width: 300, height: 340, padding: { top: 30, right: 14, bottom: 38, left: 56 } })
  const middle = cartesian(
    [
      lines([
        {
          x: result.steps.slice(0, i + 1),
          y: result.logLikelihood.slice(0, i + 1),
          name: 'log likelihood',
          markers: true,
        },
      ]),
    ],
    {
      mode,
      figure: small,
      x: { label: 'EM step', limits: [0, Math.max(...result.steps)] },
      y: {
        label: 'mean log likelihood',
        limits: [Math.min(...result.logLikelihood), Math.max(...result.logLikelihood)],
      },
      legend: false,
    },
  )
  return (
    <Figure width={720} height={366} title={meta.alt} mode={mode}>
      <Panel x={0} width={420} height={340} title="density and most responsible component" still={left} />
      <Panel x={420} width={300} height={340} title="log likelihood" still={middle} />
    </Figure>
  )
}

/**
 * README example "Pause, jump back, compare with the truth": a support vector classifier run as a session. It is
 * paused after three steps, run to the end, and asked for its decision surface at step 3 and at the end; the figure is
 * those two records. The dataset carries its truth, so the best possible accuracy is known.
 */
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import type { Exposition } from 'omai-engine/exposition'
import { key } from 'omai-engine/random'
import { svc } from 'omai-engine/svm'
import { stillOf } from 'omai-render/views'
import { INK } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Pause, jump back, compare with the truth',
  summary:
    'A support vector classifier as a session: its decision surface at step 3 and at the end, and the best possible accuracy.',
  readme: true,
  alt: 'The decision surface of a support vector classifier at step 3 and at the end',
}

interface Result {
  readonly early: Exposition
  readonly final: Exposition
  readonly steps: number
  readonly earlyAccuracy: number
  readonly finalAccuracy: number
  readonly best: number
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 120, noise: 0.25 }, key: key(0) })
  const run = svc.with({ kernel: 'rbf', C: 1 }).session(data, { key: key(1) })
  run.advance({ steps: 3 }) // pause after three steps of the solver
  run.advance() // run on to the end
  const early = run.probe(3, 'model/decision-surface', { resolution: 50 }) as Exposition // jump back to step 3
  const final = run.probe(run.t, 'model/decision-surface', { resolution: 50 }) as Exposition
  const notes = [
    `accuracy at step 3 ${run.fittedAt(3).score(data).toFixed(3)}, at the end ${run.fittedAt(run.t).score(data).toFixed(3)}`,
    `best possible ${(1 - (truth.bayesError ?? 0)).toFixed(3)}; ${run.t} steps, ${run.status}`,
  ]
  return {
    result: {
      early,
      final,
      steps: run.t,
      earlyAccuracy: run.fittedAt(3).score(data),
      finalAccuracy: run.fittedAt(run.t).score(data),
      best: 1 - (truth.bayesError ?? 0),
    },
    tapes: { session: run },
    notes,
  }
}

export function View({ result }: { result: Result }) {
  return (
    <Figure width={780} height={410} title={meta.alt}>
      <Panel
        x={0}
        width={390}
        height={340}
        title={`after 3 steps: accuracy ${result.earlyAccuracy.toFixed(2)}`}
        still={stillOf(result.early, { mode: 'light' })}
      />
      <Panel
        x={390}
        width={390}
        height={340}
        title={`after all ${result.steps} steps: accuracy ${result.finalAccuracy.toFixed(2)}`}
        still={stillOf(result.final, { mode: 'light' })}
      />
      <text x={390} y={394} textAnchor="middle" fontFamily="system-ui, sans-serif" fontSize={13} fill={INK}>
        The same run, stopped and then asked for its earlier state. The best accuracy anyone could reach on this data is{' '}
        {result.best.toFixed(2)}.
      </text>
    </Figure>
  )
}

/**
 * README example "Pause, jump back, compare with the truth": a support vector classifier run as a session. It is
 * paused after three steps, run to the end, and asked for its decision surface at step 3 and at the end; the figure is
 * those two records. The dataset carries its truth, so the best possible accuracy is known.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import type { Exposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { svc } from 'mdml-engine/svm'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

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
  return { result: { early, final, steps: run.t }, tapes: { session: run }, notes }
}

export function View({ result }: { result: Result }) {
  return (
    <Row
      alt={meta.alt}
      width={380}
      height={340}
      panels={[
        { title: 'step 3', still: stillOf(result.early, { mode: 'light' }) },
        { title: `step ${result.steps} (the end)`, still: stillOf(result.final, { mode: 'light' }) },
      ]}
    />
  )
}

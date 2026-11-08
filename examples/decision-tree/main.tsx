/**
 * Example "A decision tree and the regions it cuts" (in the index, not in the README): a CART classifier grown one node
 * per step on two-moons data. The left panel is the tree as the session's `model/structure` record (each node its split,
 * rows and class proportions); the right one is the `model/decision-surface` record of the same fit, so every leaf is
 * a rectangle of the plane and the boundary is a staircase. The data carries its truth, so the best possible accuracy
 * is known.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import type { Exposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A decision tree and the regions it cuts',
  summary: 'A classification tree drawn as its splits and as the rectangles it makes of the plane.',
  readme: false,
  alt: 'A decision tree on the two-moons data, drawn as a tree of splits beside the regions of the plane it makes',
}

interface Result {
  readonly tree: Exposition
  readonly regions: Exposition
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 150, noise: 0.3 }, key: key(0) })
  const test = generate(moons, { knobs: { n: 400, noise: 0.3 }, key: key(9) }).data
  // Five candidate splits kept per feature and node on the tape (the default keeps all of them).
  const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data, { key: key(1) })
  fit.advance() // one node decided per step
  const tree = fit.probe(fit.t, 'model/structure') as Exposition
  const regions = fit.probe(fit.t, 'model/decision-surface', { resolution: 80 }) as Exposition
  const model = fit.fittedAt(fit.t)
  const notes = [
    `${fit.t} steps, ${fit.status}; accuracy ${model.score(data).toFixed(3)} on the training rows, ${model.score(test).toFixed(3)} on 400 new ones`,
    `best possible ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`,
  ]
  return { result: { tree, regions }, tapes: { session: fit }, notes }
}

export function View({ result }: { result: Result }) {
  return (
    <Figure width={1160} height={426} title={meta.alt}>
      <Panel x={0} width={708} height={400} title="the tree" still={stillOf(result.tree, { mode: 'light' })} />
      <Panel x={720} width={440} height={400} title="the regions" still={stillOf(result.regions, { mode: 'light' })} />
    </Figure>
  )
}

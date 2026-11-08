/**
 * Example "A decision tree and the regions it cuts": a CART classifier grown one node
 * per step on two measurements of Fisher's iris flowers. Both pictures are the library's own: the left one is the
 * session's `model/structure` record drawn by the render package's tree view, and the right one its
 * `model/decision-surface` record, where every leaf of the tree is a rectangle of the plane.
 */
import { iris } from 'omai-engine/datasets'
import type { Exposition } from 'omai-engine/exposition'
import { key } from 'omai-engine/random'
import { tensor, toArray, toFlat } from 'omai-engine/tensor'
import { decisionTreeClassifier } from 'omai-engine/trees'
import { stillOf } from 'omai-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A decision tree and the regions it cuts',
  summary:
    'A classification tree on two iris measurements, drawn by the library as its splits and as the regions of the plane.',
  readme: true,
  alt: 'A decision tree on iris petal length and width, drawn as a tree of splits beside the regions of the plane it makes',
}

interface Result {
  readonly tree: Exposition
  readonly regions: Exposition
}

export function run(): Outcome<Result> {
  const all = iris()
  const wanted = [2, 3] // petal length and petal width
  const rows = (toArray(all.x) as number[][]).map((r) => wanted.map((j) => r[j] ?? 0))
  const data = {
    x: tensor(rows),
    y: all.y,
    featureNames: wanted.map((j) => all.featureNames?.[j] ?? `x${j}`),
    classes: all.classes,
  }
  const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data as never, { key: key(1) })
  fit.advance() // one node decided per step
  const tree = fit.probe(fit.t, 'model/structure') as Exposition
  const regions = fit.probe(fit.t, 'model/decision-surface', { resolution: 120 }) as Exposition
  const labels = Array.from(toFlat(all.y))
  const predicted = Array.from(toFlat(fit.fittedAt(fit.t).predict(data.x)))
  const right = predicted.filter((p, i) => p === labels[i]).length
  const notes = [`${fit.t} steps, ${fit.status}; ${right} of ${labels.length} flowers classified correctly`]
  return { result: { tree, regions }, tapes: { session: fit }, notes }
}

export function View({ result }: { result: Result }) {
  return (
    <Figure width={1200} height={456} title={meta.alt}>
      <Panel x={0} width={700} height={430} title="the tree" still={stillOf(result.tree, { mode: 'light' })} />
      <Panel x={710} width={490} height={430} title="the regions" still={stillOf(result.regions, { mode: 'light' })} />
    </Figure>
  )
}

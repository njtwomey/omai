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
import type { Mode } from '../_shared/colours.ts'
import type { ExampleModule, Live, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A decision tree and the regions it cuts',
  summary:
    'A classification tree on two iris measurements, drawn by the library as its splits and as the regions of the plane.',
  readme: true,
  alt: 'A decision tree on iris petal length and width, drawn as a tree of splits beside the regions of the plane it makes',
  home: { label: 'Decision tree', fps: 3, seconds: 7 },
}

interface Result {
  /** The tree and its regions as the session had them after each step (index 0: before any split). */
  readonly trees: readonly Exposition[]
  readonly regions: readonly Exposition[]
}

/** The iris flowers with only petal length and petal width, as the classifier takes them. */
function flowers() {
  const all = iris()
  const wanted = [2, 3] // petal length and petal width
  const rows = (toArray(all.x) as number[][]).map((r) => wanted.map((j) => r[j] ?? 0))
  return {
    all,
    data: {
      x: tensor(rows),
      y: all.y,
      featureNames: wanted.map((j) => all.featureNames?.[j] ?? `x${j}`),
      classes: all.classes,
    },
  }
}

export function run(): Outcome<Result> {
  const { all, data } = flowers()
  const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data as never, { key: key(1) })
  fit.advance() // one node decided per step
  const steps = Array.from({ length: fit.t + 1 }, (_, i) => i)
  const trees = steps.map((t) => fit.probe(t, 'model/structure') as Exposition)
  const regions = steps.map((t) => fit.probe(t, 'model/decision-surface', { resolution: 120 }) as Exposition)
  const labels = Array.from(toFlat(all.y))
  const predicted = Array.from(toFlat(fit.fittedAt(fit.t).predict(data.x)))
  const right = predicted.filter((p, i) => p === labels[i]).length
  const notes = [`${fit.t} steps, ${fit.status}; ${right} of ${labels.length} flowers classified correctly`]
  return { result: { trees, regions }, tapes: { session: fit }, notes }
}

/**
 * Grown live, as it is watched: each step is one node of the same session `run` uses, and the tree and its regions are read
 * back from the session after it.
 */
export function live(): Live<Result> {
  const { data } = flowers()
  const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data as never, { key: key(1) })
  const trees: Exposition[] = []
  const regions: Exposition[] = []
  const snapshot = (): void => {
    trees.push(fit.probe(fit.t, 'model/structure') as Exposition)
    regions.push(fit.probe(fit.t, 'model/decision-surface', { resolution: 120 }) as Exposition)
  }
  snapshot()
  return {
    step: () => {
      fit.advance({ steps: 1 })
      snapshot()
      return fit.status === undefined
    },
    result: () => ({ trees, regions }),
  }
}

/** Played in the browser: one frame per step of the session (one node decided in each). */
export const frames = (result: Result): number => result.trees.length - 1

export function View({ result, at, mode = 'light' }: { result: Result; at?: number; mode?: Mode }) {
  const i = at === undefined ? result.trees.length - 1 : Math.min(at, result.trees.length - 1)
  const tree = result.trees[i]
  const regions = result.regions[i]
  if (tree === undefined || regions === undefined) throw new Error('decision-tree: no step to draw')
  return (
    <Figure width={1200} height={456} title={meta.alt} mode={mode}>
      <Panel
        x={0}
        width={700}
        height={430}
        title={at === undefined ? 'the tree' : `the tree after ${i} steps`}
        still={stillOf(tree, { mode })}
      />
      <Panel x={710} width={490} height={430} title="the regions" still={stillOf(regions, { mode })} />
    </Figure>
  )
}

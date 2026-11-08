/**
 * Example "A decision tree and the regions it cuts" (in the index, not in the README): a CART classifier grown one node
 * per step on two-moons data. The left panel is the tree, the session's `model/structure` record: each node its split,
 * its rows and its class proportions. The right panel is the same fitted tree asked for the probability of class 1 at
 * every point of the plane: a heat map (orange towards class 1, blue towards class 0) that is constant on each leaf's
 * rectangle, with the decision boundary drawn as the contour at probability 0.5 of that same map.
 */
import { generate } from 'mdml-engine/data'
import { moons } from 'mdml-engine/datasets'
import type { Exposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { toArray, toFlat } from 'mdml-engine/tensor'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { axesOf, classMap, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'A decision tree and the regions it cuts',
  summary: 'A classification tree drawn as its splits and as the probability map and boundary it makes of the plane.',
  readme: false,
  alt: 'A decision tree on the two-moons data, drawn as a tree of splits beside the probability of class 1 over the plane with the 0.5 contour as the boundary',
}

interface Result {
  readonly tree: Exposition
  readonly rows: number[][]
  readonly labels: number[]
  readonly probability: number[][]
}

const GRID = 70

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 150, noise: 0.3 }, key: key(0) })
  const test = generate(moons, { knobs: { n: 400, noise: 0.3 }, key: key(9) }).data
  // Five candidate splits kept per feature and node on the tape (the default keeps all of them).
  const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data, { key: key(1) })
  fit.advance() // one node decided per step
  const tree = fit.probe(fit.t, 'model/structure') as Exposition
  const model = fit.fittedAt(fit.t)
  const accuracy = (d: typeof data): number => {
    const guess = toFlat(model.predict(d.x))
    const truth = toFlat(d.y)
    return guess.filter((g, i) => g === truth[i]).length / guess.length
  }
  const rows = toArray(data.x) as number[][]
  const grid = gridRows(axesOf(windowOf(rows), GRID))
  const p = toArray(model.predictProba?.(grid) as never) as number[][]
  const probability = gridOfValues(
    p.map((r) => r[1] ?? 0),
    GRID,
  )
  const notes = [
    `${fit.t} steps, ${fit.status}; accuracy ${accuracy(data).toFixed(3)} on the training rows, ${accuracy(test).toFixed(3)} on 400 new ones`,
    `best possible ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`,
  ]
  return { result: { tree, rows, labels: Array.from(toFlat(data.y)), probability }, tapes: { session: fit }, notes }
}

export function View({ result }: { result: Result }) {
  const map = classMap({
    window: windowOf(result.rows),
    field: result.probability,
    cells: true,
    centre: 0.5,
    range: [0, 1],
    label: 'P(class 1)',
    boundary: 0.5,
    rows: result.rows,
    labels: result.labels,
    classNames: ['class 0', 'class 1'],
    width: 440,
    height: 350,
    xLabel: 'x0',
    yLabel: 'x1',
  })
  return (
    <Figure width={1148} height={376} title={meta.alt}>
      <Panel x={0} width={708} height={350} title="the tree" still={stillOf(result.tree, { mode: 'light' })} />
      <Panel x={708} width={440} height={350} title="probability of class 1 and the 0.5 boundary" still={map} />
    </Figure>
  )
}

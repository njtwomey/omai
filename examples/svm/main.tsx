/**
 * Example "Support vectors and the margin": a support vector classifier with a
 * Gaussian kernel fitted twice to the same two-moons data, with a small and a large penalty C. Each panel draws one
 * quantity, the score f(x) the model gives every point of the plane: as a heat map (orange where f is positive, class 1;
 * blue where it is negative), with f = 0 as the decision boundary (solid) and f = -1 and f = 1 as the margins (dashed).
 * The rows ringed are the support vectors, taken from the session's `model/support-vectors` record.
 */
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import type { MarginRecord } from 'omai-engine/exposition'
import { key } from 'omai-engine/random'
import { svc } from 'omai-engine/svm'
import { toArray, toFlat } from 'omai-engine/tensor'
import type { ExampleModule, Outcome, Tape } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'
import { axesOf, classMap, gridOfValues, gridRows, windowOf } from '../_shared/maps.ts'

export const meta: ExampleModule['meta'] = {
  title: 'Support vectors and the margin',
  summary:
    'A kernel support vector classifier with a small and a large C: the score, the boundary, the margins and the support vectors.',
  readme: true,
  alt: 'Two support vector classifiers on the two-moons data: the score as a heat map with the boundary and margins as contours, and the support vectors ringed',
}

interface Panel {
  readonly title: string
  readonly rows: number[][]
  readonly labels: number[]
  readonly score: number[][]
  readonly support: readonly number[]
}
interface Result {
  readonly panels: readonly Panel[]
}

const GRID = 60

export function run(): Outcome<Result> {
  const { data, truth } = generate(moons, { knobs: { n: 100, noise: 0.25 }, key: key(0) })
  const test = generate(moons, { knobs: { n: 400, noise: 0.25 }, key: key(9) }).data
  const rows = toArray(data.x) as number[][]
  const labels = Array.from(toFlat(data.y))
  const axes = axesOf(windowOf(rows), GRID)
  const panels: Panel[] = []
  const tapes: Record<string, Tape> = {}
  const notes = [`best possible accuracy ${(1 - (truth.bayesError ?? 0)).toFixed(3)}`]
  for (const [name, C] of [
    ['small-c', 0.2],
    ['large-c', 50],
  ] as const) {
    const fit = svc.with({ kernel: 'rbf', C }).session(data, { key: key(1) })
    fit.advance()
    const model = fit.fittedAt(fit.t)
    const support = (fit.probe(fit.t, 'model/support-vectors') as MarginRecord).support
    const score = gridOfValues(toFlat(model.decisionFunction(gridRows(axes))), GRID)
    panels.push({ title: `C = ${C}: ${support.length} support vectors`, rows, labels, score, support })
    tapes[name] = fit
    const inside = (fit.probe(fit.t, 'model/support-vectors') as MarginRecord).margins.filter(
      (m) => m < 1 - 1e-6,
    ).length
    notes.push(
      `C = ${C}: ${fit.t} steps, ${support.length} support vectors (${inside} inside the margin), accuracy ${model.score(data).toFixed(3)} on the training rows, ${model.score(test).toFixed(3)} on 400 new ones`,
    )
  }
  return { result: { panels }, tapes, notes }
}

export function View({ result }: { result: Result }) {
  const window = windowOf(result.panels[0]?.rows ?? [])
  return (
    <Row
      alt={meta.alt}
      width={440}
      height={380}
      panels={result.panels.map((p) => ({
        title: p.title,
        still: classMap({
          window,
          field: p.score,
          centre: 0,
          range: [-2, 2],
          label: 'score f(x)',
          boundary: 0,
          dashed: [-1, 1],
          rows: p.rows,
          labels: p.labels,
          classNames: ['class 0', 'class 1'],
          ringed: p.support,
          ringName: 'support vector',
          width: 440,
          height: 380,
          xLabel: 'x0',
          yLabel: 'x1',
        }),
      }))}
    />
  )
}

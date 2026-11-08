/**
 * Example "Handwritten digits in two dimensions" (in the index, not in the README): each digit is an 8 by 8 picture, so
 * a point in 64 dimensions. Three methods place the same 600 digits on a plane: PCA (a linear projection), t-SNE and UMAP
 * (both keep neighbours together). Each point is coloured by the digit it really is. The t-SNE run is a session (its
 * tape is kept). The numbers measure the layouts: how many of a point's 5 nearest neighbours in the plane show the same
 * digit.
 */
import { digits } from 'mdml-engine/datasets'
import { pca } from 'mdml-engine/decomposition'
import { tsne, umap } from 'mdml-engine/manifold'
import { key } from 'mdml-engine/random'
import { tensor, toArray, toFlat } from 'mdml-engine/tensor'
import { session } from 'mdml-engine/trace'
import { cartesian, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { categoricalColour } from 'mdml-render/theme'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Handwritten digits in two dimensions',
  summary:
    '600 handwritten digits, each 64 numbers, placed on a plane by PCA, t-SNE and UMAP and coloured by the digit.',
  readme: false,
  alt: 'The same 600 handwritten digits placed on a plane by PCA, t-SNE and UMAP, each point coloured by its digit',
}

const PER_DIGIT = 60
const STEPS = 300

interface Layout {
  readonly name: string
  readonly rows: number[][]
  /** The share of each point's 5 nearest neighbours in this plane that are the same digit. */
  readonly agreement: number
}

interface Result {
  readonly labels: number[]
  readonly layouts: readonly Layout[]
}

/** The share of each point's `k` nearest neighbours in a layout that carry the same label as the point. */
function agreement(layout: readonly (readonly number[])[], labels: readonly number[], k = 5): number {
  let same = 0
  layout.forEach((p, i) => {
    const near = layout
      .map((q, j) => ({ j, d: Math.hypot(...p.map((v, c) => v - (q[c] ?? 0))) }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, k)
    same += near.filter((e) => labels[e.j] === labels[i]).length / k
  })
  return same / layout.length
}

export function run(): Outcome<Result> {
  const all = digits()
  const x = toArray(all.x) as number[][]
  const y = Array.from(toFlat(all.y))
  // The first PER_DIGIT examples of each digit.
  const chosen = Array.from({ length: 10 }, (_, c) =>
    y.flatMap((l, i) => (l === c ? [i] : [])).slice(0, PER_DIGIT),
  ).flat()
  const labels = chosen.map((i) => y[i] ?? 0)
  const data = {
    x: tensor(chosen.map((i) => x[i] ?? [])),
    y: tensor(labels),
    classes: all.classes,
    featureNames: all.featureNames,
  }

  const linear = toArray(
    pca
      .with({ nComponents: 2 })
      .fit(data as never)
      .transform(data.x),
  ) as number[][]
  // t-SNE as a session built from the method, so a checkpoint is kept every 100 steps and the tape stays small.
  const method = tsne.with({ perplexity: 30, maxIter: STEPS, exaggerationIter: 100 }).method(data as never)
  const fit = session(method, { key: key(1), steps: STEPS, checkpointEvery: 100 })
  fit.advance()
  const neighbours = fit.probe(fit.t, 'model/embedding') as number[][]
  const manifold = umap.with({ nNeighbors: 15, nEpochs: 200 }).fit(data as never).params.embedding as number[][]

  const layouts: Layout[] = [
    { name: 'PCA', rows: linear, agreement: agreement(linear, labels) },
    { name: 't-SNE', rows: neighbours, agreement: agreement(neighbours, labels) },
    { name: 'UMAP', rows: manifold, agreement: agreement(manifold, labels) },
  ]
  const notes = [
    `${labels.length} digits of 64 pixels each; ${fit.t} t-SNE steps, ${fit.status}`,
    `share of the 5 nearest neighbours that show the same digit: ${layouts.map((l) => `${l.name} ${l.agreement.toFixed(3)}`).join(', ')}`,
  ]
  return { result: { labels, layouts }, tapes: { session: fit }, notes }
}

const DIGIT_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const PANEL = { width: 380, height: 360 }
const PAD = { top: 12, right: 12, bottom: 38, left: 48 }

export function View({ result }: { result: Result }) {
  const scatter = (layout: Layout): unknown =>
    cartesian(
      Array.from({ length: 10 }, (_, c) => {
        const idx = result.labels.flatMap((l, i) => (l === c ? [i] : []))
        return points([
          {
            x: idx.map((i) => layout.rows[i]?.[0] ?? 0),
            y: idx.map((i) => layout.rows[i]?.[1] ?? 0),
            name: String(c),
            colour: `tab${c}`,
            size: 7,
            opacity: 0.85,
            outline: false,
          },
        ])
      }),
      {
        figure: figure({ ...PANEL, padding: PAD }),
        x: { label: 'first coordinate' },
        y: { label: 'second coordinate' },
        legend: false,
      },
    )
  const heading = 56
  return (
    <Figure width={3 * PANEL.width} height={heading + PANEL.height + 60} title={meta.alt}>
      <text x={14} y={22} fontFamily="system-ui, sans-serif" fontSize={14} fontWeight={600} fill="#333333">
        The same 600 handwritten digits (64 pixels each) on a plane, three ways, coloured by the digit
      </text>
      <g transform="translate(0, 30)">
        {result.layouts.map((layout, k) => (
          <Panel
            key={layout.name}
            x={k * PANEL.width}
            width={PANEL.width}
            height={PANEL.height}
            title={`${layout.name}: ${(100 * layout.agreement).toFixed(0)}% of neighbours are the same digit`}
            still={scatter(layout) as never}
          />
        ))}
      </g>
      {DIGIT_NAMES.map((name, c) => (
        <g key={name} transform={`translate(${160 + c * 85}, ${heading + PANEL.height + 46})`}>
          <circle r={6} fill={categoricalColour(c)} />
          <text x={12} y={5} fontSize={13} fontFamily="system-ui, sans-serif" fill="#333333">
            digit {name}
          </text>
        </g>
      ))}
    </Figure>
  )
}

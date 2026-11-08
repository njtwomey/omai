/**
 * Example "A rolled-up sheet, flattened" (in the index, not in the README): a Swiss roll is a flat sheet rolled up in
 * three dimensions, and the generator keeps each point's position along the sheet. PCA can only look at the roll from
 * one side, so layers of the roll land on top of each other; t-SNE, fitted as a session and read back at ten of its
 * steps, keeps points that are close on the sheet close in the plane. Both panels colour each point by its true
 * position along the sheet, so a good layout is a smooth gradient; the printed numbers measure it.
 */
import { generate } from 'mdml-engine/data'
import { swissRoll } from 'mdml-engine/datasets'
import { pca } from 'mdml-engine/decomposition'
import { type EmbeddingSteps, type Exposition, embeddingSteps } from 'mdml-engine/exposition'
import { tsne } from 'mdml-engine/manifold'
import { key } from 'mdml-engine/random'
import { toArray, toFlat } from 'mdml-engine/tensor'
import { session } from 'mdml-engine/trace'
import { stillOf } from 'mdml-render/views'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Row } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A rolled-up sheet, flattened',
  summary: 'PCA and t-SNE on a Swiss roll, each point coloured by its true position along the sheet.',
  readme: false,
  alt: 'A Swiss roll in two dimensions by PCA, where layers of the roll overlap, and by t-SNE, where points close on the sheet stay close',
}

interface Result {
  readonly linear: EmbeddingSteps
  readonly nonlinear: EmbeddingSteps
}

/** How far apart along the sheet (in the true coordinate t) the 5 nearest neighbours of a point are, on average. */
function neighbourGap(embedding: readonly (readonly number[])[], along: readonly number[]): number {
  let total = 0
  embedding.forEach((p, i) => {
    const near = embedding
      .map((q, j) => ({ j, d: Math.hypot((p[0] ?? 0) - (q[0] ?? 0), (p[1] ?? 0) - (q[1] ?? 0)) }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, 5)
    total += near.reduce((s, e) => s + Math.abs((along[e.j] ?? 0) - (along[i] ?? 0)), 0) / 5
  })
  return total / embedding.length
}

export function run(): Outcome<Result> {
  const { data, truth } = generate(swissRoll, { knobs: { n: 120 }, key: key(0) })
  const along = Array.from(toFlat(truth.latents?.['t'] as never)) // the true position along the sheet
  const colourBy = { kind: 'values', values: along, name: 'position along the roll' } as const

  const linear = pca.with({ nComponents: 2 }).fit(data)
  const flat = toArray(linear.transform(data.x)) as number[][]

  // The session is made from the method so that a checkpoint is kept every 100 steps (the tape stays small).
  const method = tsne.with({ perplexity: 30, maxIter: 300, exaggerationIter: 100 }).method(data)
  const fit = session(method, { key: key(1), steps: 300, checkpointEvery: 100 })
  fit.advance()
  const at = Array.from({ length: 10 }, (_, i) => Math.round(((i + 1) * fit.t) / 10))
  const nonlinear = embeddingSteps({
    method: 'tsne',
    iterations: at,
    positions: at.map((t) => fit.probe(t, 'model/embedding') as number[][]),
    objective: at.map((t) => (fit.probe(t, 'model/params') as { objective: number }).objective),
    colourBy,
  })
  const kl = nonlinear.objective ?? []
  const notes = [
    `${fit.t} t-SNE steps, ${fit.status}; KL divergence ${kl[0]?.toFixed(3)} after ${at[0]}, ${kl.at(-1)?.toFixed(3)} at the end`,
    `mean distance along the sheet between a point and its 5 nearest neighbours in the plane: PCA ${neighbourGap(flat, along).toFixed(2)}, t-SNE ${neighbourGap(nonlinear.positions.at(-1) ?? [], along).toFixed(2)} (the sheet spans t from ${Math.min(...along).toFixed(1)} to ${Math.max(...along).toFixed(1)})`,
  ]
  return {
    result: { linear: embeddingSteps({ method: 'pca', iterations: [0], positions: [flat], colourBy }), nonlinear },
    tapes: { session: fit },
    notes,
  }
}

export function View({ result }: { result: Result }) {
  return (
    <Row
      alt={meta.alt}
      width={460}
      height={400}
      panels={[
        { title: 'PCA, two components', still: stillOf(result.linear as Exposition, { mode: 'light' }) },
        { title: 't-SNE, 300 steps', still: stillOf(result.nonlinear as Exposition, { mode: 'light' }) },
      ]}
    />
  )
}

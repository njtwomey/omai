import { type ReactNode, useMemo, useState } from 'react'
import { iris } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { standardScaler } from 'whitebox-engine/learn/preprocessing'
import { tsne, umap } from 'whitebox-engine/methods/manifold'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, EmbeddingPlayer, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Embedding steps',
  question: 'How do I play back a nonlinear embedding as it is optimised?',
  explain:
    't-SNE and UMAP return an `embeddingSteps` record: the embedding every few iterations with the objective (the KL divergence, or the fuzzy cross-entropy). `EmbeddingPlayer` plays it, coloured by species: early exaggeration pulls the clusters apart, then they settle. The perplexity (or the number of neighbours) sets how local the picture is.',
}

// region
const raw = iris()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }
const labels = Array.from(raw.y.data, (c) => raw.classes[c] ?? String(c))

type Method = 't-SNE' | 'UMAP'

export default function EmbeddingStepsRecipe(): ReactNode {
  const [method, setMethod] = useState<Method>('t-SNE')
  const [locality, setLocality] = useState(20)
  const steps = useMemo(() => {
    const est =
      method === 't-SNE'
        ? tsne({ perplexity: locality, maxIter: 500, recordEvery: 10 })
        : umap({ nNeighbors: Math.round(locality), nEpochs: 200, recordEvery: 5 })
    return recordsOf(est.fit(data, { key: key(0) }).exposition, 'embeddingSteps')[0]
  }, [method, locality])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="method" value={method} onChange={setMethod} options={['t-SNE', 'UMAP'] as Method[]} />
        <NumberField
          label={method === 't-SNE' ? 'perplexity' : 'neighbours'}
          value={locality}
          onChange={setLocality}
          min={2}
          max={50}
          integer
        />
      </ControlPanel>
      {steps !== undefined && <EmbeddingPlayer record={steps} labels={labels} />}
    </div>
  )
}
// endregion

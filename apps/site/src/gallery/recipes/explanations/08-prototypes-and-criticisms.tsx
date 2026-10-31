import { mmdCritic } from 'mdml-engine/explain'
import { prototypeSet } from 'mdml-engine/exposition'
import { tensor } from 'mdml-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { blobs, normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Prototypes and criticisms',
  question: 'How do I summarise a dataset by a few typical examples and the ones they fail to represent?',
  explain:
    'A `prototypes` record names the rows chosen to represent the data (MMD-critic’s prototypes) and the rows they represent worst (criticisms), with the witness function, positive where prototypes over-represent the data and negative where they under-represent it. `PrototypeView` plots the rows with both marked, and beside it the witness. Three blobs and a few strays here: with as many prototypes as blobs each gets one, and the strays become the criticisms.',
  shows: ['prototypes', 'PrototypeView'],
}

export default function PrototypesRecipe(): ReactNode {
  const [nPrototypes, setPrototypes] = useState(3)
  const [nCriticisms, setCriticisms] = useState(4)
  const [seed, setSeed] = useState(2)
  const record = useMemo(() => {
    const { points, labels } = blobs(seed, 3, 40, 0.5)
    const strays = normals(seed + 7, 12, 4)
    const rows = [...points, ...[0, 2, 4, 6, 8].map((i) => [strays[i] ?? 0, strays[i + 1] ?? 0])]
    const r = mmdCritic(tensor(rows), { nPrototypes, nCriticisms, gamma: 0.4 })
    return prototypeSet({
      method: 'MMD-critic',
      features: ['x', 'y'],
      prototypes: [...r.prototypes],
      criticisms: [...r.criticisms],
      prototypeData: r.prototypes.map((i) => rows[i] ?? []),
      criticismData: r.criticisms.map((i) => rows[i] ?? []),
      witness: [...r.witness],
      points: rows.map((p) => [p[0] ?? 0, p[1] ?? 0] as [number, number]),
      axes: ['x', 'y'],
      labels: [...labels.map((l) => `blob ${l}`), ...new Array<string>(rows.length - labels.length).fill('stray')],
    })
  }, [nPrototypes, nCriticisms, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="prototypes" value={nPrototypes} onChange={setPrototypes} min={1} max={10} integer />
        <NumberField label="criticisms" value={nCriticisms} onChange={setCriticisms} min={0} max={10} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Typical rows and the ones they miss" />
    </div>
  )
}

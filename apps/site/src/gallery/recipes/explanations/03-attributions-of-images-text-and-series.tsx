import { featureAttributions } from 'omai-engine/exposition'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Attributions of images, text and series',
  question: 'How do I show attributions when the features are pixels, words or time steps?',
  explain:
    'The record’s `layout` says what its features are, and `ExpositionView` picks the view: an `image` layout draws the input beside a diverging attribution map (`AttributionImageView`), `tokens` highlight a sentence word by word (`TokenStripView`), `series` colour the signal and bar its steps (`AttributionSeriesView`). Each toy model is linear, so its attributions are exact: weight times input. Move the image’s object, the sentence’s negation or the series’ spike and the red and blue move with it.',
  shows: ['featureAttributions', 'AttributionImageView', 'TokenStripView', 'AttributionSeriesView'],
}

type Modality = 'image' | 'text' | 'series'

const SIDE = 10
// Weights of the image model: it looks for a bright centre, and dislikes a bright border.
const WEIGHTS = Array.from({ length: SIDE * SIDE }, (_, i) => {
  const r = Math.floor(i / SIDE) - (SIDE - 1) / 2
  const c = (i % SIDE) - (SIDE - 1) / 2
  return 1.5 * Math.exp(-(r * r + c * c) / 6) - 0.4 * (Math.abs(r) > 3.5 || Math.abs(c) > 3.5 ? 1 : 0)
})

const LEXICON: Readonly<Record<string, number>> = {
  good: 1.2,
  great: 1.8,
  fine: 0.4,
  bad: -1.4,
  awful: -2,
  slow: -0.8,
  love: 1.6,
}

function imageRecord(shift: number, seed: number) {
  const noise = normals(seed, SIDE * SIDE, 0.08)
  const x = Array.from({ length: SIDE * SIDE }, (_, i) => {
    const r = Math.floor(i / SIDE) - (SIDE - 1) / 2 - shift * 0.4
    const c = (i % SIDE) - (SIDE - 1) / 2 + shift * 0.4
    return Math.max(0, Math.exp(-(r * r + c * c) / 3) + (noise[i] ?? 0))
  })
  const values = x.map((v, i) => (WEIGHTS[i] ?? 0) * v)
  const total = values.reduce((a, b) => a + b, 0)
  return featureAttributions({
    method: 'linear',
    features: x.map((_, i) => `pixel ${i}`),
    output: 'score',
    baseValue: 0,
    values: [values],
    data: [x],
    predictions: [total],
    layout: { kind: 'image', shape: [SIDE, SIDE] },
  })
}

function textRecord(negated: boolean) {
  const words = (
    negated
      ? 'the film was not good but the acting was great and slow'
      : 'the film was good but the acting was great and slow'
  ).split(' ')
  const values = words.map((w, i) => {
    const base = LEXICON[w] ?? 0
    return words[i - 1] === 'not' ? -base : base
  })
  const text = words.join(' ')
  let at = 0
  const spans = words.map((w): [number, number] => {
    const start = text.indexOf(w, at)
    at = start + w.length
    return [start, at]
  })
  return featureAttributions({
    method: 'linear',
    features: words,
    output: 'sentiment',
    baseValue: 0,
    values: [values],
    data: [words.map(() => 1)],
    predictions: [values.reduce((a, b) => a + b, 0)],
    layout: { kind: 'tokens', text, spans },
  })
}

function seriesRecord(spike: number, seed: number) {
  const n = 40
  const noise = normals(seed, n, 0.15)
  const x = Array.from(
    { length: n },
    (_, t) => Math.sin(t / 4) + (noise[t] ?? 0) + 2.5 * Math.exp(-((t - spike) ** 2) / 4),
  )
  // The model reads the recent past more than the distant one, and a spike counts for much.
  const values = x.map((v, t) => (0.02 + (t / n) * 0.12) * v * (v > 1.8 ? 3 : 1))
  return featureAttributions({
    method: 'linear',
    features: x.map((_, t) => `t=${t}`),
    output: 'forecast',
    baseValue: 0,
    values: [values],
    data: [x],
    predictions: [values.reduce((a, b) => a + b, 0)],
    layout: { kind: 'series', time: x.map((_, t) => t) },
  })
}

export default function AttributionsOfImagesTextAndSeries(): ReactNode {
  const [modality, setModality] = useState<Modality>('image')
  const [seed, setSeed] = useState(4)
  const [shift, setShift] = useState(0)
  const [negated, setNegated] = useState<'plain' | 'negated'>('plain')
  const [spike, setSpike] = useState(30)
  const record = useMemo(
    () =>
      modality === 'image'
        ? imageRecord(shift, seed)
        : modality === 'text'
          ? textRecord(negated === 'negated')
          : seriesRecord(spike, seed),
    [modality, shift, seed, negated, spike],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice
          label="features are"
          value={modality}
          options={['image', 'text', 'series'] as const}
          onChange={setModality}
        />
        {modality !== 'text' && <SeedInput value={seed} onChange={setSeed} />}
        {modality === 'image' && (
          <NumberField label="object off centre" value={shift} onChange={setShift} min={-4} max={4} step={0.5} />
        )}
        {modality === 'text' && (
          <Choice label="sentence" value={negated} options={['plain', 'negated'] as const} onChange={setNegated} />
        )}
        {modality === 'series' && (
          <NumberField label="spike at" value={spike} onChange={setSpike} min={2} max={38} integer />
        )}
      </ControlPanel>
      <ExpositionView record={record} />
    </div>
  )
}

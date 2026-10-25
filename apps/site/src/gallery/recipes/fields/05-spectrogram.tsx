import { spectrogram } from 'mdml-engine/signal'
import { toArray } from 'mdml-engine/tensor'
import { cartesian, heatmap } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Spectrogram',
  question: "How do I show how a signal's frequencies change over time?",
  explain:
    "A spectrogram (`spectrogram(x, { fs, nperseg })` in the engine's signal domain) is the power at each frequency in each short window; drawn as a `heatmap` with time across and frequency up. The signal is a chirp sweeping up through the frequencies plus a steady tone. Longer windows sharpen frequency and blur time, the uncertainty trade-off.",
}

const FIGURE = figure({ width: 820, height: 420 })

export default function Spectrogram(): ReactNode {
  const mode = useColourMode()
  const [window, setWindow] = useState(128)
  const [sweep, setSweep] = useState(150)
  const [db, setDb] = useState(true)
  const option = useMemo(() => {
    const fs = 1000
    const n = 4000
    const x = Array.from({ length: n }, (_, i) => {
      const t = i / fs
      return Math.sin(2 * Math.PI * (20 * t + (sweep / 2) * t * t)) + 0.5 * Math.sin(2 * Math.PI * 300 * t)
    })
    const s = spectrogram(x, { fs, nperseg: window, noverlap: Math.floor(window * 0.75) })
    const p = toArray(s.values) as number[][]
    const z = p.map((row) => row.map((v) => (db ? 10 * Math.log10(v + 1e-12) : v)))
    const f = toArray(s.frequencies) as number[]
    const t = toArray(s.times) as number[]
    return cartesian([heatmap(z, { x: t, y: f, mode, label: db ? 'power (dB)' : 'power' })], {
      mode,
      x: { label: 'time (s)' },
      y: { label: 'frequency (Hz)' },
    })
  }, [window, sweep, db, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="window" value={window} onChange={setWindow} min={16} max={1024} integer log />
        <NumberField label="sweep rate (Hz/s)" value={sweep} onChange={setSweep} min={0} max={240} integer />
        <Toggle label="decibels" value={db} onChange={setDb} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="spectrogram">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}

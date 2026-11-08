import { key, normal } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { type Candle, candlestickChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Candlestick chart',
  question: 'How do I draw open, high, low and close prices with their volume?',
  explain:
    '`candlestickChart(candles, { average })` draws each period as a box from open to close (green up, red down) with wicks to the high and low, a moving average over it and the volume beneath; drag the slider to zoom. The prices are a toy random walk with intraday noise.',
}

const FIGURE = figure({ width: 860, height: 460 })

export default function Candlestick(): ReactNode {
  const mode = useColourMode()
  const [volatility, setVolatility] = useState(1)
  const [average, setAverage] = useState(20)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const n = 160
    const z = toArray(normal(key(seed), [5 * n])) as number[]
    let price = 100
    const candles: Candle[] = []
    for (let i = 0; i < n; i++) {
      const open = price
      const close = open * Math.exp(0.012 * volatility * (z[5 * i] ?? 0) + 0.0005)
      const high = Math.max(open, close) * (1 + 0.006 * volatility * Math.abs(z[5 * i + 1] ?? 0))
      const low = Math.min(open, close) * (1 - 0.006 * volatility * Math.abs(z[5 * i + 2] ?? 0))
      const volume = Math.round(
        1e5 * (1 + 0.5 * Math.abs(z[5 * i + 3] ?? 0)) * (1 + (Math.abs(close - open) / open) * 30),
      )
      candles.push({ label: `day ${i + 1}`, open, high, low, close, volume })
      price = close
    }
    return candlestickChart(candles, { mode, average })
  }, [volatility, average, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="volatility" value={volatility} onChange={setVolatility} min={0.2} max={4} step={0.1} log />
        <NumberField label="average window" value={average} onChange={setAverage} min={2} max={60} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="candlestick">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}

import { type ModelRun, modelRun } from 'omai-engine/exposition'
import { structure } from 'omai-engine/scene'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MessageReplay } from './MessageReplay.tsx'
import { MAX_REPLAY_STEPS, replayRecord } from './replay.ts'

const graph = structure(
  'factor-graph',
  [
    { id: 'x', label: 'x', role: 'latent' },
    { id: 'factor:x', label: '', role: 'factor', data: { site: 'x' } },
  ],
  [{ from: 'factor:x', to: 'x', directed: false }],
)

/** A run of n sweeps whose beliefs move less each time. */
const longRun = (n: number): ModelRun =>
  modelRun({
    structure: graph,
    status: 'converged',
    method: 'mp',
    series: [
      {
        name: 'change',
        label: 'largest change of a belief',
        steps: Array.from({ length: n }, (_, i) => i + 1),
        values: Array.from({ length: n }, (_, i) => 1 / (i + 1)),
        direction: 'lower',
      },
    ],
    steps: Array.from({ length: n + 1 }, (_, t) => ({
      step: t,
      nodes: { x: { mean: 1 - 1 / (t + 1), sd: 1 / (t + 1), text: `step ${t}` } },
      ...(t === 0 ? {} : { messages: [{ from: 'factor:x', to: 'x', text: `N(μ = ${t}, σ² = 1)` }] }),
    })),
  })

describe('replayRecord', () => {
  it('leaves a short run alone and cuts a long one to the cap, keeping its first and last step', () => {
    const short = longRun(5)
    expect(replayRecord(short)).toBe(short)
    const long = longRun(300)
    const cut = replayRecord(long)
    expect(cut.steps).toHaveLength(MAX_REPLAY_STEPS)
    expect(cut.steps[0]?.step).toBe(0)
    expect(cut.steps.at(-1)?.step).toBe(300)
    // the steps where the beliefs moved most are the early ones
    expect(cut.steps[1]?.step).toBe(1)
    expect(JSON.stringify(cut).length).toBeLessThan(60_000)
  })

  it('makes a record that survives a structured clone', () => {
    const cut = replayRecord(longRun(100))
    expect(structuredClone(cut)).toEqual(cut)
  })
})

describe('MessageReplay', () => {
  it('draws a figure with the caption, the graph, a play button and a scrubber', () => {
    const html = renderToString(<MessageReplay record={longRun(3)} caption="THE CAPTION" label="Messages of a test" />)
    expect(html).toContain('THE CAPTION')
    expect(html).toContain('story-reveal')
    expect(html).toContain('aria-label="Messages of a test"')
    expect(html).toContain('aria-label="Play the replay"')
    expect(html).toContain('type="range"')
    expect(html).toContain('<svg')
  })
})

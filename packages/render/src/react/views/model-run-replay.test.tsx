import { type ModelRun, modelRun, modelRunAt } from 'omai-engine/exposition'
import { structure } from 'omai-engine/scene'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { modelRunOverlay } from '../../views/model-run.ts'
import { ModelRunReplay, replayDrawing, replayLines } from './model-run-replay.tsx'

const graph = structure(
  'factor-graph',
  [
    { id: 'x', label: 'x', role: 'latent' },
    { id: 'factor:x', label: '', role: 'factor', data: { site: 'x', spec: 'factor', of: 'latent' } },
    { id: 'y', label: 'y', role: 'observed' },
    { id: 'factor:y', label: '', role: 'factor', data: { site: 'y', spec: 'factor', of: 'observed' } },
  ],
  [
    { from: 'factor:x', to: 'x', directed: false },
    { from: 'x', to: 'factor:y', directed: false },
    { from: 'factor:y', to: 'y', directed: false },
  ],
)

const run: ModelRun = modelRun({
  structure: graph,
  status: 'converged',
  method: 'mp',
  series: [],
  steps: [
    { step: 0, nodes: { x: { improper: true, text: 'improper' }, y: { observed: 2, text: 'observed 2' } } },
    {
      step: 1,
      nodes: { x: { mean: 1.5, sd: 0.7, text: '1.5 ± 0.7' } },
      messages: [
        { from: 'factor:x', to: 'x', text: 'N(μ = 0, σ² = 4)', operator: 'prior' },
        { from: 'factor:y', to: 'x', text: 'N(μ = 2, σ² = 1)' },
      ],
    },
  ],
})

describe('replayLines', () => {
  it('names the messages by their ends and gives the beliefs of the nodes that are not observed', () => {
    expect(replayLines(run, 1)).toEqual({
      messages: ['factor of x to x: N(μ = 0, σ² = 4)', 'factor of y to x: N(μ = 2, σ² = 1)'],
      beliefs: ['x: 1.5 ± 0.7'],
    })
  })

  it('has nothing to say before any step belief or message is known', () => {
    expect(replayLines(run, 0)).toEqual({ messages: [], beliefs: [] })
  })
})

describe('replayDrawing', () => {
  const later = modelRun({
    structure: graph,
    status: 'running',
    series: [],
    steps: [
      { step: 3, nodes: { x: { mean: 1.5, sd: 0.7, text: '1.5 ± 0.7' } } },
      { step: 5, nodes: { x: { improper: true, text: 'improper (precision -1)' } } },
    ],
  })

  it('does not let the last proper belief show at a step where the node has none', () => {
    const drawn = replayDrawing(later)
    expect(modelRunAt(drawn, 3)?.nodes['x']?.mean).toBe(1.5)
    const at5 = modelRunAt(drawn, 5)?.nodes['x']
    expect(at5?.mean).toBeUndefined()
    expect(at5?.text).toBe('')
    expect(modelRunOverlay(drawn, 5).badges?.['x']).toBeUndefined()
    expect(modelRunOverlay(drawn, 3).badges?.['x']).toBeDefined()
  })
})

describe('ModelRunReplay on the server', () => {
  it('draws the graph, the step in words, the play button and a scrubber with a spoken value', () => {
    const html = renderToString(<ModelRunReplay record={run} label="Messages in the test" />)
    expect(html).toContain('aria-label="Messages in the test"')
    expect(html).toContain('aria-label="Play the replay"')
    expect(html).toContain('type="range"')
    expect(html).toContain('aria-valuetext="step 1 of 2"')
    expect(html).toContain('step 0')
    expect(html).toContain('<svg')
  })

  it('can open on the last step, where the messages and the belief are written out', () => {
    // Rendering to a string does not run effects, so the opening step is checked through the lines the last step gives.
    const lines = replayLines(run, run.steps.at(-1)?.step ?? 0)
    expect(lines.messages).toHaveLength(2)
    expect(renderToString(<ModelRunReplay record={run} startAt="last" />)).toContain('type="range"')
  })

  it('has no controls for a run of one step, and takes a description of its own', () => {
    const one = modelRun({ ...run, steps: [run.steps[1] as ModelRun['steps'][number]] })
    const html = renderToString(<ModelRunReplay record={one} describe={(step) => `custom words for step ${step}`} />)
    expect(html).not.toContain('type="range"')
    expect(html).toContain('custom words for step 1')
  })
})

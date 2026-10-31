import {
  audioClip,
  decisionSurface,
  equationFit,
  sampleGrid,
  sampleLandscape,
  sampleStabilityRegion,
  transcript,
} from 'mdml-engine/exposition'
import { dict, int } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  type Method,
  session,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { staticLive } from './parts.tsx'
import { SessionPanels } from './SessionPanels.tsx'

// A method of nothing but its configuration: a counter, and a probe for each specialised panel.
interface State {
  readonly k: number
}

const toy: Method<State, Config> = defineMethod<State, Config>({
  name: 'toy',
  config: dict({ n: int(1, 10, { default: 2 }) }),
  make: () => ({
    name: 'toy',
    config: {},
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('position', [s.k, 1])
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= 10 ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as State,
  }),
  contract: {
    position: tensor({ role: 'state', shape: [2] }),
    picture: derived<State>({
      description: 'a grid over the viewport asked for',
      compute: (_, args) => {
        const a = args as { view: { x: [number, number]; y: [number, number] }; size: [number, number] }
        const xs = Array.from(
          { length: a.size[0] },
          (_, j) => a.view.x[0] + (j / (a.size[0] - 1)) * (a.view.x[1] - a.view.x[0]),
        )
        const ys = Array.from(
          { length: a.size[1] },
          (_, i) => a.view.y[0] + (i / (a.size[1] - 1)) * (a.view.y[1] - a.view.y[0]),
        )
        return decisionSurface({
          features: ['re', 'im'],
          x: xs,
          y: ys,
          values: ys.map((y) => xs.map((x) => x * x + y)),
        })
      },
    }),
    slice: derived<State>({
      description: 'a loss slice',
      compute: (s) =>
        sampleLandscape((a) => a * a, [-1, 0, 1], undefined, { paths: [{ coords: [[-1], [s.k * 0.1]] }] }),
    }),
    region: derived<State>({
      description: 'a stability region',
      compute: () =>
        sampleStabilityRegion(
          'forward Euler',
          { numerator: [1, 1] },
          { real: [-3, 1], imaginary: [-2, 2] },
          {
            n: 41,
            marks: [{ at: [-1, 0] }, { at: [-2.5, 0] }],
          },
        ),
    }),
    law: derived<State>({
      description: 'a fitted equation',
      compute: () =>
        equationFit({
          terms: [
            { tex: 'x', estimate: 2, truth: 2 },
            { tex: '1', estimate: 0.5, truth: 0 },
          ],
          x: [0, 1, 2],
          fit: [0.5, 2.5, 4.5],
          truth: [0, 2, 4],
        }),
    }),
    clip: derived<State>({
      description: 'an audio clip',
      compute: () =>
        audioClip({
          sampleRate: 1000,
          samples: Array.from({ length: 1000 }, (_, i) => Math.sin(i / 5)),
          frames: [
            { step: 0, time: 0 },
            { step: 10, time: 1 },
          ],
        }),
    }),
    words: derived<State>({
      description: 'a transcript',
      compute: () =>
        transcript({
          tokens: [
            { text: 'Hello', role: 'user' },
            { text: ' there', role: 'assistant' },
          ],
          tracks: [
            { name: 'head 1', weights: [0.25, 0.75] },
            { name: 'head 2', weights: [1, 0] },
          ],
          emitted: 1,
        }),
    }),
    gallery: derived<State>({
      description: 'samples over training',
      compute: () =>
        sampleGrid({
          modality: 'text',
          steps: [0, 2, 4].map((step) => ({ step, samples: [`sample at ${step}`] })),
        }),
    }),
    wrong: derived<State>({ description: 'not a surface', compute: () => ({ kind: 'nope' }) }),
  },
})

const live = staticLive(session(toy, { mode: 'eager', steps: 6 }))

describe('the zoom panel', () => {
  it('draws the picture with its controls, minimap and inspector, probing at the viewport', () => {
    const html = renderToString(
      <SessionPanels
        id="test/zoom"
        live={live}
        at={3}
        layout={[{ kind: 'zoom', path: 'picture', view: { x: [-1, 1], y: [0, 2] }, title: 'the zoom' }]}
      />,
    )
    expect(html).toContain('the zoom at step 3')
    expect(html).toContain('<canvas')
    expect(html).toContain('Zoom in')
    expect(html).toContain('Reset the viewport')
    expect(html).toContain('Minimap')
    expect(html).toContain('zoom 1.00×')
    expect(html).toContain('Move the pointer')
    expect(html).not.toContain('unavailable')
  })

  it('says why a probe is not a surface with values', () => {
    const html = renderToString(
      <SessionPanels
        id="test/zoom-wrong"
        live={live}
        at={2}
        layout={[{ kind: 'zoom', path: 'wrong', view: { x: [0, 1], y: [0, 1] } }]}
      />,
    )
    expect(html).toContain('the probe is not a decision surface with values')
  })
})

describe('the landscape panel', () => {
  it('draws the slice probed at the viewed step, or says why it cannot', () => {
    const html = renderToString(
      <SessionPanels
        id="test/landscape"
        live={live}
        at={4}
        layout={[
          { kind: 'landscape', path: 'slice', title: 'the slice' },
          { kind: 'landscape', path: 'wrong', title: 'not a slice' },
        ]}
      />,
    )
    expect(html).toContain('the slice at step 4')
    expect(html).toContain('Lowest on the slice: 0.00')
    expect(html).toContain('the probe is not a landscapeSlice record')
  })
})

describe('the stability panel', () => {
  it('draws the region with its marks and the verdict in words', () => {
    const html = renderToString(
      <SessionPanels
        id="test/stability"
        live={live}
        at={2}
        layout={[{ kind: 'stability', path: 'region', title: 'the region' }]}
      />,
    )
    expect(html).toContain('the region at step 2')
    expect(html).toContain('forward Euler is unstable: 1 of 2 values of hλ are outside the region')
  })
})

describe('the equation panel', () => {
  it('typesets the fit beside the truth with the status of each term in words', () => {
    const html = renderToString(
      <SessionPanels
        id="test/equation"
        live={live}
        at={1}
        layout={[{ kind: 'equation', path: 'law', title: 'the law' }]}
      />,
    )
    expect(html).toContain('the law at step 1')
    expect(html).toContain('katex')
    expect(html).toContain('matches the truth')
    expect(html).toContain('kept, but not in the law')
    expect(html).toContain('1 of 2 terms match the truth, 1 extra, 0 missing')
  })
})

describe('the audio panel', () => {
  it('draws the clip with the playhead of the viewed step and a play button', () => {
    const html = renderToString(
      <SessionPanels
        id="test/audio"
        live={live}
        at={5}
        layout={[{ kind: 'audio', path: 'clip', title: 'the clip' }]}
      />,
    )
    expect(html).toContain('the clip at step 5')
    expect(html).toContain('playhead 0.50 s of 1.00 s')
    expect(html).toContain('Play the clip from the playhead')
    expect(html).toContain('Waveform and spectrogram of a 1 s clip, playhead at 0.5 s')
  })
})

describe('the transcript panel', () => {
  it('draws the turns, marks the emitted token and names the heaviest', () => {
    const html = renderToString(
      <SessionPanels
        id="test/transcript"
        live={live}
        at={2}
        layout={[{ kind: 'transcript', path: 'words', title: 'the words' }]}
      />,
    )
    expect(html).toContain('the words at step 2')
    expect(html).toContain('data-emitted="true"')
    expect(html).toContain('Heaviest head 1: “there” 0.75, “Hello” 0.25.')
    expect(html).toContain('Tint the tokens by head 2')
  })
})

describe('the samples panel', () => {
  it('draws the samples of the viewed step with the filmstrip of earlier ones', () => {
    const html = renderToString(
      <SessionPanels
        id="test/samples"
        live={live}
        at={4}
        layout={[{ kind: 'samples', path: 'gallery', filmstrip: 2, title: 'the samples' }]}
      />,
    )
    expect(html).toContain('the samples at step 4')
    expect(html).toContain('sample at 4')
    expect(html).toContain('earlier steps')
    expect(html).toContain('Show earlier steps beneath the grid')
  })
})

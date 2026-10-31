import { decisionSurface, sampleLandscape } from 'mdml-engine/exposition'
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

import {
  complexPlane,
  imageSteps,
  markedTable,
  pointsSteps,
  reachabilityRecord,
  sampleVectorField,
  sceneSteps,
  simplexRecord,
  stripRecord,
  structureSteps,
} from 'omai-engine/exposition'
import { circle, scene, structure } from 'omai-engine/scene'
import { dict, int } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  session,
  stateFromJson,
  stateToJson,
  tensor,
} from 'omai-engine/trace'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { SessionPanels } from './SessionPanels.tsx'
import { type LiveSession, STILL_HANDLE } from './useSession.ts'

// A method of nothing but its configuration: a counter, and a probe of each record kind the panels draw.
interface State {
  readonly k: number
}
const graph = structure('graph', [{ id: 'a', label: 'a', role: 'latent' }], [])
const toy: Method<State, Config> = defineMethod<State, Config>({
  name: 'toy',
  config: dict({ n: int(1, 10, { default: 2 }) }),
  make: () => ({
    name: 'toy',
    config: {},
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('position', [s.k, 1])
      ctx.log('matrix', [
        [s.k, 1],
        [2, 3],
      ])
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= 10 ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as State,
  }),
  contract: {
    position: tensor({ role: 'state', shape: [2] }),
    matrix: tensor({ role: 'state', shape: [2, 2] }),
    flow: derived<State>({
      description: 'a field',
      compute: () => sampleVectorField((x, y) => [y, -x], [-1, 0, 1], [-1, 0, 1]),
    }),
    tags: derived<State>({
      description: 'a strip',
      compute: () => stripRecord({ tokens: ['a', 'b'], tracks: [{ name: 'p', values: [0, 1] }] }),
    }),
    results: derived<State>({
      description: 'a table',
      compute: () => markedTable({ columns: ['a'], cells: [[1]], marks: [{ row: 0, status: 'decision' }] }),
    }),
    glyphs: derived<State>({
      description: 'images',
      compute: (s) =>
        imageSteps({
          method: 'toy',
          height: 1,
          width: 1,
          groups: ['g'],
          steps: [{ iteration: s.k, images: [[[0], [1]]], labels: [['x', 'y']] }],
        }),
    }),
    net: derived<State>({
      description: 'a graph',
      compute: (s) => structureSteps({ structure: graph, steps: [{ step: s.k, nodes: { a: { value: 1 } } }] }),
    }),
    cloud: derived<State>({
      description: 'points',
      compute: () => pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 0]] }] }),
    }),
    picture: derived<State>({
      description: 'a scene',
      compute: (s) => sceneSteps({ steps: [{ step: s.k, scene: scene(40, 40, [circle(20, 20, 5)]), caption: 'hi' }] }),
    }),
    poles: derived<State>({
      description: 'a complex plane',
      compute: () => complexPlane({ domain: 's', poles: [[-1, 0]] }),
    }),
    game: derived<State>({
      description: 'a simplex',
      compute: () => simplexRecord({ shape: 'square', labels: ['a', 'b'] }),
    }),
    optics: derived<State>({
      description: 'a reachability plot',
      compute: () => reachabilityRecord({ reach: [null, 0.1] }),
    }),
    marks: derived<State>({ description: 'marks on a matrix', compute: () => [{ row: 1 }] }),
    wrong: derived<State>({ description: 'not a strip', compute: () => ({ kind: 'nope' }) }),
  },
  defaults: { position: Log({ detail: 'full' }), matrix: Log({ detail: 'full' }) },
})

function liveOf(steps: number): LiveSession<State, Config> {
  const s = session(toy, { mode: 'eager', steps })
  return {
    session: s,
    madeWith: { config: s.config, seed: s.seed },
    version: 0,
    head: s.t,
    done: s.status !== undefined,
    computing: false,
    compute: () => undefined,
    halt: () => undefined,
    advance: () => undefined,
    lastMs: 0,
    stale: false,
    held: false,
    ...STILL_HANDLE,
    apply: () => s,
  }
}

describe('the panels of probed records', () => {
  it('draw a field with a trajectory, a strip, a table, images, a graph, points and a scene to markup', () => {
    const html = renderToString(
      <SessionPanels
        id="test/records"
        live={liveOf(6)}
        at={4}
        onScrub={() => undefined}
        layout={[
          { kind: 'field', path: 'flow', trajectory: { paths: ['position'] }, title: 'the field' },
          { kind: 'strip', path: 'tags', title: 'the strip' },
          { kind: 'table', path: 'results', title: 'the table' },
          { kind: 'images', path: 'glyphs', title: 'the images' },
          { kind: 'graph', path: 'net', title: 'the graph' },
          { kind: 'points', path: 'cloud', title: 'the points' },
          { kind: 'scene', path: 'picture', title: 'the scene' },
          { kind: 'probe', path: 'poles', title: 'the plane' },
          { kind: 'probe', path: 'game', title: 'the game' },
          { kind: 'probe', path: 'optics', title: 'the optics' },
          {
            kind: 'tensor',
            path: 'matrix',
            rowLabels: ['r1', 'r2'],
            colLabels: ['c1', 'c2'],
            marks: [{ colEdge: 1 }],
            marksPath: 'marks',
            title: 'the matrix',
          },
        ]}
      />,
    )
    for (const title of ['the field', 'the strip', 'the table', 'the images', 'the graph', 'the points', 'the scene'])
      expect(html, title).toContain(`${title} at step 4`)
    expect(html).toContain('the plane at step 4')
    expect(html).toContain('the matrix at step 4 (2×2)')
    // A record of its own view shows its caption or an SVG scene; none says nothing is available.
    expect(html).not.toContain('unavailable')
    expect(html).not.toContain('nothing recorded yet')
  })

  it('say why a probe is not the record the panel draws', () => {
    const html = renderToString(
      <SessionPanels
        id="test/wrong"
        live={liveOf(3)}
        at={2}
        layout={[
          { kind: 'strip', path: 'wrong', title: 'a strip' },
          { kind: 'field', path: 'tags', title: 'a field' },
        ]}
      />,
    )
    expect(html).toContain('the probe is not a strip record')
    expect(html).toContain('the probe is not a vector field')
  })
})

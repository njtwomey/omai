import { dict, int } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  session,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { sessionPreferences, setSessionPreferences } from './preferences.ts'
import { initialSettings, logsOf } from './SessionControls.tsx'
import { SessionPanels } from './SessionPanels.tsx'
import { SessionPlayer } from './SessionPlayer.tsx'
import { nearestStep } from './scrub.ts'
import type { LiveSession } from './useSession.ts'

// A method of nothing but its configuration: a point walking a circle, with its angle, position and a 2-by-2 matrix
// on the tape and the plane's distance from the origin as a probe.
interface State {
  readonly k: number
  readonly x: number
  readonly y: number
}
const walk: Method<State, Config> = defineMethod<State, Config>({
  name: 'walk',
  config: dict({ turns: int(1, 10, { default: 2 }) }),
  make: (c) => {
    const turns = c['turns'] as number
    return {
      name: 'walk',
      config: { turns },
      init: () => ({ k: 0, x: 1, y: 0 }),
      step: (s, ctx) => {
        const k = s.k + 1
        const a = (2 * Math.PI * turns * k) / 100
        const next = { k, x: Math.cos(a), y: Math.sin(a) }
        ctx.log('angle', a)
        ctx.log('position', [next.x, next.y])
        ctx.log('matrix', [
          [next.x, -next.y],
          [next.y, next.x],
        ])
        ctx.log('cov', [
          [0.1, 0],
          [0, 0.1],
        ])
        return next
      },
      done: (s) => (s.k >= 100 ? 'converged' : undefined),
      // Checkpoints (and so probes at any step) need states that serialise.
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    angle: scalar(),
    position: tensor({ role: 'state', shape: [2] }),
    matrix: tensor({ role: 'state', shape: [2, 2] }),
    cov: tensor({ role: 'state', shape: [2, 2] }),
    'plane/distance': derived<State>({
      description: 'the distance from the origin over the plane',
      compute: () => {
        const grid = [-1, 0, 1]
        return {
          kind: 'decisionSurface',
          features: ['x', 'y'],
          x: grid,
          y: grid,
          values: grid.map((y) => grid.map((x) => Math.hypot(x, y))),
          field: { name: 'distance' },
        }
      },
    }),
  },
  // Tensors are drawn element by element, so they are kept in full (`all` keeps summaries).
  defaults: {
    angle: Log(),
    position: Log({ detail: 'full' }),
    matrix: Log({ detail: 'full' }),
    cov: Log({ detail: 'full' }),
  },
})

function liveOf(steps: number): LiveSession<State, Config> {
  const s = session(walk, { mode: 'eager', steps })
  return {
    session: s,
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
    apply: () => s,
  }
}

describe('session panels', () => {
  it('draws every panel kind of a dataset-free method to markup', () => {
    const live = liveOf(20)
    const html = renderToString(
      <SessionPanels
        id="test/walk"
        live={live}
        at={10}
        onScrub={() => undefined}
        layout={[
          { kind: 'curves', paths: ['angle'] },
          { kind: 'series', path: 'position', names: ['x', 'y'] },
          { kind: 'series', path: 'position', x: 'angle', title: 'against the angle' },
          { kind: 'trajectory', paths: ['position'], over: { path: 'plane/distance' }, ellipse: 'cov' },
          { kind: 'trajectory', paths: ['position'], points: true, trail: 5 },
          { kind: 'tensor', path: 'matrix' },
          { kind: 'probe', path: 'plane/distance' },
          { kind: 'rows', overlay: 'fit' },
        ]}
      />,
    )
    expect(html).toContain('data-testid="panels"')
    expect(html).toContain('against the angle')
    expect(html).toContain('matrix at step 10 (2×2)')
    expect(html).toContain('position to step 10')
  })

  it('clips the viewed step to the head and says when a tensor has nothing yet', () => {
    const live = liveOf(0)
    const html = renderToString(
      <SessionPanels id="t" live={live} at={99} layout={[{ kind: 'tensor', path: 'matrix' }]} />,
    )
    expect(html).toContain('nothing recorded yet')
  })

  it('finds the nearest step by step number or by position', () => {
    expect(nearestStep([0, 5, 10], 6)).toBe(5)
    expect(nearestStep([0, 5, 10], 6, [0, 0.5, 1])).toBe(10)
    expect(nearestStep([], 3)).toBeUndefined()
  })

  it('turns settings into a subscription', () => {
    expect(logsOf(initialSettings('defaults'))).toBe('defaults')
    expect(initialSettings('off', 50).steps).toBe(50)
    const custom = initialSettings({ angle: { every: 1 } as never })
    expect(custom.preset).toBe('custom')
    expect(Object.keys(logsOf(custom))).toEqual(['angle'])
  })

  it('draws the player with the eager toggle, both seeks and the head in the stamp', () => {
    const html = renderToString(<SessionPlayer live={liveOf(20)} at={5} onScrub={() => undefined} />)
    expect(html).toContain('step 5 of 20')
    expect(html).toContain('aria-label="Eager"')
    expect(html).toContain('aria-label="Go to the start"')
    expect(html).toContain('aria-label="Go to the end"')
  })

  it('keeps the reader’s preferences, with or without storage', () => {
    expect(sessionPreferences()).toEqual({ mode: 'live', hold: true })
    setSessionPreferences({ mode: 'eager' })
    expect(sessionPreferences()).toEqual({ mode: 'eager', hold: true })
    setSessionPreferences({ mode: 'live', hold: false })
    expect(sessionPreferences().hold).toBe(false)
    setSessionPreferences({ hold: true })
  })
})

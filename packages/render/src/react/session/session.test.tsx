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
import { advanceInStep, overlayCurves, sharedHead, sweepTrials } from './compare.ts'
import { sessionPreferences, setSessionPreferences } from './preferences.ts'
import { SessionPanels } from './SessionPanels.tsx'
import { SessionPlayer } from './SessionPlayer.tsx'
import { nearestStep } from './scrub.ts'
import type { LiveSession, LiveSessions } from './useSession.ts'

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

function liveOf(steps: number, config: Partial<Config> = {}): LiveSession<State, Config> {
  const s = session(walk, { mode: 'eager', steps, config })
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

// A comparison of eager sessions, as `useSessions` would hand one over once every variant has ended.
function comparisonOf(
  variants: readonly { label: string; turns: number; steps: number }[],
): LiveSessions<State, Config> {
  const list = variants.map((v) => ({ label: v.label, live: liveOf(v.steps, { turns: v.turns }) }))
  return {
    variants: list,
    version: 0,
    head: sharedHead(list.map((v) => v.live)),
    done: list.every((v) => v.live.done),
    computing: false,
    compute: () => undefined,
    halt: () => undefined,
    advance: () => undefined,
    lastMs: 0,
    stale: false,
    held: false,
    apply: () => list.map((v) => v.live.session),
  }
}

describe('comparisons', () => {
  it('overlays the curves of the variants and draws the per-step panels side by side, each titled by its label', () => {
    const live = comparisonOf([
      { label: 'slow', turns: 1, steps: 20 },
      { label: 'fast', turns: 3, steps: 20 },
    ])
    const html = renderToString(
      <SessionPanels
        id="test/compare"
        live={live}
        at={10}
        onScrub={() => undefined}
        layout={[
          { kind: 'curves', paths: ['angle'] },
          { kind: 'sweep', path: 'angle', over: 'angle' },
          { kind: 'tensor', path: 'matrix' },
          { kind: 'probe', path: 'plane/distance' },
          { kind: 'tape' },
        ]}
      />,
    )
    expect(html).toContain('data-testid="compared"')
    expect(html).toContain('slow · matrix at step 10 (2×2)')
    expect(html).toContain('fast · matrix at step 10 (2×2)')
    expect(html).toContain('slow · plane/distance at step 10')
    expect(html).toContain('angle against angle')
    const layer = overlayCurves(
      live.variants.map((v) => ({ label: v.label, run: v.live.session.run() })),
      ['angle'],
    )
    expect(layer.series.map((x) => x.name)).toEqual(['slow', 'fast'])
    const two = overlayCurves(
      live.variants.map((v) => ({ label: v.label, run: v.live.session.run() })),
      ['angle', 'angle'],
    )
    expect(two.series.map((x) => x.name)).toEqual(['slow · angle', 'slow · angle', 'fast · angle', 'fast · angle'])
    // One run is its own metric curves, named by path.
    expect(
      overlayCurves([{ label: 'only', run: live.variants[0]?.live.session.run() as never }], ['angle']).series[0]?.name,
    ).toBe('angle')
  })

  it('reads the trials of a sweep against the step or against a knob logged at the same step', () => {
    const run = liveOf(5).session.run()
    const byStep = sweepTrials(run, 'angle')
    expect(byStep.steps).toEqual([0, 1, 2, 3, 4])
    expect(byStep.x).toEqual(byStep.steps)
    const byKnob = sweepTrials(run, 'angle', 'angle')
    expect(byKnob.x).toEqual(byKnob.y)
    expect(sweepTrials(run, 'angle', 'missing').steps).toEqual([])
  })

  it('shares the head of the slowest unfinished variant, or the furthest once all have ended', () => {
    expect(sharedHead([])).toBe(0)
    expect(
      sharedHead([
        { head: 10, done: false },
        { head: 30, done: false },
      ]),
    ).toBe(10)
    expect(
      sharedHead([
        { head: 10, done: true },
        { head: 30, done: false },
      ]),
    ).toBe(30)
    expect(
      sharedHead([
        { head: 10, done: true },
        { head: 30, done: true },
      ]),
    ).toBe(30)
  })

  it('advances several sessions in step: a step budget brings the slowest up, a time budget takes turns', () => {
    const a = session(walk, { steps: 40 })
    const b = session(walk, { steps: 40 })
    b.advance({ steps: 3 })
    advanceInStep([a, b], { steps: 2 })
    expect([a.t, b.t]).toEqual([2, 3])
    advanceInStep([a, b], { steps: 5 })
    expect([a.t, b.t]).toEqual([7, 7])
    let now = 0
    advanceInStep([a, b], { ms: 10_000, clock: () => (now += 1) })
    expect([a.t, b.t]).toEqual([40, 40])
    expect(a.status).toBe('completed')
    advanceInStep([a, b], { steps: 1 })
    expect([a.t, b.t]).toEqual([40, 40])
  })

  it('plays a comparison with the slowest head in the stamp, and offers the reset when the page has one', () => {
    const live = comparisonOf([
      { label: 'slow', turns: 1, steps: 12 },
      { label: 'fast', turns: 3, steps: 20 },
    ])
    const html = renderToString(
      <SessionPlayer live={live} at={5} onScrub={() => undefined} onReset={() => undefined} />,
    )
    expect(html).toContain('step 5 of 20 · completed')
    expect(html).toContain('× 2 variants')
    expect(html).toContain('aria-label="Reset to defaults"')
    const plain = renderToString(<SessionPlayer live={liveOf(20)} at={5} onScrub={() => undefined} />)
    expect(plain).not.toContain('aria-label="Reset to defaults"')
  })
})

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

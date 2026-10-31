/**
 * The inner-session panel: a session inside a step of another (an optimiser inside a training step, an inner loop of a
 * meta-learner), drawn as a framed box with its own layout. The page registers the inner sessions by name; the panel
 * makes the one for the viewed outer step (or, following the outer cursor, for the end of the outer run so far), runs
 * it to its end, and hands it to a render function that draws its layout. Its cursor is its own, with a slider, or the
 * outer cursor's. `innerCursor` is the pure rule that picks the inner step.
 */

import type { Json } from 'mdml-engine/base'
import { type Config, type Method, type Session, type SessionOptions, session } from 'mdml-engine/trace'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { staticLive } from './parts.tsx'
import type { LiveSession } from './useSession.ts'

// biome-ignore lint/suspicious/noExplicitAny: sessions of any method
type AnySession = Session<any, any>
// biome-ignore lint/suspicious/noExplicitAny: the inner method has its own state and configuration
type AnyOptions = Omit<SessionOptions<any, Config>, 'mode'>

/**
 * How the page makes an inner session from the outer one: the inner method and the options of the run at an outer step.
 */
export interface InnerSessionSpec {
  // biome-ignore lint/suspicious/noExplicitAny: the inner method has its own state and configuration
  readonly method: Method<any, any>
  /**
   * The inner run's options at an outer step: its configuration from what the outer run holds there (probe the outer
   * session for it), its seed and its length. The panel makes it eager, so it runs whole.
   */
  readonly options: (outer: { readonly session: AnySession; readonly t: number; readonly args?: Json }) => AnyOptions
}

/** How the inner cursor moves: on its own (`own`, a slider) or with the outer cursor (`outer`). */
export type InnerFollow = 'own' | 'outer'

/**
 * The inner step to show. On its own, the inner cursor's step (the end of the inner run when it has not been moved).
 * Following the outer cursor, the outer step itself, held at the inner run's last step once the outer cursor is past it.
 *
 * @param follow - How the inner cursor moves.
 * @param outerStep - The outer step viewed.
 * @param innerHead - The inner run's last step.
 * @param own - The inner cursor when it moves on its own (`null`: not moved, at the end).
 * @returns The inner step, between zero and `innerHead`.
 * @example
 * ```ts
 * import { innerCursor } from 'mdml-render/react'
 * innerCursor('outer', 7, 20, null) // => 7
 * innerCursor('outer', 70, 20, null) // => 20
 * innerCursor('own', 7, 20, 3) // => 3
 * innerCursor('own', 7, 20, null) // => 20
 * ```
 */
export function innerCursor(follow: InnerFollow, outerStep: number, innerHead: number, own: number | null): number {
  const step = follow === 'outer' ? outerStep : (own ?? innerHead)
  return Math.max(0, Math.min(innerHead, Math.round(step)))
}

/**
 * A session inside a step of the outer one, framed, with a slider for its own cursor (or following the outer one).
 *
 * @param props - The spec of the inner session, the outer session and viewed step, how the cursor moves, the title and
 *   a function that draws the inner layout given the inner session, its cursor and its setter.
 * @returns The panel.
 */
export function InnerPanel(props: {
  readonly spec: InnerSessionSpec | undefined
  readonly name: string
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly t: number
  readonly head: number
  readonly args?: Json
  readonly follow?: InnerFollow
  readonly title: string
  // biome-ignore lint/suspicious/noExplicitAny: the inner method has its own state and configuration
  readonly render: (inner: LiveSession<any, any>, at: number, scrub: (t: number | null) => void) => ReactNode
}): ReactNode {
  const { spec, live, follow = 'own' } = props
  // Following the outer cursor, the one inner run is made at the outer head; on its own, at the viewed step.
  const at = follow === 'outer' ? props.head : props.t
  const argsKey = JSON.stringify(props.args ?? null)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the arguments are keyed by their text
  const made = useMemo(() => {
    if (spec === undefined) return { error: `no inner session named “${props.name}” is registered` }
    try {
      const options = spec.options({
        session: live.session,
        t: at,
        ...(props.args === undefined ? {} : { args: props.args }),
      })
      return { inner: session(spec.method, { ...options, mode: 'eager' }) }
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) }
    }
  }, [spec, props.name, live.session, at, argsKey])
  const [own, setOwn] = useState<number | null>(null)
  // A new inner run starts at its end again.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset when the inner run is remade
  useEffect(() => setOwn(null), [made.inner])
  return (
    <section
      className="flex flex-col gap-2 rounded-lg border border-(--wb-grid) p-3"
      aria-label={props.title}
      data-testid="inner-session"
    >
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="m-0 text-sm font-semibold text-(--wb-fg)">{props.title}</h3>
        {made.inner === undefined ? null : (
          <InnerCursor
            follow={follow}
            head={made.inner.t}
            at={innerCursor(follow, props.t, made.inner.t, own)}
            onChange={(v) => setOwn(v)}
          />
        )}
      </div>
      {made.inner === undefined ? (
        <span className="text-xs text-muted">unavailable: {made.error}</span>
      ) : (
        props.render(staticLive(made.inner), innerCursor(follow, props.t, made.inner.t, own), (v) => setOwn(v))
      )}
    </section>
  )
}

// The inner cursor's readout, with a slider when it moves on its own.
function InnerCursor(props: {
  readonly follow: InnerFollow
  readonly head: number
  readonly at: number
  readonly onChange: (step: number | null) => void
}): ReactNode {
  if (props.follow === 'outer')
    return (
      <span className="text-xs text-muted tabular-nums">{`inner step ${props.at} of ${props.head}, following the outer cursor`}</span>
    )
  return (
    <label className="flex items-center gap-2 text-xs text-muted tabular-nums">
      <span>{`inner step ${props.at} of ${props.head}`}</span>
      <input
        type="range"
        min={0}
        max={props.head}
        value={props.at}
        aria-label="Inner step"
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="w-40 accent-(--wb-accent)"
      />
    </label>
  )
}

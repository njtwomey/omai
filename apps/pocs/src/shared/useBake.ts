import { useEffect, useRef, useState } from 'react'

/** The part of a live session the bake needs: whether the run has ended, and which run it is. */
export interface Bakeable {
  readonly done: boolean
  readonly head: number
  /** Changes when the run is remade: a bake of an old run is thrown away. */
  readonly session: unknown
}

/** What {@link useBake} gives back. */
export interface Baked<T> {
  /** The baked view at the latest baked step at or before `t` (undefined until there is one). */
  readonly at: (t: number) => T | undefined
  /** Whether the views are still being computed (after the run has ended). */
  readonly baking: boolean
  /** Views computed and views to compute (0 before the run has ended). */
  readonly progress: { readonly done: number; readonly total: number }
}

const SLICE_MS = 14

/**
 * Computes the view of a finished run once, at up to `frames` evenly spaced steps, in slices of a few milliseconds
 * between animation frames so the page stays alive, and keeps them: playing back is then a lookup, where deriving a view
 * from the session on each frame would rebuild the state at that step every time (and, for a model that trains, re-run
 * part of the training).
 *
 * @param live - The run: it starts baking when `done` and restarts when `session` changes.
 * @param frames - The most views to bake.
 * @param view - The view at a step.
 * @returns The lookup, whether it is baking, and the progress.
 */
export function useBake<T>(live: Bakeable, frames: number, view: (t: number) => T): Baked<T> {
  const store = useRef(new Map<number, T>())
  const steps = useRef<number[]>([])
  const latest = useRef(view)
  latest.current = view
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [baking, setBaking] = useState(false)
  const [, setVersion] = useState(0)
  // biome-ignore lint/correctness/useExhaustiveDependencies: a bake is made for one run, and restarts with the next
  useEffect(() => {
    store.current = new Map()
    steps.current = []
    setProgress({ done: 0, total: 0 })
    setBaking(false)
    if (!live.done || live.head < 1) return
    const stride = Math.max(1, Math.ceil(live.head / frames))
    const list = Array.from({ length: Math.floor(live.head / stride) + 1 }, (_, i) => i * stride)
    if (list[list.length - 1] !== live.head) list.push(live.head)
    let next = 0
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    setBaking(true)
    setProgress({ done: 0, total: list.length })
    const slice = (): void => {
      if (cancelled) return
      const started = performance.now()
      while (next < list.length && performance.now() - started < SLICE_MS) {
        const t = list[next] as number
        store.current.set(t, latest.current(t))
        next++
      }
      steps.current = list.slice(0, next)
      setProgress({ done: next, total: list.length })
      if (next < list.length) timer = setTimeout(slice, 0)
      else {
        setBaking(false)
        setVersion((v) => v + 1)
      }
    }
    timer = setTimeout(slice, 0)
    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [live.session, live.done])
  const at = (t: number): T | undefined => {
    let best: number | undefined
    for (const s of steps.current) {
      if (s > t) break
      best = s
    }
    return best === undefined ? undefined : store.current.get(best)
  }
  return { at, baking, progress }
}

/**
 * A harness for testing hooks and small components in Node, with no DOM library: React DOM runs against a stand-in
 * document whose elements record what they are given and nothing else, animation frames are made by hand
 * (`fakeFrames`), and `performance.now` ticks by a fixed amount on every reading so a time budget ends the same way
 * every run. A test mounts a hook (`mountHook`) or an element (`mountElement`), steps frames, and reads what the hook
 * returned most recently.
 */

import { act, type ReactElement, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { vi } from 'vitest'

class FakeNode {
  readonly nodeType: number
  readonly nodeName: string
  readonly tagName: string
  readonly namespaceURI = 'http://www.w3.org/1999/xhtml'
  readonly style = { setProperty: () => undefined, removeProperty: () => undefined } as Record<string, unknown>
  readonly childNodes: FakeNode[] = []
  parentNode: FakeNode | null = null
  text = ''
  constructor(
    readonly ownerDocument: FakeDocument,
    name: string,
    kind = 1,
  ) {
    this.nodeType = kind
    this.nodeName = name.toUpperCase()
    this.tagName = this.nodeName
  }
  get firstChild(): FakeNode | null {
    return this.childNodes[0] ?? null
  }
  get nextSibling(): FakeNode | null {
    const i = this.parentNode?.childNodes.indexOf(this) ?? -1
    return this.parentNode?.childNodes[i + 1] ?? null
  }
  set textContent(value: string) {
    this.text = value
    this.childNodes.length = 0
  }
  get textContent(): string {
    return this.text
  }
  set nodeValue(value: string) {
    this.text = value
  }
  appendChild(child: FakeNode): FakeNode {
    return this.insertBefore(child, null)
  }
  insertBefore(child: FakeNode, before: FakeNode | null): FakeNode {
    child.parentNode?.removeChild(child)
    const i = before === null ? -1 : this.childNodes.indexOf(before)
    if (i < 0) this.childNodes.push(child)
    else this.childNodes.splice(i, 0, child)
    child.parentNode = this
    return child
  }
  removeChild(child: FakeNode): FakeNode {
    const i = this.childNodes.indexOf(child)
    if (i >= 0) this.childNodes.splice(i, 1)
    child.parentNode = null
    return child
  }
  setAttribute(): void {}
  removeAttribute(): void {}
  addEventListener(): void {}
  removeEventListener(): void {}
}

class FakeDocument extends FakeNode {
  activeElement = null
  readonly defaultView: unknown = undefined
  constructor() {
    super(undefined as never, '#document', 9)
  }
  createElement(name: string): FakeNode {
    return new FakeNode(this, name)
  }
  createElementNS(_ns: string, name: string): FakeNode {
    return new FakeNode(this, name)
  }
  createTextNode(text: string): FakeNode {
    const n = new FakeNode(this, '#text', 3)
    n.text = text
    return n
  }
}

/** A frame clock made by hand. */
export interface FakeFrames {
  /** Run the animation frames asked for so far, once, `ms` (default 16) later than the last (inside `act`). */
  frame(ms?: number): Promise<void>
  /** Run `n` frames. */
  frames(n: number): Promise<void>
  /** How many frames are waiting to run. */
  readonly pending: number
  /** Put the globals back. */
  restore(): void
}

/**
 * Replace `requestAnimationFrame`, `cancelAnimationFrame` and `performance.now` with a frame clock made by hand:
 * frames run when the test says, and every reading of `performance.now` moves time on by `readMs`.
 *
 * @param readMs - Milliseconds per reading of `performance.now` (default 1).
 * @returns The frame clock.
 */
export function fakeFrames(readMs = 1): FakeFrames {
  const g = globalThis as Record<string, unknown>
  const before = { raf: g['requestAnimationFrame'], caf: g['cancelAnimationFrame'] }
  const queue = new Map<number, FrameRequestCallback>()
  let id = 0
  let stamp = 0
  let reads = 0
  g['requestAnimationFrame'] = (cb: FrameRequestCallback): number => {
    queue.set(++id, cb)
    return id
  }
  g['cancelAnimationFrame'] = (h: number): void => {
    queue.delete(h)
  }
  const now = vi.spyOn(performance, 'now').mockImplementation(() => {
    reads += readMs
    return reads
  })
  const frame = async (ms = 16): Promise<void> => {
    stamp += ms
    const due = [...queue]
    queue.clear()
    await act(async () => {
      for (const [, cb] of due) cb(stamp)
    })
  }
  return {
    frame,
    frames: async (n) => {
      for (let i = 0; i < n; i++) await frame()
    },
    get pending() {
      return queue.size
    },
    restore: () => {
      now.mockRestore()
      g['requestAnimationFrame'] = before.raf
      g['cancelAnimationFrame'] = before.caf
    },
  }
}

/** A mounted hook or element. */
export interface Mounted<T> {
  /** What the hook returned on its latest render. */
  readonly current: T
  /** How many times it rendered. */
  readonly renders: number
  /** Render again (after changing what the hook reads). */
  rerender(): Promise<void>
  /** Unmount, running the cleanups. */
  unmount(): Promise<void>
}

/** The globals as they were before the first mounted tree, and how many trees are mounted now. */
const mounted: {
  saved: { window: unknown; hadWindow: boolean; act: unknown; hadAct: boolean } | undefined
  trees: number
} = {
  saved: undefined,
  trees: 0,
}

/**
 * Put back the globals a mounted tree replaced (`window`, `IS_REACT_ACT_ENVIRONMENT`). It runs by itself when the last
 * mounted tree unmounts; call it from `afterEach` in a test file as well, so a test that does not unmount leaves
 * nothing behind for the next file when files share one module graph (CI runs them that way).
 */
export function resetGlobals(): void {
  const g = globalThis as Record<string, unknown>
  const saved = mounted.saved
  mounted.saved = undefined
  mounted.trees = 0
  if (saved === undefined) return
  if (saved.hadWindow) g['window'] = saved.window
  else delete g['window']
  if (saved.hadAct) g['IS_REACT_ACT_ENVIRONMENT'] = saved.act
  else delete g['IS_REACT_ACT_ENVIRONMENT']
}

async function mount(element: (tick: () => number) => ReactElement, strict: boolean): Promise<Mounted<unknown>> {
  const g = globalThis as Record<string, unknown>
  mounted.saved ??= {
    window: g['window'],
    hadWindow: 'window' in g,
    act: g['IS_REACT_ACT_ENVIRONMENT'],
    hadAct: 'IS_REACT_ACT_ENVIRONMENT' in g,
  }
  mounted.trees++
  g['IS_REACT_ACT_ENVIRONMENT'] = true
  const doc = new FakeDocument()
  const window = { event: undefined, document: doc, HTMLIFrameElement: class {} }
  g['window'] = window
  ;(doc as { defaultView: unknown }).defaultView = window
  const container = doc.createElement('div')
  const root = createRoot(container as unknown as Element)
  let renders = 0
  const tree = (): ReactElement => {
    const inner = element(() => ++renders)
    return strict ? <StrictMode>{inner}</StrictMode> : inner
  }
  await act(async () => {
    root.render(tree())
  })
  return {
    current: undefined,
    get renders() {
      return renders
    },
    rerender: async () => {
      await act(async () => {
        root.render(tree())
      })
    },
    unmount: async () => {
      await act(async () => {
        root.unmount()
      })
      if (--mounted.trees <= 0) resetGlobals()
    },
  }
}

function HookHost(props: { readonly run: () => void }): null {
  props.run()
  return null
}

/**
 * Mount a hook and keep what it returns.
 *
 * @param hook - The hook, called on every render.
 * @param options - `strict` mounts it in `StrictMode` (effects run twice on mount, as in development).
 * @returns The mounted hook.
 */
export async function mountHook<T>(hook: () => T, options: { readonly strict?: boolean } = {}): Promise<Mounted<T>> {
  let latest: T | undefined
  let count = 0
  const mounted = await mount(
    () => (
      <HookHost
        run={() => {
          latest = hook()
          count += 1
        }}
      />
    ),
    options.strict ?? false,
  )
  return Object.defineProperties(mounted, {
    current: { get: () => latest as T },
    renders: { get: () => count },
  }) as Mounted<T>
}

/**
 * Mount an element.
 *
 * @param element - The element.
 * @param options - `strict` mounts it in `StrictMode`.
 * @returns The mounted element (its `current` is unused).
 */
export async function mountElement(
  element: ReactElement,
  options: { readonly strict?: boolean } = {},
): Promise<Mounted<void>> {
  return (await mount(() => element, options.strict ?? false)) as Mounted<void>
}

/**
 * Run something that sets state (pressing play, say) the way React expects a test to, and let the effects settle.
 *
 * @param action - What to do.
 */
export async function inAct(action: () => void): Promise<void> {
  await act(async () => {
    action()
  })
}

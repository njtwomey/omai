/**
 * The contract of an example: what `examples/<name>/main.tsx` exports so that `scripts/readme-examples.ts` can run it.
 *
 * - `meta`: the title, one line on what it shows, and whether the README shows it (`readme: true`) or only the index.
 * - `run()`: the compute. It uses the library the way the README snippet does and returns the result the figure is
 *   drawn from, and the tapes of the runs that made it (a session, or the `Run` a recorder kept), which the runner
 *   writes as `tape*.json` so the run can be replayed later.
 * - `animation` (optional, in what `run` returns): a timeline of scenes the runner writes as `learning.gif`.
 * - `View`: a React component (rendered to static markup, no browser) that draws the result with `mdml-render`.
 *
 * Adding an example means adding a folder with a `main.tsx`: nothing else lists it.
 */

import type { Timeline } from 'mdml-engine/scene'
import type { Run, Session } from 'mdml-engine/trace'
import type { ReactElement } from 'react'

/** What the index and the README say about an example. */
export interface Meta {
  readonly title: string
  /** One line on what it shows. */
  readonly summary: string
  /** Whether the README shows it (the best handful); every example is in `examples/README.md`. */
  readonly readme: boolean
  /** The alt text of its figure. */
  readonly alt: string
}

/** A run kept for replay: a session (written as a tape file with its method and probes) or the events of a recorder. */
// biome-ignore lint/suspicious/noExplicitAny: sessions of any method
export type Tape = Run | Session<any, any>

/** What `run` returns: the result the figure is drawn from, and the tapes of the runs behind it. */
export interface Outcome<R> {
  readonly result: R
  /** Tapes by name: one is written as `tape.json`, several as `tape-<name>.json`. */
  readonly tapes?: Readonly<Record<string, Tape>>
  /**
   * An animation of the run, as a timeline of scenes (drawn from the tape, one frame per step shown): the runner writes it
   * as `learning.gif`. Scenes are rasterised in Node, which draws no text, so a frame carries shapes only.
   */
  readonly animation?: Timeline
  /** Numbers worth printing when the example runs: the values the README quotes in its comments. */
  readonly notes?: readonly string[]
}

/** The module shape of a `main.tsx`. */
export interface ExampleModule<R = unknown> {
  readonly meta: Meta
  run(): Outcome<R> | Promise<Outcome<R>>
  View(props: { readonly result: R }): ReactElement
}

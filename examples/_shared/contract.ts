/**
 * The contract of an example: what `examples/<name>/main.tsx` exports so that `scripts/readme-examples.ts` can run it.
 *
 * - `meta`: the title, one line on what it shows, and whether the README shows it (`readme: true`) or only the index.
 * - `run()`: the compute. It uses the library the way the README snippet does and returns the result the figure is
 *   drawn from, and the tapes of the runs that made it (a session, or the `Run` a recorder kept), which the runner
 *   writes as `tape*.json` so the run can be replayed later.
 * - `animation` (optional, in what `run` returns): a timeline of scenes the runner writes as `learning.gif`.
 * - `View`: a React component (rendered to static markup, no browser) that draws the result with `omai-render`.
 *
 * Adding an example means adding a folder with a `main.tsx`: nothing else lists it.
 */

import type { Timeline } from 'omai-engine/scene'
import type { Run, Session } from 'omai-engine/trace'
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
  /**
   * Set for the examples the site's front page plays in a carousel (the site imports the example and runs it in the
   * browser): the tab's name, the playing rate and how long the slide shows.
   */
  readonly home?: {
    readonly label: string
    /** How long the slide shows before the next, in seconds: the time the run takes to play at `fps`, and a pause on the last frame. */
    readonly seconds?: number
    /** The frames a second the run plays at (default 10); the reader can change it with the player. */
    readonly fps?: number
  }
}

/** A run kept for replay: a session (written as a tape file with its method and probes) or the events of a recorder. */
// biome-ignore lint/suspicious/noExplicitAny: sessions of any method
export type Tape = Run | Session<any, any>

/**
 * A run that is stepped by whoever shows it, instead of being run to the end first: the site's front page calls `step` a
 * few times a second and draws `result()` after each, so what is on screen is being computed as it is watched.
 */
export interface Live<R> {
  /** Do one more step of the run; false once it is over. */
  step(): boolean
  /** What the run has produced so far, in the shape `run` returns, for `View` (drawn at its last frame). */
  result(): R
}

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
  /**
   * The figure. Without `at` it is the finished figure (what the runner writes as `figure.svg`); with `at`, the figure
   * as it was after that many frames of the run, so the site can play the recorded run (see `frames`). `mode` is the colour
   * mode to draw in (default light, which is what the runner writes): the site passes its own, so the figures follow the page.
   */
  View(props: { readonly result: R; readonly at?: number; readonly mode?: 'light' | 'dark' }): ReactElement
  /**
   * Start the run live: the site steps it and draws it as it goes (and starts it again when it is over), where `run` computes
   * everything first. The finished figure and tape still come from `run`.
   */
  live?(): Live<R>
  /** How many frames the run can be played through (`at` runs from 0 to `frames(result)`), for an example that animates. */
  frames?(result: R): number
}

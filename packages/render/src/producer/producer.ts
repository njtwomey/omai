/**
 * The producer: the half of a live session that computes. A page reads a tape at its own pace (the consumer: the
 * player and the panels) while a producer computes it ahead; this is the one interface between them, with two
 * implementations, the main-thread one that runs slices in the spare time of each animation frame (`inlineProducer`)
 * and the worker one that computes in a Web Worker and streams the tape (`workerProducer`). A `ProducerFactory` makes
 * the producer of a hook's sessions, so the consumer does not change with where the computing happens.
 */

import type { Json } from 'omai-engine/base'
import type { Config, Derived, Method, Session, SessionOptions, SliceBudget } from 'omai-engine/trace'
import type { Variant } from './make.ts'

/**
 * A producer of the sessions of a comparison (one variant for a session on its own).
 *
 * @index structure The half of a live session that computes; main-thread or worker.
 */
export interface Producer<S, C extends Config> {
  /** Where it computes: on the main thread in slices, or in a worker. */
  readonly kind: 'inline' | 'worker'
  /** Why a worker was asked for and not used (unavailable, or options that cannot cross), when that happened. */
  readonly fallback?: string
  /** The sessions the consumer reads, one per variant. */
  readonly sessions: readonly Session<S, C>[]
  readonly labels: readonly string[]
  /** Whether it computes by itself once started (a worker), rather than in slices the caller gives it. */
  readonly background: boolean
  /** Compute a slice now (inline), or ask the worker for this many more steps. */
  advance(budget: SliceBudget): void
  /** Start computing ahead (background producers). `eager` ignores the lookahead. */
  start(options?: { readonly eager?: boolean }): void
  /** Stop computing ahead (the tape stays). */
  stop(): void
  /** The step the consumer is viewing, so a producer with a lookahead knows how far ahead it is. */
  follow(step: number): void
  /** A derived path of a variant at a step, whenever it is ready. */
  probeAsync(variant: number, t: number, path: string, args?: Json): Promise<Derived>
  /** What went wrong, once something has. */
  readonly error: string | undefined
  /** Milliseconds the last slice took. */
  readonly lastMs: number
  /** Counts the answers to probes that have arrived, so views of probes can ask again. */
  readonly probed: number
  /** Be told when the tape grew, a probe was answered or something failed. Returns the way to stop. */
  subscribe(listener: () => void): () => void
  /**
   * Drop everything it holds (the tape, a worker's thread: terminated); a producer used again after this starts from
   * nothing, at step 0.
   */
  dispose(): void
}

/** Makes the producer of a hook's sessions. */
export type ProducerFactory = <S, C extends Config>(
  method: Method<S, C>,
  options: SessionOptions<S, C>,
  variants: readonly Variant<C>[],
) => Producer<S, C>

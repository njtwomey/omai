// The hub entry of a playground pairing (private to collections/rl): the pairing's parameters as controls, a run
// that reports the episodes so far as it learns, the learning curve, and the last recorded episode played back.
import type { Exposition } from 'mdml-engine/exposition'
import { playground, playgroundPairing, replayRecorded } from 'mdml-engine/methods/rl'
import { defineEntry, type HubEntry, type MethodRef, type SlotContext } from '../../entry/entry.ts'

/** Options of {@link playgroundEntry}. */
export interface PlaygroundEntryOptions {
  readonly tags: readonly string[]
  /** The agent's own source, for the docs link. */
  readonly agent: MethodRef
  readonly references?: readonly string[]
}

/** The props of render's `EpisodePlayer` for the last recorded episode of a record: its exact replay. */
function lastEpisode(pairing: string): (record: Exposition, context: SlotContext) => Readonly<Record<string, unknown>> {
  return (record, context) => {
    const episode = record.kind === 'episodes' ? record.recorded?.at(-1) : undefined
    if (episode === undefined) return {}
    const replay = replayRecorded(pairing, episode, context.params)
    return { env: replay.env, policy: replay.policy, maxSteps: episode.actions.length }
  }
}

/**
 * The hub entry of a playground pairing.
 *
 * @param pairing - The pairing's id (`cliff-walking/q-learning`).
 * @param family - The hub collection under `rl/` (`gridworlds`).
 * @param name - The entry's name in that collection.
 * @param options - Tags, the agent's source and references.
 * @returns The entry.
 */
export function playgroundEntry(
  pairing: string,
  family: string,
  name: string,
  options: PlaygroundEntryOptions,
): HubEntry {
  const p = playgroundPairing(pairing)
  return defineEntry({
    id: `rl/${family}/${name}`,
    title: p.title,
    summary: p.summary,
    collection: `rl/${family}`,
    tags: ['reinforcement learning', ...options.tags],
    datasets: [],
    params: p.hyperSpace,
    run: ({ params, key, progress }) => {
      const episodes = typeof params['episodes'] === 'number' ? params['episodes'] : 100
      // About fifty reports a run; the widget throttles them further.
      const every = Math.max(1, Math.floor(episodes / 50))
      return playground({
        pairing,
        params,
        key,
        ...(progress === undefined
          ? {}
          : {
              onEpisode: (e) => {
                if (e.episode % every === 0)
                  progress({ fraction: e.fraction, label: `episode ${e.episode}`, exposition: [e.record()] })
              },
            }),
      })
    },
    views: [
      { record: 'episodes', title: 'Returns per episode' },
      {
        record: 'episodes',
        view: 'EpisodePlayer',
        props: lastEpisode(pairing),
        area: 'side',
        title: 'The last episode, played back',
      },
    ],
    methods: [{ file: 'methods/rl/playground.ts', name: 'playground' }, options.agent],
    ...(options.references === undefined ? {} : { references: options.references }),
  })
}

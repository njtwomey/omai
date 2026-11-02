/**
 * A tape file played back: `useTape` turns a loaded tape file into the same live handle the player and the panels take
 * from `useSession`, over sessions replayed from the file (the recorded events and probes, nothing computed), done
 * from the start so the cursor can go anywhere at once. A page that offers loading a file shows this handle in place
 * of its own.
 */

import { useMemo } from 'react'
import type { MethodRegistry } from '../../producer/registry.ts'
import { sharedHead } from '../../producer/step.ts'
import { replaySessions, type TapeFile } from '../../producer/tape.ts'
import type { LiveSessions, LiveVariant } from './useSession.ts'

/**
 * The live handle of a tape file.
 *
 * @param file - The tape file.
 * @param registry - The methods on offer; the file's method is looked up by name.
 * @returns A comparison of the file's variants (one for a single session), all done.
 * @throws MdmlError with code `value` when the registry does not know the file's method.
 */
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
export function replayLive(file: TapeFile, registry: MethodRegistry): LiveSessions<any, any> {
  const sessions = replaySessions(file, registry)
  const none = (): void => undefined
  const handle = {
    version: 0,
    computing: false,
    compute: none,
    halt: none,
    advance: none,
    lastMs: 0,
    stale: false,
    held: false,
    producer: { kind: 'inline' as const },
    probed: 0,
  }
  const apply = (): readonly (typeof sessions)[number]['session'][] => sessions.map((s) => s.session)
  const variants = sessions.map(
    // biome-ignore lint/suspicious/noExplicitAny: as above
    (s): LiveVariant<any, any> => ({
      label: s.label,
      live: { ...handle, session: s.session, head: s.session.t, done: true, apply: () => s.session },
    }),
  )
  return {
    ...handle,
    variants,
    head: sharedHead(variants.map((v) => v.live)),
    done: true,
    apply,
  }
}

/**
 * A loaded tape file as a live handle, or `undefined` while no file is loaded (so a page keeps its own session).
 *
 * @param file - The tape file, if one is loaded.
 * @param registry - The methods on offer.
 * @returns The handle, replayed from the file.
 * @index function React hook: replay a loaded tape file as a live handle: `useTape(file, registry)`
 */
// biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
export function useTape(file: TapeFile | undefined, registry: MethodRegistry): LiveSessions<any, any> | undefined {
  return useMemo(() => (file === undefined ? undefined : replayLive(file, registry)), [file, registry])
}

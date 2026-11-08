/**
 * A tape file played back: `useTape` turns a loaded tape file into the same live handle the player and the panels take
 * from `useSession`, over sessions replayed from the file (the recorded events and probes, nothing computed), done
 * from the start so the cursor can go anywhere at once. A page that offers loading a file shows this handle in place
 * of its own.
 */

import { useEffect, useMemo, useRef } from 'react'
import type { MethodRegistry } from '../../producer/registry.ts'
import { sharedHead } from '../../producer/step.ts'
import { replaySessions, type TapeFile } from '../../producer/tape.ts'
import { type LiveSessions, type LiveVariant, STILL_HANDLE } from './useSession.ts'

/**
 * The live handle of a tape file.
 *
 * @param file - The tape file.
 * @param registry - The methods on offer; the file's method is looked up by name.
 * @returns A comparison of the file's variants (one for a single session), all done.
 * @throws OmaiError with code `value` when the registry does not know the file's method.
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
    ...STILL_HANDLE,
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
 * A loaded tape file as a live handle, or `undefined` while no file is loaded (so a page keeps its own session). A file
 * that cannot be replayed (its method is not in the registry, say) also gives `undefined`, and `onError` is called with
 * the reason, so the page shows a message and keeps its own session instead of failing while it renders.
 *
 * @param file - The tape file, if one is loaded.
 * @param registry - The methods on offer.
 * @param onError - Called once per file that cannot be replayed, with why (an effect, not during render).
 * @returns The handle, replayed from the file.
 * @index function React hook: replay a loaded tape file as a live handle: `useTape(file, registry)`
 */
export function useTape(
  file: TapeFile | undefined,
  registry: MethodRegistry,
  onError?: (message: string) => void,
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
): LiveSessions<any, any> | undefined {
  const made = useMemo(() => {
    if (file === undefined) return undefined
    try {
      return { live: replayLive(file, registry) }
    } catch (e) {
      return { message: e instanceof Error ? e.message : String(e) }
    }
  }, [file, registry])
  const report = useRef(onError)
  report.current = onError
  useEffect(() => {
    if (made !== undefined && 'message' in made) report.current?.(made.message)
  }, [made])
  return made !== undefined && 'live' in made ? made.live : undefined
}

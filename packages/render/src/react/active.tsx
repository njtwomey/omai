/**
 * The ambient `active` flag: `ActiveProvider` says whether the figures below it may compute and play (a slide that is not
 * showing is inactive), and `useActive` reads it. The hooks that compute or run a clock (`useSession`, `useSessions`,
 * `useLiveRun`, `usePlayback`, `useJob`) and `SessionPlayer` take it when their own `active` option is not given, so a
 * deck wraps each slide once instead of passing the flag through every hook.
 */

import { createContext, type ReactElement, type ReactNode, useContext } from 'react'

const ActiveContext = createContext<boolean>(true)

/**
 * Say whether everything below may compute and play. The rule is a logical AND down the tree: a subtree is active only
 * when every `ActiveProvider` above it, and this one, say so, so an inactive deck keeps every slide inactive whatever the
 * slides say, and an inactive slide inside an active deck is inactive. With no provider the flag is `true`. A hook's own
 * `active` option, when given, replaces the ambient value (so a host can force a figure on or off).
 *
 * @param props - `active` (default true) and the children.
 * @returns The provider.
 * @index function React component: switch computing and playback of everything below on or off: `<ActiveProvider active={shown}>`
 * @example
 * ```tsx
 * <ActiveProvider active={slide === current}>
 *   <SessionPlayer live={live} at={at} onScrub={setAt} />
 * </ActiveProvider>
 * ```
 */
export function ActiveProvider(props: { readonly active?: boolean; readonly children?: ReactNode }): ReactElement {
  const parent = useContext(ActiveContext)
  return <ActiveContext.Provider value={parent && (props.active ?? true)}>{props.children}</ActiveContext.Provider>
}

/**
 * The ambient `active` flag: the AND of every {@link ActiveProvider} above (true with none). Given `explicit`, that
 * wins instead: it is how a hook resolves its own `active` option.
 *
 * @param explicit - A hook's own `active` option; when it is not `undefined` it replaces the ambient value.
 * @returns Whether the caller may compute and play.
 * @index function React hook: whether the figures here may compute and play: `useActive()`
 */
export function useActive(explicit?: boolean): boolean {
  const ambient = useContext(ActiveContext)
  return explicit ?? ambient
}

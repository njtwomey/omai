/**
 * The scale of the controls and players: the CSS variable `--omai-control-scale` (a number, default 1) that every size
 * of a control multiplies (heights, icon buttons, text, paddings, switch parts, the player's scrubber), `ControlScaleProvider`
 * to set it from React, and the hook that gives a popup drawn in a portal the same scale and the theme of the place it
 * was opened from. It replaces CSS `zoom` on the controls, which makes popups drawn outside the zoomed subtree the wrong size.
 */

import {
  type CSSProperties,
  createContext,
  type ReactElement,
  type ReactNode,
  type RefObject,
  useContext,
  useLayoutEffect,
  useMemo,
  useState,
} from 'react'
import { useThemeVariables } from '../theme.tsx'

/** The CSS custom property that scales the controls and players: a number, 1 when unset. */
export const CONTROL_SCALE_VAR = '--omai-control-scale'

const ScaleContext = createContext<number | undefined>(undefined)

/**
 * Scale the controls and players below by a factor: sets `--omai-control-scale` on one wrapper `div` (`display:
 * contents` by default, so the layout is unchanged). Popups of the controls (`Popover`, a select's menu) are drawn in a
 * portal outside the wrapper; they copy the variable onto themselves, so their text and padding follow. A host can set
 * the variable in its own CSS on any element instead (`.deck { --omai-control-scale: 1.3 }`), and popups read it from the
 * element they hang from. Use it instead of CSS `zoom`, which a portal escapes.
 *
 * @param props - The factor (1 is the standard size), the wrapper's `className` and `style`, and the children.
 * @returns The provider.
 * @index function React component: scale every control and player below by a factor: `<ControlScaleProvider scale={1.3}>`
 * @example
 * ```tsx
 * <ControlScaleProvider scale={1.3}>
 *   <NumberField label="k" value={k} onChange={setK} />
 * </ControlScaleProvider>
 * ```
 */
export function ControlScaleProvider(props: {
  readonly scale: number
  readonly className?: string
  readonly style?: CSSProperties
  readonly children?: ReactNode
}): ReactElement {
  return (
    <ScaleContext.Provider value={props.scale}>
      <div
        className={props.className}
        style={{ display: 'contents', [CONTROL_SCALE_VAR]: String(props.scale), ...props.style } as CSSProperties}
      >
        {props.children}
      </div>
    </ScaleContext.Provider>
  )
}

/**
 * The factor set by the nearest {@link ControlScaleProvider}, or `undefined` outside one (the standard size, or what the
 * host's CSS says).
 *
 * @returns The factor.
 */
export function useControlScale(): number | undefined {
  return useContext(ScaleContext)
}

/**
 * What a popup drawn in a portal copies from the place it was opened from, so that it looks as it would have inside:
 * the control scale (from the anchor's computed `--omai-control-scale`, else the nearest {@link ControlScaleProvider})
 * and the variables and `data-theme` of the enclosing `ThemeProvider`s.
 *
 * @param anchor - The element the popup hangs from, whose inherited scale is read when the popup opens.
 * @param open - Whether the popup is open.
 * @returns The `style` and `data-theme` for the popup element.
 */
export function usePortalStyle(
  anchor: RefObject<HTMLElement | null> | undefined,
  open: boolean,
): { readonly style: CSSProperties; readonly theme: string | undefined } {
  const provided = useControlScale()
  const { vars, mode } = useThemeVariables()
  const [inherited, setInherited] = useState<string | undefined>(undefined)
  useLayoutEffect(() => {
    if (!open) return
    const el = anchor?.current
    if (el === null || el === undefined || typeof getComputedStyle !== 'function') return
    const v = getComputedStyle(el).getPropertyValue(CONTROL_SCALE_VAR).trim()
    setInherited(v === '' ? undefined : v)
  }, [open, anchor])
  const scale = inherited ?? (provided === undefined ? undefined : String(provided))
  const style = useMemo(
    () => ({ ...vars, ...(scale === undefined ? {} : { [CONTROL_SCALE_VAR]: scale }) }) as CSSProperties,
    [vars, scale],
  )
  return { style, theme: mode }
}

/**
 * The width the controls of a row take when they are not given one: a row of controls (`Controls`, the `controls` of a
 * `FigureFrame`) sets it for the number fields and selects below it, so they line up without a `width` on each.
 */

import { type Context, createContext, useContext } from 'react'
import type { ControlWidth } from './classes.ts'

/** The width a row of controls gives the controls below it, or `undefined` outside a row. */
export const ControlWidthContext: Context<ControlWidth | undefined> = createContext<ControlWidth | undefined>(undefined)

/**
 * The width of a control: its own `width` prop, else the one its row of controls gives, else `undefined` (the control's
 * own default of a container-wide field between a minimum and a maximum).
 *
 * @param own - The control's own `width` prop.
 * @returns The width to use.
 */
export function useControlWidth(own: ControlWidth | undefined): ControlWidth | undefined {
  const ambient = useContext(ControlWidthContext)
  return own ?? ambient
}

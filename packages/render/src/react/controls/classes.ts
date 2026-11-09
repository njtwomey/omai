/**
 * Class names shared by the controls. The controls are styled with Tailwind utilities over the page's `--wb-*` CSS
 * variables (the theme tokens), so a page that defines the variables and lets Tailwind scan this package gets them
 * drawn in its colours, light or dark.
 */

import type { CSSProperties } from 'react'

/**
 * Join class names, skipping empty ones.
 *
 * @param parts - Class names; `false`, `null`, `undefined` and empty strings are skipped.
 * @returns The names joined by spaces.
 */
export function cx(...parts: readonly (string | false | null | undefined)[]): string {
  return parts.filter((p) => typeof p === 'string' && p !== '').join(' ')
}

/** The class of a control's label. */
export const labelClass: string = 'text-xs font-medium text-(--wb-muted) select-none'

/** The class of an outlined square icon button (step, cycle, play). */
export const iconButtonClass: string =
  'inline-flex size-8 shrink-0 items-center justify-center border border-(--wb-grid) bg-(--wb-bg) text-(--wb-fg) outline-none hover:bg-(--wb-surface) focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-40'

/** The class of a text input's box. */
export const inputBoxClass: string =
  'relative flex h-8 min-w-0 flex-1 items-center border border-(--wb-grid) bg-(--wb-bg) text-(--wb-fg) focus-within:z-10 focus-within:border-(--wb-accent) focus-within:ring-2 focus-within:ring-(--wb-accent)/40'

/** The class of a floating menu's panel (placed by `Popover`). */
export const menuClass: string =
  'z-[1200] max-h-64 min-w-24 overflow-y-auto rounded-md border border-(--wb-grid) bg-(--wb-bg) py-1 text-(--wb-fg) shadow-lg'

/** The class of one item of a floating menu (`data-selected` marks the current one). */
export const menuItemClass: string =
  'flex w-full cursor-pointer items-center px-2.5 py-1 text-left text-xs outline-none hover:bg-(--wb-surface) focus-visible:bg-(--wb-surface) data-[selected=true]:font-semibold data-[selected=true]:text-(--wb-accent)'

/** The size of a control: `sm` (compact, 28px), `md` (the default, 32px) or `lg` (touch-friendly, 40px). */
export type ControlSize = 'sm' | 'md' | 'lg'

/**
 * The width of a control: a number is pixels, a string is a CSS length (`'12rem'`, `'50%'`), `'fill'` takes the whole
 * width of its container and `'auto'` the natural width of its content. Not given, a control keeps its default of
 * filling its container between a minimum and a maximum width.
 */
export type ControlWidth = number | string

/** The classes one {@link ControlSize} changes: row height, icon button side, text size, and the switch's parts. */
export interface ControlSizeClasses {
  /** Height of a field, select or segmented box. */
  readonly height: string
  /** Side of a square icon button. */
  readonly icon: string
  /** Text size of the value and the label. */
  readonly text: string
  /** The track, the knob and the knob's travel of a switch. */
  readonly track: string
  readonly knob: string
  readonly knobOn: string
  readonly knobOff: string
}

/** The classes of each {@link ControlSize}, spelled out whole so Tailwind finds them. */
export const controlSizes: Readonly<Record<ControlSize, ControlSizeClasses>> = {
  sm: {
    height: 'h-7',
    icon: 'size-7',
    text: 'text-xs',
    track: 'h-[14px] w-[26px]',
    knob: 'size-2.5',
    knobOn: 'translate-x-[13px]',
    knobOff: 'translate-x-px',
  },
  md: {
    height: 'h-8',
    icon: 'size-8',
    text: 'text-xs',
    track: 'h-[18px] w-8',
    knob: 'size-3.5',
    knobOn: 'translate-x-[15px]',
    knobOff: 'translate-x-px',
  },
  lg: {
    height: 'h-10',
    icon: 'size-10',
    text: 'text-sm',
    track: 'h-[22px] w-10',
    knob: 'size-[18px]',
    knobOn: 'translate-x-[19px]',
    knobOff: 'translate-x-px',
  },
}

/**
 * The class of a control's label at a size, truncated to one line only when asked (the default wraps).
 *
 * @param size - The control's size.
 * @param truncate - Cut the label to one line with an ellipsis instead of wrapping.
 * @returns The class names.
 */
export function labelClassFor(size: ControlSize, truncate = false): string {
  return cx(
    controlSizes[size].text,
    'min-w-0 font-medium text-(--wb-muted) select-none',
    truncate ? 'truncate' : 'break-words',
  )
}

/**
 * The class of an outlined square icon button at a size.
 *
 * @param size - The control's size.
 * @returns The class names.
 */
export function iconButtonClassFor(size: ControlSize): string {
  return iconButtonClass.replace('size-8', controlSizes[size].icon)
}

/**
 * The class of a text input's box at a size.
 *
 * @param size - The control's size.
 * @returns The class names.
 */
export function inputBoxClassFor(size: ControlSize): string {
  return inputBoxClass.replace('h-8', controlSizes[size].height)
}

/**
 * The outer class and inline style that give a control its width: the control's own default when `width` is not given,
 * `w-full` for `'fill'`, the content's width for `'auto'` (never below 9rem), and an exact width for a number (pixels)
 * or a CSS length.
 *
 * @param width - The `width` prop.
 * @param fallback - The control's default width classes.
 * @returns The class names and, for an exact width, the style.
 * @example
 * ```ts
 * import { controlWidth } from 'omai-render/react'
 * controlWidth(176, 'w-full max-w-xs').style // => { width: 176 }
 * controlWidth('fill', 'w-full max-w-xs').className // => 'w-full'
 * ```
 */
export function controlWidth(
  width: ControlWidth | undefined,
  fallback: string,
): { readonly className: string; readonly style?: CSSProperties } {
  if (width === undefined) return { className: fallback }
  if (width === 'fill') return { className: 'w-full' }
  if (width === 'auto') return { className: 'w-fit min-w-36' }
  return { className: 'shrink-0', style: { width } }
}

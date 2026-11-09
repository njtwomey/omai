/**
 * Class names shared by the controls. The controls are styled with Tailwind utilities over the page's `--wb-*` CSS
 * variables (the theme tokens), so a page that defines the variables and lets Tailwind scan this package gets them
 * drawn in its colours, light or dark.
 *
 * Every size is written as `calc(<standard size> * var(--omai-control-scale, 1))`, so a host (or `ControlScaleProvider`) scales
 * the whole set with one number and, with the variable unset, the sizes are exactly the standard ones. The classes are spelled out
 * in full so Tailwind finds them.
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

/** The 12px text size with its line height, scaled by `--omai-control-scale`. */
export const textXsClass: string =
  'text-[length:calc(0.75rem*var(--omai-control-scale,1))]/[calc(1rem*var(--omai-control-scale,1))]'

/** The 12px text size without a line height (for text that sets its own), scaled by `--omai-control-scale`. */
export const text12Class: string = 'text-[length:calc(0.75rem*var(--omai-control-scale,1))]'

/** The 11px text size, scaled by `--omai-control-scale`. */
export const text11Class: string = 'text-[length:calc(11px*var(--omai-control-scale,1))]'

/** The class of a control's label. */
export const labelClass: string = `${textXsClass} font-medium text-(--wb-muted) select-none`

/** The class of an outlined square icon button (step, cycle, play). */
export const iconButtonClass: string =
  'inline-flex size-[calc(2rem*var(--omai-control-scale,1))] shrink-0 items-center justify-center border border-(--wb-grid) bg-(--wb-bg) text-(--wb-fg) outline-none hover:bg-(--wb-surface) focus-visible:z-10 focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-40'

/** The class of a text input's box. */
export const inputBoxClass: string =
  'relative flex h-[calc(2rem*var(--omai-control-scale,1))] min-w-0 flex-1 items-center border border-(--wb-grid) bg-(--wb-bg) text-(--wb-fg) focus-within:z-10 focus-within:border-(--wb-accent) focus-within:ring-2 focus-within:ring-(--wb-accent)/40'

/**
 * The class of a floating menu's panel (placed by `Popover`). It is tall enough (18rem) for the 11 values of a number
 * field's dropdown with their padding and border, so the last one is not cut off; a longer list scrolls.
 */
export const menuClass: string =
  'z-[1200] max-h-[calc(18rem*var(--omai-control-scale,1))] min-w-[calc(6rem*var(--omai-control-scale,1))] overflow-y-auto rounded-md border border-(--wb-grid) bg-(--wb-bg) py-[calc(0.25rem*var(--omai-control-scale,1))] text-(--wb-fg) shadow-lg'

/** The class of one item of a floating menu (`data-selected` marks the current one). */
export const menuItemClass: string =
  'flex w-full cursor-pointer items-center px-[calc(0.625rem*var(--omai-control-scale,1))] py-[calc(0.25rem*var(--omai-control-scale,1))] text-left text-[length:calc(0.75rem*var(--omai-control-scale,1))]/[calc(1rem*var(--omai-control-scale,1))] outline-none hover:bg-(--wb-surface) focus-visible:bg-(--wb-surface) data-[selected=true]:font-semibold data-[selected=true]:text-(--wb-accent)'

/**
 * The size of a control: `sm` (compact, 28px), `md` (the default, 32px) or `lg` (touch-friendly, 40px), each times the
 * host's `--omai-control-scale` (default 1).
 */
export type ControlSize = 'sm' | 'md' | 'lg'

/**
 * The width of a control: a number is pixels, a string is a CSS length (`'12rem'`, `'50%'`), `'fill'` takes the whole
 * width of its container and `'auto'` the natural width of its content. Not given, a control keeps its default of
 * filling its container between a minimum and a maximum width, which follow `--omai-control-scale`. A number is the host's
 * explicit pixels and stays literal at any scale; a string such as `'calc(176px*var(--omai-control-scale,1))'` scales.
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
    height: 'h-[calc(1.75rem*var(--omai-control-scale,1))]',
    icon: 'size-[calc(1.75rem*var(--omai-control-scale,1))]',
    text: 'text-[length:calc(0.75rem*var(--omai-control-scale,1))]/[calc(1rem*var(--omai-control-scale,1))]',
    track: 'h-[calc(14px*var(--omai-control-scale,1))] w-[calc(26px*var(--omai-control-scale,1))]',
    knob: 'size-[calc(0.625rem*var(--omai-control-scale,1))]',
    knobOn: 'translate-x-[calc(13px*var(--omai-control-scale,1))]',
    knobOff: 'translate-x-px',
  },
  md: {
    height: 'h-[calc(2rem*var(--omai-control-scale,1))]',
    icon: 'size-[calc(2rem*var(--omai-control-scale,1))]',
    text: 'text-[length:calc(0.75rem*var(--omai-control-scale,1))]/[calc(1rem*var(--omai-control-scale,1))]',
    track: 'h-[calc(18px*var(--omai-control-scale,1))] w-[calc(2rem*var(--omai-control-scale,1))]',
    knob: 'size-[calc(0.875rem*var(--omai-control-scale,1))]',
    knobOn: 'translate-x-[calc(15px*var(--omai-control-scale,1))]',
    knobOff: 'translate-x-px',
  },
  lg: {
    height: 'h-[calc(2.5rem*var(--omai-control-scale,1))]',
    icon: 'size-[calc(2.5rem*var(--omai-control-scale,1))]',
    text: 'text-[length:calc(0.875rem*var(--omai-control-scale,1))]/[calc(1.25rem*var(--omai-control-scale,1))]',
    track: 'h-[calc(22px*var(--omai-control-scale,1))] w-[calc(2.5rem*var(--omai-control-scale,1))]',
    knob: 'size-[calc(18px*var(--omai-control-scale,1))]',
    knobOn: 'translate-x-[calc(19px*var(--omai-control-scale,1))]',
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
  return iconButtonClass.replace(controlSizes.md.icon, controlSizes[size].icon)
}

/**
 * The class of a text input's box at a size.
 *
 * @param size - The control's size.
 * @returns The class names.
 */
export function inputBoxClassFor(size: ControlSize): string {
  return inputBoxClass.replace(controlSizes.md.height, controlSizes[size].height)
}

/** The default width of a number field or a select: its container's width between 9rem and 20rem, times the control scale. */
export const fieldWidthClass: string =
  'w-full min-w-[calc(9rem*var(--omai-control-scale,1))] max-w-[calc(20rem*var(--omai-control-scale,1))]'

/** The default width of a segmented choice: its container's width between 9rem and 24rem, times the control scale. */
export const choiceWidthClass: string =
  'w-full min-w-[calc(9rem*var(--omai-control-scale,1))] max-w-[calc(24rem*var(--omai-control-scale,1))]'

/** The default width of a switch: at least 9rem and at most 20rem, times the control scale. */
export const toggleWidthClass: string =
  'min-w-[calc(9rem*var(--omai-control-scale,1))] max-w-[calc(20rem*var(--omai-control-scale,1))]'

/** The width of a control given `'auto'`: its content's, never below 9rem times the control scale. */
export const autoWidthClass: string = 'w-fit min-w-[calc(9rem*var(--omai-control-scale,1))]'

/**
 * The outer class and inline style that give a control its width: the control's own default when `width` is not given,
 * `w-full` for `'fill'`, the content's width for `'auto'` (never below 9rem), and an exact width for a number (pixels)
 * or a CSS length.
 *
 * The defaults and `'auto'` follow `--omai-control-scale` (the fallback classes are the scaled ones: see
 * {@link fieldWidthClass}). A number is the host's explicit pixels and stays literal at any scale; to give a width that
 * scales, pass a CSS length (`'calc(176px*var(--omai-control-scale,1))'`, as the `Controls` row does for its default).
 *
 * @param width - The `width` prop.
 * @param fallback - The control's default width classes.
 * @returns The class names and, for an exact width, the style.
 * @example
 * ```ts
 * import { controlWidth, fieldWidthClass } from 'omai-render/react'
 * controlWidth(176, fieldWidthClass).style // => { width: 176 }
 * controlWidth('fill', fieldWidthClass).className // => 'w-full'
 * ```
 */
export function controlWidth(
  width: ControlWidth | undefined,
  fallback: string,
): { readonly className: string; readonly style?: CSSProperties } {
  if (width === undefined) return { className: fallback }
  if (width === 'fill') return { className: 'w-full' }
  if (width === 'auto') return { className: autoWidthClass }
  return { className: 'shrink-0', style: { width } }
}

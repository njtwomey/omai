/**
 * Class names shared by the controls. The controls are styled with Tailwind utilities over the page's `--wb-*` CSS
 * variables (the theme tokens), so a page that defines the variables and lets Tailwind scan this package gets them
 * drawn in its colours, light or dark.
 */

/** Join class names, skipping empty ones. */
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
  'z-50 max-h-64 min-w-24 overflow-y-auto rounded-md border border-(--wb-grid) bg-(--wb-bg) py-1 text-(--wb-fg) shadow-lg'

/** The class of one item of a floating menu (`data-selected` marks the current one). */
export const menuItemClass: string =
  'flex w-full cursor-pointer items-center px-2.5 py-1 text-left text-xs outline-none hover:bg-(--wb-surface) focus-visible:bg-(--wb-surface) data-[selected=true]:font-semibold data-[selected=true]:text-(--wb-accent)'

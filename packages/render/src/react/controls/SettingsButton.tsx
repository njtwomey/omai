/**
 * A settings button: a gear that opens a modal holding a draft of some settings, and changes nothing until the modal is
 * confirmed. Edit as much as you like inside; Apply hands the whole draft to the page at once, Cancel (or Escape, or
 * the close button) throws it away. It sits in a figure's toolbar beside the export and expand buttons, or anywhere in
 * a control bar. `SpaceSettings` is the common case: the settings are an engine `Space` drawn by `SpaceControls`.
 */

import { type Space, type SpaceValue, defaults as spaceDefaults } from 'mdml-engine/space'
import { type ReactNode, useCallback, useEffect, useId, useReducer, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useFocusTrap } from './_focus.ts'
import { ControlPanel } from './ControlPanel.tsx'
import { cx, iconButtonClass } from './classes.ts'
import { CloseIcon, SettingsIcon } from './icons.tsx'
import { SpaceControls, type SpaceControlsProps } from './SpaceControls.tsx'
import { type SettingsState, sameSettings, settingsReducer } from './settings.ts'

/** Props of {@link SettingsButton}. */
export interface SettingsButtonProps<T> {
  /** The applied settings: the dialog opens on a copy of them. */
  readonly value: T
  /** Called with the whole draft when Apply is pressed. */
  readonly onApply: (value: T) => void
  /** The dialog's contents: given the draft and a way to change it (nothing outside the dialog changes until Apply). */
  readonly children: (draft: T, setDraft: (draft: T) => void) => ReactNode
  /** The dialog's title, also the button's label (default `Settings`). */
  readonly title?: string
  /** With these, a Reset button puts the draft back to them (still to be applied). */
  readonly defaults?: T
  readonly applyLabel?: string
  readonly cancelLabel?: string
  readonly disabled?: boolean
  /** `toolbar`: the small button of a figure's toolbar; `control`: the size of the other controls (default). */
  readonly appearance?: 'control' | 'toolbar'
  /** Whether two values are the same setting (default: structural equality). */
  readonly equals?: (a: T, b: T) => boolean
  readonly className?: string
}

const toolbarButtonClass: string =
  'inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md border border-(--wb-grid) bg-(--wb-surface) p-0 text-(--wb-muted) outline-none hover:text-(--wb-fg) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-40'

const dialogButtonClass: string =
  'inline-flex h-8 cursor-pointer items-center rounded-md border border-(--wb-grid) bg-(--wb-bg) px-3 text-xs font-medium text-(--wb-fg) outline-none hover:bg-(--wb-surface) focus-visible:ring-2 focus-visible:ring-(--wb-accent)/50 disabled:pointer-events-none disabled:opacity-40'

/**
 * A gear button that opens a modal of settings and changes nothing until the modal is confirmed: the dialog edits a
 * draft; Apply gives the whole draft to `onApply` and closes, Cancel, Escape and the close button drop it, and a press
 * on the backdrop closes it only when the draft is untouched. Opening always starts from the applied value. Apply is
 * disabled while the draft equals the applied value. Focus moves into the dialog, is kept inside it, and returns to the
 * button when it closes.
 *
 * @param props - The applied value, what to call on Apply, and a render function for the dialog's contents.
 * @returns The button, and the dialog while open.
 * @index function React component: a gear opening a modal applied only on confirm: `<SettingsButton value={v} onApply={setV}>...</SettingsButton>`
 */
export function SettingsButton<T>(props: SettingsButtonProps<T>): ReactNode {
  const {
    value,
    onApply,
    children,
    title = 'Settings',
    defaults,
    applyLabel = 'Apply',
    cancelLabel = 'Cancel',
    disabled = false,
    appearance = 'control',
    equals = sameSettings,
    className,
  } = props
  const [state, dispatch] = useReducer(settingsReducer<T>, { open: false, draft: value } as SettingsState<T>)
  const trigger = useRef<HTMLButtonElement>(null)
  const dialog = useRef<HTMLFormElement>(null)
  const titleId = useId()
  const dirty = !equals(state.draft, value)
  const close = useCallback((): void => {
    dispatch({ type: 'cancel' })
    trigger.current?.focus()
  }, [])
  const apply = (): void => {
    if (!dirty) return
    onApply(state.draft)
    dispatch({ type: 'apply' })
    trigger.current?.focus()
  }
  // Focus goes into the dialog, stays inside it and comes back to the button.
  useFocusTrap(state.open, dialog)
  useEffect(() => {
    if (!state.open) return
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== 'Escape') return
      // Only this dialog closes: a figure's expanded view behind it stays.
      e.stopPropagation()
      close()
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [state.open, close])
  return (
    <>
      <button
        ref={trigger}
        type="button"
        className={cx(appearance === 'toolbar' ? toolbarButtonClass : cx(iconButtonClass, 'rounded-md'), className)}
        aria-label={title}
        title={title}
        aria-haspopup="dialog"
        aria-expanded={state.open}
        disabled={disabled}
        onClick={() => dispatch({ type: 'open', value })}
      >
        <SettingsIcon size={appearance === 'toolbar' ? 16 : 18} />
      </button>
      {state.open && typeof document !== 'undefined'
        ? createPortal(
            // The backdrop sits above a figure's expanded view (z 1000) and below nothing; menus raised from inside the dialog
            // (a select's list) sit above it.
            <div
              className="fixed inset-0 z-[1100] flex items-center justify-center bg-black/50 p-4"
              onPointerDown={(e) => {
                if (e.target === e.currentTarget) dispatch({ type: 'backdrop', dirty })
              }}
            >
              <form
                ref={dialog}
                role="dialog"
                aria-modal="true"
                aria-labelledby={titleId}
                tabIndex={-1}
                onSubmit={(e) => {
                  e.preventDefault()
                  apply()
                }}
                className="flex max-h-full w-full max-w-xl flex-col gap-3 overflow-y-auto rounded-lg border border-(--wb-grid) bg-(--wb-bg) p-4 text-(--wb-fg) shadow-xl outline-none"
              >
                <div className="flex items-center justify-between gap-2">
                  <h2 id={titleId} className="m-0 text-sm font-semibold">
                    {title}
                  </h2>
                  <button
                    type="button"
                    className={cx(iconButtonClass, 'rounded-md')}
                    aria-label="Close"
                    onClick={close}
                  >
                    <CloseIcon size={16} />
                  </button>
                </div>
                <div className="flex flex-col gap-3">
                  {children(state.draft, (draft) => dispatch({ type: 'edit', draft }))}
                </div>
                <div className="flex items-center gap-2 pt-1">
                  {defaults !== undefined && (
                    <button
                      type="button"
                      className={dialogButtonClass}
                      onClick={() => dispatch({ type: 'reset', defaults })}
                    >
                      Reset
                    </button>
                  )}
                  <span className="flex-1 text-xs text-(--wb-muted)" aria-live="polite">
                    {dirty ? `Changes are applied when you press ${applyLabel}.` : 'No changes.'}
                  </span>
                  <button type="button" className={dialogButtonClass} onClick={close}>
                    {cancelLabel}
                  </button>
                  <button
                    type="submit"
                    disabled={!dirty}
                    className={cx(
                      dialogButtonClass,
                      'border-(--wb-accent) bg-(--wb-accent) text-white hover:bg-(--wb-accent)/90',
                    )}
                  >
                    {applyLabel}
                  </button>
                </div>
              </form>
            </div>,
            document.body,
          )
        : null}
    </>
  )
}

/** Props of {@link SpaceSettings}. */
export interface SpaceSettingsProps<V extends SpaceValue>
  extends Omit<SettingsButtonProps<V>, 'children' | 'defaults'>,
    Pick<SpaceControlsProps<V>, 'space' | 'exclude' | 'admissible' | 'why' | 'segmentedUpTo'> {
  readonly space: Space
  /** Columns of the controls inside the dialog (default 2). */
  readonly columns?: number
  /** Content shown above the controls (an explanation of what the settings do). */
  readonly intro?: ReactNode
}

/**
 * A settings button for an engine `Space`: the dialog draws the space's controls on a draft of the value, with a Reset
 * to the space's defaults, and the value changes only on Apply. Conditional dimensions and unions behave as in
 * `SpaceControls`; `admissible` and `why` grey the options a page rules out.
 *
 * @param props - The space, the applied value and what to call on Apply.
 * @returns The button and its dialog.
 * @index function React component: gear button editing a Space in a modal: `<SpaceSettings space={s} value={v} onApply={setV} />`
 */
export function SpaceSettings<V extends SpaceValue>(props: SpaceSettingsProps<V>): ReactNode {
  const { space, columns = 2, intro, exclude, admissible, why, segmentedUpTo, ...button } = props
  return (
    <SettingsButton<V> {...button} defaults={spaceDefaults(space) as V}>
      {(draft, setDraft) => (
        <>
          {intro !== undefined && <p className="m-0 text-xs text-(--wb-muted)">{intro}</p>}
          <ControlPanel columns={columns}>
            <SpaceControls
              space={space}
              value={draft}
              onChange={setDraft}
              {...(exclude === undefined ? {} : { exclude })}
              {...(admissible === undefined ? {} : { admissible })}
              {...(why === undefined ? {} : { why })}
              {...(segmentedUpTo === undefined ? {} : { segmentedUpTo })}
            />
          </ControlPanel>
        </>
      )}
    </SettingsButton>
  )
}

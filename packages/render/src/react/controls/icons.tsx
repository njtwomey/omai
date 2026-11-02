/**
 * The small inline SVG icons the controls and player use: chevrons, play, pause, step by a frame (double chevrons), skip to either end and rewind,
 * a bolt, a die, copy, check, close, sliders and a settings gear, each sized by one prop.
 */

import type { ReactNode, SVGProps } from 'react'

/** Props of an icon: any SVG attribute; `size` sets width and height in pixels (default 16). */
export type IconProps = SVGProps<SVGSVGElement> & { readonly size?: number }

function Svg({ size = 16, children, ...props }: IconProps): ReactNode {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  )
}

/** A chevron pointing left (previous, decrease). */
export function ChevronLeftIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="m15 18-6-6 6-6" />
    </Svg>
  )
}

/** A chevron pointing right (next, increase). */
export function ChevronRightIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="m9 18 6-6-6-6" />
    </Svg>
  )
}

/** A chevron pointing down (open a menu). */
export function ChevronDownIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="m6 9 6 6 6-6" />
    </Svg>
  )
}

/** Play. */
export function PlayIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M7 4.5v15l12-7.5z" fill="currentColor" />
    </Svg>
  )
}

/** Pause. */
export function PauseIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor" />
      <rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor" />
    </Svg>
  )
}

/** Step back one frame: a double chevron pointing left. */
export function StepBackIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="m11 17-5-5 5-5" />
      <path d="m18 17-5-5 5-5" />
    </Svg>
  )
}

/** Step forward one frame: a double chevron pointing right. */
export function StepForwardIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="m6 17 5-5-5-5" />
      <path d="m13 17 5-5-5-5" />
    </Svg>
  )
}

/** Back to the start. */
export function RewindIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </Svg>
  )
}

/** A die (a new random seed). */
export function DiceIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <rect x="4" y="4" width="16" height="16" rx="3" />
      <circle cx="9" cy="9" r="1" fill="currentColor" />
      <circle cx="15" cy="15" r="1" fill="currentColor" />
      <circle cx="15" cy="9" r="1" fill="currentColor" />
      <circle cx="9" cy="15" r="1" fill="currentColor" />
    </Svg>
  )
}

/** Copy. */
export function CopyIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </Svg>
  )
}

/** A tick (done). */
export function CheckIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M20 6 9 17l-5-5" />
    </Svg>
  )
}

/** A cross (close). */
export function CloseIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M18 6 6 18M6 6l12 12" />
    </Svg>
  )
}

/** Sliders (a control panel). */
export function SlidersIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4" />
    </Svg>
  )
}

/** A gear (settings). */
export function SettingsIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </Svg>
  )
}

/** Skip to the start. */
export function SkipBackIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M21 5v14l-8-7z" fill="currentColor" />
      <path d="M13 5v14l-8-7z" fill="currentColor" />
      <path d="M3 5v14" />
    </Svg>
  )
}

/** Skip to the end. */
export function SkipForwardIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M3 5v14l8-7z" fill="currentColor" />
      <path d="M11 5v14l8-7z" fill="currentColor" />
      <path d="M21 5v14" />
    </Svg>
  )
}

/** A bolt (eager: all at once). */
export function ZapIcon(props: IconProps): ReactNode {
  return (
    <Svg {...props}>
      <path d="M13 2 3 14h9l-1 8 10-12h-9z" fill="currentColor" strokeWidth={1} />
    </Svg>
  )
}

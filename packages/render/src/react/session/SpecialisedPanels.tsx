/**
 * The dispatcher of the specialised panel kinds (zoomable picture, loss-landscape slice, stability region, equation
 * against truth, audio, transcript, samples over training, an inner session): each is a small file of its own, and the
 * session panels hand a kind here with the viewed step and the frame's title and file name.
 */

import type { ReactNode } from 'react'
import type { Panel } from './layout.ts'
import { EquationPanel, LandscapePanel, StabilityPanel } from './RecordPanels.tsx'
import type { LiveSession } from './useSession.ts'
import { ZoomPanel } from './ZoomPanel.tsx'

/** The kinds of panel this module draws. */
export type SpecialisedPanelSpec = Extract<Panel, { kind: 'zoom' | 'landscape' | 'stability' | 'equation' }>

/**
 * One specialised panel of a session at the viewed step.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function SpecialisedPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: SpecialisedPanelSpec
  readonly t: number
  readonly title: string
  readonly filename: string
}): ReactNode {
  const { panel, ...common } = props
  switch (panel.kind) {
    case 'zoom':
      return <ZoomPanel {...common} panel={panel} />
    case 'landscape':
      return <LandscapePanel {...common} panel={panel} />
    case 'stability':
      return <StabilityPanel {...common} panel={panel} />
    case 'equation':
      return <EquationPanel {...common} panel={panel} />
  }
}

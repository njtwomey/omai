/**
 * The React view of a transcript: the tokens as text in turns by role, each tinted by its weight on the chosen track
 * (a head's attention, say) with the token emitted at the viewed step outlined and marked, and the heaviest tokens
 * listed in words beneath.
 */

import type { Transcript } from 'omai-engine/exposition'
import { type ReactElement, useState } from 'react'
import { heaviestTokens, transcriptTurns } from '../../views/transcript.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { ToolButton } from '../session/parts.tsx'
import { type RecordViewProps, useViewFigure } from './_frame.tsx'

const DEFAULTS = { width: 440, height: 320 } as const

/**
 * A transcript with its highlight. The track is chosen with the buttons when the record has several; the emitted
 * token has an outline and a marker in front, and every token's weight is in its tooltip.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 */
export function TranscriptView(props: RecordViewProps<Transcript>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const [track, setTrack] = useState(0)
  const turns = transcriptTurns(record, track)
  const heavy = heaviestTokens(record, track)
  const tracks = record.tracks ?? []
  return (
    <FigureFrame
      figure={figure}
      filename={props.filename ?? 'transcript'}
      exportFormats={[]}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={
        <div className="flex flex-col gap-1">
          {tracks.length > 1 ? (
            <div
              className="flex flex-wrap items-center gap-1"
              role="toolbar"
              aria-label="Which weights tint the tokens"
            >
              {tracks.map((t, k) => (
                <ToolButton
                  key={t.name}
                  label={t.name}
                  title={`Tint the tokens by ${t.name}`}
                  pressed={k === track}
                  onClick={() => setTrack(k)}
                />
              ))}
            </div>
          ) : null}
          <span>
            {props.caption ?? record.caption ?? ''}
            {heavy.length === 0
              ? ''
              : ` Heaviest ${record.weightName ?? tracks[track]?.name ?? 'weight'}: ${heavy.map((h) => `“${h.text}” ${Number(h.weight.toPrecision(2))}`).join(', ')}.`}
          </span>
        </div>
      }
    >
      <div className="flex h-full flex-col gap-2 overflow-y-auto pr-1 text-sm leading-7" data-testid="transcript">
        {turns.map((turn, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: turns are positional and never reordered
          <div key={i} className="flex gap-2">
            {turn.role === undefined ? null : (
              <span className="mt-1 w-16 shrink-0 text-right text-xs font-semibold text-muted uppercase">
                {turn.role}
              </span>
            )}
            <p className="m-0 min-w-0 whitespace-pre-wrap">
              {turn.spans.map((s, k) => (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: tokens are positional
                  key={k}
                  title={s.title}
                  data-emitted={s.emitted ? 'true' : undefined}
                  className={s.emitted ? 'rounded-sm font-semibold outline-2 outline-(--wb-fg)' : 'rounded-sm'}
                  style={{
                    background:
                      s.level > 0
                        ? `color-mix(in srgb, var(--wb-accent) ${Math.round(s.level * 70)}%, transparent)`
                        : undefined,
                  }}
                >
                  {s.emitted ? '▸' : ''}
                  {s.text}
                </span>
              ))}
            </p>
          </div>
        ))}
      </div>
    </FigureFrame>
  )
}

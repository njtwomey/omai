import { type ReactElement, useCallback, useMemo } from 'react'
import type { ComponentsRecord } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type TopicWordsViewOptions, topicWordsView } from '../../views/topics.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * Topics by their top words (a components record whose features are words: LDA's topics, NMF's parts): a small bar
 * chart of each topic's most probable words.
 *
 * @param props - The record, words per topic, columns and topic names, and the figure settings.
 * @returns The figure.
 */
export function TopicWordsView(
  props: RecordViewProps<ComponentsRecord> & Omit<TopicWordsViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const { record, top, columns, names } = props
  const defaults = useMemo(() => {
    const k = Math.max(1, record.components.length)
    const c = Math.min(k, columns ?? Math.min(4, k))
    return { aspect: Math.min(2.2, Math.max(1.1, (c * 1.25) / Math.ceil(k / c))) }
  }, [record, columns])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) =>
      topicWordsView(record, {
        figure: fig,
        ...(top === undefined ? {} : { top }),
        ...(columns === undefined ? {} : { columns }),
        ...(names === undefined ? {} : { names }),
      }),
    [record, top, columns, names],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'topics'} />
}

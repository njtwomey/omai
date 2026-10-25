/**
 * React views of rule and concept records: a rule list and TCAV scores per concept.
 */

import { type ReactElement, useCallback, useMemo } from 'react'
import type { ConceptScores, RuleSet } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { conceptView, ruleListView } from '../../views/rules.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const CONCEPT_FIGURE = { aspect: 1.6 } as const

/**
 * A rule list: each rule's text beside its importance (coloured by its coefficient's sign), or an anchor's precision
 * and coverage.
 *
 * @param props - The record, how many rules, and the figure settings.
 * @returns The figure.
 */
export function RuleListView(props: RecordViewProps<RuleSet> & { readonly maxRules?: number }): ReactElement {
  const { record, maxRules } = props
  const defaults = useMemo(() => {
    const n = Math.min(maxRules ?? 12, record.rules.length)
    const anchors = record.rules.every((r) => r.precision !== undefined)
    return { height: Math.max(180, 80 + (anchors ? 44 : 28) * n) }
  }, [record, maxRules])
  const figure = useViewFigure(props.figure, defaults)
  const build = useCallback(
    (fig: Figure) => ruleListView(record, { figure: fig, ...(maxRules === undefined ? {} : { maxRules }) }),
    [record, maxRules],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'rules'} />
}

/**
 * TCAV scores per concept against random directions, each run a dot, significant concepts starred.
 *
 * @param props - The record and the figure settings.
 * @returns The figure.
 */
export function ConceptView(props: RecordViewProps<ConceptScores>): ReactElement {
  const { record } = props
  const figure = useViewFigure(props.figure, CONCEPT_FIGURE)
  const build = useCallback((fig: Figure) => conceptView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'tcav'} />
}

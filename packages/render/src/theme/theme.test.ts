import { palettes } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { axisStyle, baseOption, echartsTheme, themeName } from './echarts.ts'
import {
  formatDuration,
  formatNumber,
  formatPercent,
  formatPower,
  formatTime,
  logTicks,
  niceNumber,
  niceStep,
  niceTicks,
  stepDecimals,
} from './format.ts'
import {
  categoricalColour,
  colourBlindCheck,
  colourDistance,
  colourScale,
  contrastRatio,
  divergingRamp,
  interpolateColours,
  mixColours,
  parseColour,
  relativeLuminance,
  sampleRamp,
  scaleFraction,
  sequentialRamp,
  simulateColourVision,
  tab10,
  toHex,
} from './palettes.ts'
import { cssText, cssVariables, isThemeToken, paintRgba, resolvePaint, tokens } from './tokens.ts'

describe('tokens', () => {
  it('agree with the engine palettes for every scene token', () => {
    for (const mode of ['light', 'dark'] as const)
      for (const [k, v] of Object.entries(palettes[mode])) expect(tokens(mode)[k as 'fg']).toBe(v)
  })
  it('resolve paints and css variables', () => {
    expect(resolvePaint('accent', 'dark')).toBe(palettes.dark.accent)
    expect(resolvePaint('none')).toBe('none')
    expect(cssVariables('dark')['--wb-fg-secondary']).toBe(tokens('dark').fgSecondary)
    expect(cssText('light')).toContain('--wb-tab9: #17becf;')
    expect(isThemeToken('grid')).toBe(true)
    expect(isThemeToken('toString')).toBe(false)
    expect(paintRgba('none')).toBeUndefined()
  })
  it('keep text readable on the background', () => {
    for (const mode of ['light', 'dark'] as const) {
      const t = tokens(mode)
      expect(contrastRatio(t.fg, t.bg)).toBeGreaterThan(10)
      expect(contrastRatio(t.muted, t.bg)).toBeGreaterThan(3)
    }
  })
})

describe('palettes', () => {
  it('cycle tab10', () => {
    expect(tab10).toHaveLength(10)
    expect(categoricalColour(10)).toBe(tab10[0])
    expect(categoricalColour(-1)).toBe(tab10[9])
  })
  it('parse, print and mix colours', () => {
    expect(parseColour('#abc')).toEqual([170, 187, 204, 1])
    expect(toHex(parseColour('rgba(1, 2, 3, 0.5)'))).toBe('#01020380')
    expect(mixColours('#ff0000', '#0000ff', 0)).toBe('#ff0000')
    expect(() => parseColour('none')).toThrow()
  })
  it('interpolate ramps and scales', () => {
    expect(interpolateColours(['#000000', '#ffffff'], Number.NaN)).toBe('#000000')
    expect(interpolateColours(['#000000', '#ffffff'], 2)).toBe('#ffffff')
    expect(sampleRamp(sequentialRamp('viridis'), 9)).toEqual([...sequentialRamp('viridis')])
    expect(sampleRamp(['#000', '#fff'], 0)).toEqual([])
    const s = colourScale(divergingRamp(), [-1, 0, 1])
    expect(s(0)).toBe('#f7f7f7')
    expect(s(-1)).toBe('#2166ac')
    expect(scaleFraction(3, [3, 3])).toBe(0.5)
    expect(scaleFraction(-2, [-1, 0, 4])).toBe(0)
  })
  it('make sequential ramps lighten or darken monotonically', () => {
    for (const mode of ['light', 'dark'] as const)
      for (const name of ['viridis', 'magma', 'blues', 'greys'] as const) {
        const l = sequentialRamp(name, mode).map(relativeLuminance)
        const up = l.every((v, i) => i === 0 || v >= (l[i - 1] ?? 0))
        const down = l.every((v, i) => i === 0 || v <= (l[i - 1] ?? 0))
        expect(up || down).toBe(true)
      }
  })
  it('simulate colour vision and measure distinguishability', () => {
    expect(simulateColourVision('#000000', 'protanopia')).toBe('#000000')
    expect(colourDistance('#123456', '#123456')).toBe(0)
    const report = colourBlindCheck(tab10.slice(0, 4))
    expect(report.normal).toBeGreaterThan(report.deuteranopia)
    expect(report.worstPair[0]).toBeLessThan(report.worstPair[1])
    expect(colourBlindCheck(['#ff0000', '#00ff00'], 200).safe).toBe(false)
  })
})

describe('format', () => {
  it('formats numbers', () => {
    expect(formatNumber(0)).toBe('0')
    expect(formatNumber(-0)).toBe('0')
    expect(formatNumber(1 / 3)).toBe('0.3333')
    expect(formatNumber(-2, { unicodeMinus: true })).toBe('−2')
    expect(formatNumber(Number.NaN)).toBe('NaN')
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe('−∞')
    expect(formatNumber(1e-7)).toBe('1e-7')
    expect(formatPower(1e5)).toBe('10⁵')
    expect(formatPower(-1)).toBe('-1')
    expect(formatPercent(Number.NaN)).toBe('NaN')
  })
  it('formats durations and times', () => {
    expect(formatDuration(12.5)).toBe('12.5 s')
    expect(formatDuration(7500)).toBe('2 h 5 min')
    expect(formatDuration(-0.5)).toBe('-500 ms')
    expect(formatTime(Date.UTC(2024, 0, 1, 1, 2, 3), 1000)).toBe('01:02:03')
  })
  it('finds nice ticks', () => {
    expect(niceNumber(0)).toBe(1)
    expect(niceTicks(5, 5).ticks.length).toBeGreaterThan(1)
    expect(niceTicks(10, 0).range).toEqual([0, 10])
    expect(niceTicks(Number.NaN, 1).ticks).toEqual([])
    const t = niceTicks(-0.37, 2.1)
    expect(t.range[0]).toBeLessThanOrEqual(-0.37)
    expect(t.range[1]).toBeGreaterThanOrEqual(2.1)
    expect(logTicks(-1, 10)).toEqual({ major: [], minor: [] })
    expect(logTicks(1e-10, 1e10).major.length).toBeLessThanOrEqual(8)
    expect(niceStep(0, 0)).toBe(1)
    expect(stepDecimals(2.5e-7)).toBe(8)
    expect(stepDecimals(0)).toBe(0)
  })
})

describe('echarts theme', () => {
  it('styles both modes from tokens', () => {
    expect(themeName('light')).toBe('omai-light')
    expect(echartsTheme('dark').backgroundColor).toBe('transparent')
    expect(axisStyle('dark', 14).nameTextStyle.fontSize).toBe(14)
    expect(baseOption('light').color).toEqual(tab10)
    // ECharts describes every chart to a screen reader unless a view gives its own text.
    expect(echartsTheme('light')['aria']).toEqual({ enabled: true })
  })
})

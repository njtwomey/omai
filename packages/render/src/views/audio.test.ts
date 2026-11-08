import { audioClip } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { audioLayout, audioSceneAt, spectrogramOf, spectrogramPixels, waveformEnvelope } from './audio.ts'

const tone = (hz: number, n: number, rate = 8000): number[] =>
  Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * hz * i) / rate))

describe('the audio views', () => {
  it('finds a pure tone in the bin of its frequency in every window', () => {
    const s = spectrogramOf(tone(1000, 1024), 8000, { window: 128, hop: 64 })
    expect(s.frequencies.length).toBe(65)
    expect(s.frequencies[64]).toBe(4000)
    for (const k of [0, 5, s.times.length - 1]) {
      const column = s.magnitude.map((row) => row[k] ?? 0)
      expect(s.frequencies[column.indexOf(Math.max(...column))]).toBe(1000)
    }
    expect(s.times[0]).toBeCloseTo(64 / 8000, 12)
  })

  it('takes the envelope of a waveform per column', () => {
    expect(waveformEnvelope([0, 1, -1, 0.5], 2)).toEqual([
      [0, 1],
      [-1, 0.5],
    ])
    expect(waveformEnvelope([], 2)).toEqual([
      [0, 0],
      [0, 0],
    ])
  })

  it('draws the spectrogram with low frequencies at the bottom and the peak at the top of the ramp', () => {
    const p = spectrogramPixels({ times: [0], frequencies: [0, 1], magnitude: [[1], [0.001]] }, 'dark')
    expect([p.width, p.height]).toEqual([1, 2])
    // Row 0 of the picture is the highest frequency, which is quiet here: darker than the loud bottom row.
    const lum = (r: number): number => (p.rgba[4 * r] ?? 0) + (p.rgba[4 * r + 1] ?? 0) + (p.rgba[4 * r + 2] ?? 0)
    expect(lum(1)).toBeGreaterThan(lum(0))
  })

  it('places the playhead of a step on the plot, from the frames, and says so in the title', () => {
    const clip = audioClip({
      sampleRate: 8000,
      samples: tone(500, 8000),
      frames: [
        { step: 0, time: 0 },
        { step: 10, time: 1 },
      ],
    })
    const L = audioLayout()
    const at = (step: number) => audioSceneAt(clip, step)
    expect(at(5).title).toBe('Waveform and spectrogram of a 1 s clip, playhead at 0.5 s')
    const playheadX = (step: number): number =>
      at(step).children.flatMap((n) => (n.kind === 'line' && n.stroke === 'tab3' ? [n.x1] : []))[0] ?? Number.NaN
    expect(playheadX(0)).toBe(L.left)
    expect(playheadX(5)).toBeCloseTo(L.left + L.plot / 2, 9)
    expect(playheadX(10)).toBeCloseTo(L.left + L.plot, 9)
  })
})

import { useState } from 'react'
import type { PresetInfo, SimulationMeta, SimulationStatus, CrisisEvent } from '../hooks/useSimulation'

type MapMode = 'routing' | 'simulation'

interface SimulationPanelProps {
  presets: PresetInfo[]
  simulations: SimulationMeta[]
  status: SimulationStatus
  activeMeta: SimulationMeta | null
  showSimulation: boolean
  onToggleSimulation: () => void
  onRunPreset: (key: string) => void
  onRunCustom: (events: CrisisEvent[], name?: string, description?: string) => void
  onLoadSaved: (filename: string) => void
  onClear: () => void
  propDepth: number
  propDecay: number
  onPropDepthChange: (v: number) => void
  onPropDecayChange: (v: number) => void
  activeEvents: CrisisEvent[]
  onAddEvent: (event: CrisisEvent) => void
  onRemoveEvent: (index: number) => void
  onUpdateEvent: (index: number, event: CrisisEvent) => void
  mapMode: MapMode
  placingEventType: string
  onPlacingEventTypeChange: (type: string) => void
  defaultSpeed: number
  onDefaultSpeedChange: (v: number) => void
  onSaveAsPreset: (name: string, description: string) => void
  onDeletePreset: (key: string) => void
}

const EVENT_ICONS: Record<string, string> = { construction: '🚧', congestion: '🐢', damage: '💥' }
const EVENT_LABELS: Record<string, string> = { construction: 'Construction', congestion: 'Congestion', damage: 'Damage' }

function formatDate(ts: string) {
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

function SimulationPanel({
  presets, simulations, status, activeMeta,
  showSimulation, onToggleSimulation,
  onRunPreset, onLoadSaved, onClear,
  propDepth, propDecay, onPropDepthChange, onPropDecayChange,
  activeEvents, onRemoveEvent, onUpdateEvent,
  mapMode, placingEventType, onPlacingEventTypeChange,
  defaultSpeed, onDefaultSpeedChange,
  onSaveAsPreset, onDeletePreset,
}: SimulationPanelProps) {
  const isLoading = status === 'loading'
  const isPreviewing = status === 'previewing'
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null)
  const [presetName, setPresetName] = useState('')
  const [showSaveForm, setShowSaveForm] = useState(false)

  const builtInPresets = presets.filter(p => !p.user_created)
  const userPresets = presets.filter(p => p.user_created)

  return (
    <div>
      <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '10px 0' }} />
      <h3 style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        Simulation
        {isPreviewing && <span style={{ fontSize: 11, color: '#6b7280', fontWeight: 400 }}>updating...</span>}
      </h3>

      {/* Propagation sliders */}
      <div className="info-item">
        <label style={{ display: 'block', marginBottom: 2 }}>
          Propagation depth: <strong>{propDepth} hops</strong>
        </label>
        <input type="range" min={0} max={6} step={1} value={propDepth}
          onChange={e => onPropDepthChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }} disabled={isLoading} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
          <span>0 (none)</span><span>6 (wide)</span>
        </div>
      </div>

      <div className="info-item">
        <label style={{ display: 'block', marginBottom: 2 }}>
          Decay per hop: <strong>{Math.round(propDecay * 100)}%</strong>
          <span style={{ color: '#888', fontSize: 11, marginLeft: 4 }}>
            ({propDecay <= 0.4 ? 'sharp' : propDecay <= 0.6 ? 'moderate' : 'wide'})
          </span>
        </label>
        <input type="range" min={0.3} max={0.8} step={0.05} value={propDecay}
          onChange={e => onPropDecayChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }} disabled={isLoading} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
          <span>30%</span><span>80%</span>
        </div>
      </div>

      <div className="info-item">
        <label style={{ display: 'block', marginBottom: 2 }}>
          Event speed: <strong>{defaultSpeed}%</strong>
          <span style={{ color: '#888', fontSize: 11, marginLeft: 4 }}>
            {defaultSpeed === 0 ? '(impassable)' : defaultSpeed === 100 ? '(normal)' : `(${100 - defaultSpeed}% slower)`}
          </span>
        </label>
        <input type="range" min={0} max={100} step={5} value={defaultSpeed}
          onChange={e => onDefaultSpeedChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
          disabled={isLoading || placingEventType === 'damage'} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
          <span>0% (closed)</span><span>100% (normal)</span>
        </div>
        {placingEventType === 'damage' && (
          <div style={{ fontSize: 10, color: '#dc2626' }}>Damage is always impassable</div>
        )}
      </div>

      {/* Active simulation banner */}
      {activeMeta && (
        <div className="sim-active-banner">
          <strong style={{ color: '#92400e' }}>{activeMeta.scenario_name}</strong>
          {activeMeta.is_preview && <span style={{ fontSize: 10, color: '#888', marginLeft: 4 }}>(preview)</span>}
          <br />
          {activeMeta.scenario_description && (
            <span style={{ color: '#555', fontSize: 11 }}>{activeMeta.scenario_description}<br /></span>
          )}
          <span style={{ color: '#333', fontSize: 11 }}>
            {activeMeta.direct_edges ?? activeMeta.total_edges_affected} direct
            {activeMeta.propagated_edges != null && ` + ${activeMeta.propagated_edges} propagated`}
            {' '}edges
          </span>
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <button onClick={onToggleSimulation} style={{ marginTop: 0, fontSize: 11, padding: '3px 8px' }}>
              {showSimulation ? 'Hide' : 'Show'}
            </button>
            <button onClick={onClear} style={{ marginTop: 0, fontSize: 11, padding: '3px 8px', background: '#ef4444' }}>
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Active events list */}
      {activeEvents.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div className="info-item" style={{ color: '#555', marginBottom: 4, fontWeight: 600 }}>
            Events ({activeEvents.length}):
          </div>
          {activeEvents.map((ev, i) => (
            <div key={i} style={{
              fontSize: 11, padding: '4px 6px', marginBottom: 3,
              background: ev.speed_reduction_pct === 0 ? '#fee2e2' : '#fef9c3',
              borderRadius: 3,
              border: `1px solid ${ev.speed_reduction_pct === 0 ? '#fca5a5' : '#fde68a'}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span>{EVENT_ICONS[ev.type] ?? '?'}</span>
                <span style={{ flex: 1, cursor: 'pointer' }} onClick={() => setExpandedEvent(expandedEvent === i ? null : i)}>
                  {ev.description || EVENT_LABELS[ev.type]}
                  <span style={{ color: '#888', marginLeft: 4 }}>
                    r={ev.radius_m}m · {ev.speed_reduction_pct === 0 ? 'impassable' : `${ev.speed_reduction_pct}%`}
                  </span>
                </span>
                <button onClick={() => onRemoveEvent(i)}
                  style={{ marginTop: 0, padding: '1px 5px', fontSize: 10, background: '#fca5a5', border: 'none', borderRadius: 3, cursor: 'pointer', color: '#7f1d1d' }}>✕</button>
              </div>
              {expandedEvent === i && (
                <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid #e5e7eb' }}>
                  <div style={{ marginBottom: 4 }}>
                    <label style={{ fontSize: 10, color: '#666' }}>Speed: <strong>{ev.speed_reduction_pct}%</strong></label>
                    <input type="range" min={0} max={100} step={5} value={ev.speed_reduction_pct}
                      disabled={ev.type === 'damage'}
                      onChange={e => onUpdateEvent(i, { ...ev, speed_reduction_pct: Number(e.target.value) })}
                      style={{ width: '100%', cursor: ev.type === 'damage' ? 'not-allowed' : 'pointer' }} />
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: '#666' }}>Radius: <strong>{ev.radius_m}m</strong></label>
                    <input type="range" min={50} max={3000} step={50} value={ev.radius_m}
                      onChange={e => onUpdateEvent(i, { ...ev, radius_m: Number(e.target.value) })}
                      style={{ width: '100%', cursor: 'pointer' }} />
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Save as preset */}
          {!showSaveForm ? (
            <button onClick={() => setShowSaveForm(true)}
              style={{ marginTop: 4, fontSize: 11, padding: '4px 8px', background: '#8b5cf6' }}>
              Save as preset
            </button>
          ) : (
            <div style={{
              marginTop: 4, padding: '6px 8px', background: '#f3f4f6',
              borderRadius: 4, border: '1px solid #d1d5db',
            }}>
              <input type="text" placeholder="Preset name..." value={presetName}
                onChange={e => setPresetName(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && presetName.trim()) {
                    onSaveAsPreset(presetName.trim(), '')
                    setPresetName(''); setShowSaveForm(false)
                  }
                }}
                style={{
                  width: '100%', padding: '4px 6px', fontSize: 12,
                  border: '1px solid #ccc', borderRadius: 3,
                  fontFamily: 'Arial, sans-serif', boxSizing: 'border-box',
                }}
                autoFocus />
              <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                <button onClick={() => {
                  if (presetName.trim()) { onSaveAsPreset(presetName.trim(), ''); setPresetName(''); setShowSaveForm(false) }
                }} disabled={!presetName.trim()}
                  style={{ marginTop: 0, fontSize: 11, padding: '3px 8px', background: '#22c55e' }}>Save</button>
                <button onClick={() => { setShowSaveForm(false); setPresetName('') }}
                  style={{ marginTop: 0, fontSize: 11, padding: '3px 8px', background: '#9ca3af' }}>Cancel</button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Event type picker */}
      {mapMode === 'simulation' && (
        <div className="info-item" style={{ marginBottom: 8 }}>
          <div style={{ background: '#f0f9ff', border: '1px solid #bfdbfe', borderRadius: 4, padding: '6px 8px', fontSize: 12 }}>
            <strong>Click map to place events</strong>
            <div style={{ display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap' }}>
              {(['construction', 'congestion', 'damage'] as const).map(type => (
                <button key={type} onClick={() => onPlacingEventTypeChange(type)}
                  style={{
                    marginTop: 0, fontSize: 11, padding: '3px 8px',
                    background: placingEventType === type ? '#2563eb' : '#3388ff',
                    outline: placingEventType === type ? '2px solid #1d4ed8' : 'none',
                    fontWeight: placingEventType === type ? 700 : 400,
                  }}>
                  {EVENT_ICONS[type]} {EVENT_LABELS[type]}
                  {type === 'damage' && ' (impassable)'}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {mapMode !== 'simulation' && (
        <div className="info-item" style={{ color: '#888', fontSize: 11, marginBottom: 4 }}>
          Switch to Simulation mode to place events
        </div>
      )}

      <hr style={{ border: 'none', borderTop: '1px solid #f3f4f6', margin: '8px 0' }} />

      {/* Built-in presets */}
      {builtInPresets.length > 0 && (
        <>
          <div className="info-item" style={{ color: '#555', marginBottom: 4 }}>Built-in scenarios:</div>
          {builtInPresets.map(p => (
            <button key={p.key} onClick={() => onRunPreset(p.key)} disabled={isLoading}
              title={p.description} className="sim-preset-btn">
              {p.name}
            </button>
          ))}
        </>
      )}

      {/* User presets */}
      {userPresets.length > 0 && (
        <>
          <div className="info-item" style={{ color: '#555', marginBottom: 4, marginTop: 8 }}>Your scenarios:</div>
          {userPresets.map(p => (
            <div key={p.key} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
              <button onClick={() => onRunPreset(p.key)} disabled={isLoading}
                title={p.description} className="sim-preset-btn" style={{ flex: 1 }}>
                {p.name}
                <span style={{ color: 'rgba(255,255,255,0.7)', marginLeft: 4, fontSize: 10 }}>
                  ({p.event_count} events)
                </span>
              </button>
              <button onClick={() => onDeletePreset(p.key)} title="Delete preset"
                style={{
                  marginTop: 0, padding: '4px 6px', fontSize: 10,
                  background: '#fca5a5', color: '#7f1d1d',
                  border: 'none', borderRadius: 3, cursor: 'pointer',
                }}>✕</button>
            </div>
          ))}
        </>
      )}

      {/* Simulation history */}
      {simulations.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="info-item" style={{ color: '#555', marginBottom: 4 }}>History:</div>
          <select onChange={e => { if (e.target.value) onLoadSaved(e.target.value) }}
            defaultValue="" style={{ width: '100%', maxWidth: '100%' }}>
            <option value="">— select —</option>
            {[...simulations].reverse().map(s => (
              <option key={s.filename} value={s.filename ?? ''}>
                {s.scenario_name} · {formatDate(s.timestamp)}
              </option>
            ))}
          </select>
        </div>
      )}

      {status === 'error' && (
        <div className="info-item" style={{ color: '#dc2626', marginTop: 4 }}>Simulation failed</div>
      )}
    </div>
  )
}

export default SimulationPanel
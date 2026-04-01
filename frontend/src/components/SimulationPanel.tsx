import type { PresetInfo, SimulationMeta, SimulationStatus } from '../hooks/useSimulation'

interface SimulationPanelProps {
  presets: PresetInfo[]
  simulations: SimulationMeta[]
  status: SimulationStatus
  activeMeta: SimulationMeta | null
  showSimulation: boolean
  onToggleSimulation: () => void
  onRunPreset: (key: string) => void
  onLoadSaved: (filename: string) => void
  onClear: () => void
  propDepth: number
  propDecay: number
  onPropDepthChange: (v: number) => void
  onPropDecayChange: (v: number) => void
}

const SIM_ICONS: Record<string, string> = {
  bridge_collapse:        '💥',
  city_centre_congestion: '🐢',
  multi_closure:          '🚫',
  eastern_district_damage:'💥',
}

function formatDate(ts: string) {
  try { return new Date(ts).toLocaleString() } catch { return ts }
}

function SimulationPanel({
  presets, simulations, status, activeMeta,
  showSimulation, onToggleSimulation,
  onRunPreset, onLoadSaved, onClear,
  propDepth, propDecay, onPropDepthChange, onPropDecayChange,
}: SimulationPanelProps) {
  const isLoading = status === 'loading'

  return (
    <div>
      <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '10px 0' }} />
      <h3>🧪 Simulation</h3>

      {/* Propagation sliders */}
      <div className="info-item">
        <label style={{ display: 'block', marginBottom: 2 }}>
          Propagation depth: <strong>{propDepth} hops</strong>
        </label>
        <input
          type="range" min={0} max={6} step={1}
          value={propDepth}
          onChange={e => onPropDepthChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
          disabled={isLoading}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
          <span>0 (no spread)</span><span>6 (city-wide)</span>
        </div>
      </div>

      <div className="info-item">
        <label style={{ display: 'block', marginBottom: 2 }}>
          Decay per hop: <strong>{Math.round(propDecay * 100)}%</strong>
          <span style={{ color: '#888', fontSize: 11, marginLeft: 4 }}>
            ({propDecay <= 0.4 ? 'sharp' : propDecay <= 0.6 ? 'moderate' : 'wide'} spread)
          </span>
        </label>
        <input
          type="range" min={0.3} max={0.8} step={0.05}
          value={propDecay}
          onChange={e => onPropDecayChange(Number(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
          disabled={isLoading}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
          <span>30% (sharp)</span><span>80% (wide)</span>
        </div>
      </div>

      {/* Active simulation banner */}
      {activeMeta && (
        <div className="sim-active-banner">
          <strong style={{ color: '#92400e' }}>⚠ {activeMeta.scenario_name}</strong><br />
          {activeMeta.scenario_description && (
            <span style={{ color: '#555', fontSize: 11 }}>{activeMeta.scenario_description}<br /></span>
          )}
          <span style={{ color: '#333', fontSize: 11 }}>
            {activeMeta.direct_edges ?? activeMeta.total_edges_affected} direct
            {activeMeta.propagated_edges != null && ` + ${activeMeta.propagated_edges} propagated`}
            {' '}edges affected
          </span>
          {activeMeta.propagation_depth != null && (
            <span style={{ color: '#666', fontSize: 11, display: 'block' }}>
              depth={activeMeta.propagation_depth} decay={Math.round((activeMeta.propagation_decay ?? 0.55) * 100)}%
            </span>
          )}
          <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
            <button onClick={onToggleSimulation} style={{ marginTop: 0, fontSize: 11, padding: '3px 8px' }}>
              {showSimulation ? 'Hide' : 'Show'} on map
            </button>
            <button onClick={onClear} style={{ marginTop: 0, fontSize: 11, padding: '3px 8px', background: '#ef4444' }}>
              ✕ Clear
            </button>
          </div>
        </div>
      )}

      {/* Preset buttons */}
      <div className="info-item" style={{ color: '#555', marginBottom: 4 }}>Run preset:</div>
      {presets.map(p => (
        <button
          key={p.key}
          onClick={() => onRunPreset(p.key)}
          disabled={isLoading}
          title={p.description}
          className="sim-preset-btn"
        >
          {isLoading ? '⏳' : (SIM_ICONS[p.key] ?? '▶')} {p.name}
        </button>
      ))}

      {/* Saved simulations */}
      {simulations.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="info-item" style={{ color: '#555', marginBottom: 4 }}>Load saved:</div>
          <select
            onChange={e => { if (e.target.value) onLoadSaved(e.target.value) }}
            defaultValue=""
            style={{ width: '100%', maxWidth: '100%' }}
          >
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
        <div className="info-item" style={{ color: '#dc2626', marginTop: 4 }}>⚠ Simulation failed</div>
      )}

      <div className="info-item" style={{ marginTop: 6, color: '#555' }}>
        💥 Infrastructure · 🚫 Closure · 🐢 Congestion
      </div>
    </div>
  )
}

export default SimulationPanel
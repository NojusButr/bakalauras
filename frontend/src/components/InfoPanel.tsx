import type { GeoJsonObject } from 'geojson'
import type { TrafficStatus } from '../hooks/useTrafficData'
import type { RouteMode } from '../hooks/useRouteData'

interface InfoPanelProps {
  city: string
  onCityChange: (city: string) => void
  startPoint: [number, number] | null
  endPoint: [number, number] | null
  classicRoute: GeoJsonObject | null
  trafficRoute: GeoJsonObject | null
  onClear: () => void
  // traffic
  showTraffic: boolean
  onToggleTraffic: () => void
  trafficStatus: TrafficStatus
  trafficLastUpdated: string | null
  snapshots: string[]
  selectedSnapshot: string | null
  onSelectSnapshot: (filename: string | null) => void
  onTriggerSnapshot: () => void
  // routing
  routeMode: RouteMode
  onRouteModeChange: (mode: RouteMode) => void
}

function formatSnapshotLabel(filename: string): string {
  // filename like "2026-03-23T11-27-27.813279"
  // restore colons and parse as date
  try {
    const iso = filename.replace(
      /^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2}).*$/,
      '$1T$2:$3:$4'
    )
    return new Date(iso).toLocaleString()
  } catch {
    return filename
  }
}

function RouteStats({ route, label, color }: { route: GeoJsonObject | null, label: string, color: string }) {
  if (!route) return null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const props = (route as any).features?.[0]?.properties
  if (!props) return null

  const dist = props.total_length_m
    ? `${(props.total_length_m / 1000).toFixed(2)} km`
    : '—'
  const time = props.estimated_time_s
    ? `~${Math.round(props.estimated_time_s / 60)} min`
    : '—'

  return (
    <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
      <span style={{ color, fontWeight: 600 }}>● {label}:</span>{' '}
      {dist}{props.estimated_time_s ? ` · ${time}` : ''}
    </div>
  )
}

function InfoPanel({
  city, onCityChange,
  startPoint, endPoint,
  classicRoute, trafficRoute,
  onClear,
  showTraffic, onToggleTraffic,
  trafficStatus, trafficLastUpdated,
  snapshots, selectedSnapshot, onSelectSnapshot,
  onTriggerSnapshot,
  routeMode, onRouteModeChange,
}: InfoPanelProps) {
  const availableCities = ['Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys']
  const isLoading = trafficStatus === 'loading'

  return (
    <div className="info-panel">
      <h2>Routing</h2>

      <div className="info-item">
        <label htmlFor="city-select">City: </label>
        <select id="city-select" value={city} onChange={e => onCityChange(e.target.value)}>
          {availableCities.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <div className="info-item">
        Start: {startPoint ? `${startPoint[0].toFixed(4)}, ${startPoint[1].toFixed(4)}` : 'Click map'}
      </div>
      <div className="info-item">
        End: {endPoint ? `${endPoint[0].toFixed(4)}, ${endPoint[1].toFixed(4)}` : 'Click map'}
      </div>

      {/* Route stats */}
      <RouteStats route={classicRoute} label="Shortest" color="#3b82f6" />
      <RouteStats route={trafficRoute} label="Fastest" color="#f97316" />

      {(startPoint || endPoint) && (
        <button onClick={onClear} style={{ marginTop: 6 }}>Clear</button>
      )}

      {/* Route mode */}
      <hr style={{ margin: '10px 0' }} />
      <div className="info-item">
        <label>Route: </label>
        <select value={routeMode} onChange={e => onRouteModeChange(e.target.value as RouteMode)}>
          <option value="both">Compare both</option>
          <option value="classic">Shortest only</option>
          <option value="traffic">Traffic-aware only</option>
        </select>
      </div>

      {/* Traffic layer */}
      <hr style={{ margin: '10px 0' }} />
      <h3 style={{ margin: '0 0 8px' }}>Traffic</h3>

      <div className="info-item" style={{ display: 'flex', gap: 8 }}>
        <button onClick={onToggleTraffic} disabled={isLoading}>
          {showTraffic ? 'Hide' : 'Show'} traffic
        </button>
        <button
          onClick={onTriggerSnapshot}
          disabled={isLoading}
          title="Sample TomTom API now (~5 min)"
        >
          {isLoading ? '⏳ Collecting...' : '📡 New snapshot'}
        </button>
      </div>

      {/* Snapshot picker */}
      {snapshots.length > 0 && (
        <div className="info-item">
          <label>Snapshot: </label>
          <select
            value={selectedSnapshot ?? ''}
            onChange={e => onSelectSnapshot(e.target.value || null)}
            style={{ fontSize: '0.75rem', maxWidth: 160 }}
          >
            <option value="">Latest</option>
            {[...snapshots].reverse().map(s => (
              <option key={s} value={s}>{formatSnapshotLabel(s)}</option>
            ))}
          </select>
        </div>
      )}

      {trafficStatus === 'error' && (
        <div className="info-item" style={{ color: 'red' }}>⚠ Traffic fetch failed</div>
      )}

      {trafficLastUpdated && (
        <div className="info-item" style={{ fontSize: '0.75rem', color: '#666' }}>
          {new Date(trafficLastUpdated).toLocaleString()}
        </div>
      )}

      {trafficStatus === 'loaded' && (
        <div className="info-item" style={{ fontSize: '0.75rem' }}>
          <span style={{ color: '#22c55e' }}>● Free</span>{' '}
          <span style={{ color: '#eab308' }}>● Slow</span>{' '}
          <span style={{ color: '#ef4444' }}>● Congested</span>{' '}
          <span style={{ color: '#94a3b8' }}>● No data</span>
        </div>
      )}
    </div>
  )
}

export default InfoPanel
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
  gnnRoute: GeoJsonObject | null
  classifierRoute: GeoJsonObject | null
  onClear: () => void
  showTraffic: boolean
  onToggleTraffic: () => void
  showIncidents: boolean
  onToggleIncidents: () => void
  trafficData: GeoJsonObject | null
  trafficStatus: TrafficStatus
  trafficLastUpdated: string | null
  snapshots: string[]
  selectedSnapshot: string | null
  onSelectSnapshot: (filename: string | null) => void
  onTriggerSnapshot: () => void
  routeMode: RouteMode
  onRouteModeChange: (mode: RouteMode) => void
  dataPct: number
  onDataPctChange: (pct: number) => void
  onRoute: () => void
  routeLoading: boolean
  degradedInfo: { data_pct: number; total_edges: number; edges_with_traffic: number; edges_stripped: number; edges_remaining: number; mode?: string } | null
  showDegraded: boolean
  onToggleDegraded: () => void
  degradeMode: string
  onDegradeModeChange: (mode: string) => void
  corridorWidth: number
  onCorridorWidthChange: (w: number) => void
  zoneRadius: number
  onZoneRadiusChange: (r: number) => void
}

function formatSnapshotLabel(filename: string): string {
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
  const props = (route as unknown as { features?: Array<{ properties?: Record<string, unknown> }> })
    ?.features?.[0]?.properties
  if (!props) return null

  const dist = props.total_length_m
    ? `${((props.total_length_m as number) / 1000).toFixed(2)} km`
    : '—'
  const time = props.estimated_time_s
    ? `~${Math.round((props.estimated_time_s as number) / 60)} min`
    : null
  const gnnTime = props.gnn_predicted_time_s
    ? `~${Math.round((props.gnn_predicted_time_s as number) / 60)} min`
    : null

  return (
    <div style={{ fontSize: '0.8rem', marginTop: 4 }}>
      <span style={{ color, fontWeight: 600 }}>● {label}:</span>{' '}
      {dist}{time ? ` · ${time}` : ''}
      {gnnTime && <span style={{ color: '#888', fontSize: '0.75rem' }}> (GNN est: {gnnTime})</span>}
    </div>
  )
}

function InfoPanel({
  city, onCityChange,
  startPoint, endPoint,
  classicRoute, trafficRoute, gnnRoute, classifierRoute,
  onClear,
  showTraffic, onToggleTraffic,
  showIncidents, onToggleIncidents,
  trafficData, trafficStatus, trafficLastUpdated,
  snapshots, selectedSnapshot, onSelectSnapshot,
  onTriggerSnapshot,
  routeMode, onRouteModeChange,
  dataPct, onDataPctChange,
  onRoute, routeLoading,
  degradedInfo,
  showDegraded, onToggleDegraded,
  degradeMode, onDegradeModeChange,
  corridorWidth, onCorridorWidthChange,
  zoneRadius, onZoneRadiusChange,
}: InfoPanelProps) {
  const availableCities = ['Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys']
  const isLoading = trafficStatus === 'loading'
  const meta = (trafficData as unknown as { metadata?: Record<string, unknown> } | null)?.metadata
  const canRoute = startPoint != null && endPoint != null && !routeLoading

  return (
    <div>
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

      {/* Route button */}
      <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
        <button onClick={onRoute} disabled={!canRoute}
          style={{
            flex: 1, padding: '8px 12px', fontSize: 13, fontWeight: 600,
            background: canRoute ? '#2563eb' : '#93c5fd',
            opacity: routeLoading ? 0.7 : 1,
          }}>
          {routeLoading ? 'Routing...' : 'Route'}
        </button>
        {(startPoint || endPoint) && (
          <button onClick={onClear} style={{ padding: '8px 12px', fontSize: 13 }}>Clear</button>
        )}
      </div>

      <RouteStats route={classicRoute} label="Shortest" color="#2563eb" />
      <RouteStats route={trafficRoute} label="Traffic-aware" color="#ffee00" />
      <RouteStats route={gnnRoute} label="GNN predicted" color="#d946ef" />
      <RouteStats route={classifierRoute} label="GNN classifier" color="#00fff2" />

      {degradedInfo && (
        <div style={{
          marginTop: 8, padding: '6px 8px', fontSize: 11,
          background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 4,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong style={{ color: '#dc2626' }}>Degraded data mode</strong>
            <button
              onClick={onToggleDegraded}
              style={{
                fontSize: 10, padding: '2px 6px', cursor: 'pointer',
                background: showDegraded ? '#fecaca' : '#fee2e2',
                border: '1px solid #fca5a5', borderRadius: 3,
                color: '#dc2626',
              }}
            >
              {showDegraded ? 'Hide' : 'Show'} removed roads
            </button>
          </div>
          <span style={{ color: '#555' }}>
            {degradedInfo.edges_stripped} of {degradedInfo.edges_with_traffic} traffic edges removed
            ({degradedInfo.data_pct}% retained)
            {degradedInfo.mode && (
              <span style={{ color: '#888' }}> — {degradedInfo.mode} pattern</span>
            )}
          </span><br/>
          <span style={{ color: '#888', fontSize: 10 }}>
            Routes computed with partial data, evaluated with full data
          </span>
          {showDegraded && (
            <div style={{ marginTop: 3, fontSize: 10, color: '#dc2626' }}>
              <span style={{ borderBottom: '2px dashed #ef4444' }}>━ ━ ━</span> = roads with no traffic data
            </div>
          )}
        </div>
      )}

      <hr style={{ margin: '10px 0' }} />
      <div className="info-item">
        <label>Route: </label>
        <select value={routeMode} onChange={e => onRouteModeChange(e.target.value as RouteMode)}>
          <option value="both">Compare all</option>
          <option value="classic">Shortest only</option>
          <option value="traffic">Traffic-aware only</option>
          <option value="gnn">GNN only</option>
          <option value="classifier">Classifier only</option>
        </select>
      </div>

      <div className="info-item" style={{ marginTop: 6 }}>
        <div style={{ marginBottom: 4 }}>
          <label style={{ fontSize: 11 }}>Degradation pattern: </label>
          <select value={degradeMode} onChange={e => onDegradeModeChange(e.target.value)}
            style={{ fontSize: 11 }}>
            <option value="random">Random</option>
            <option value="corridor">Route corridor</option>
            <option value="minor">Minor roads only</option>
            <option value="zone">Geographic zone</option>
          </select>
        </div>
        {degradeMode === 'random' && (
          <div>
            <label style={{ display: 'block', marginBottom: 2, fontSize: 11 }}>
              Traffic data retained: <strong>{dataPct}%</strong>
            </label>
            <input type="range" min={0} max={100} step={5} value={dataPct}
              onChange={e => onDataPctChange(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
              <span>0% (none)</span><span>100% (full)</span>
            </div>
          </div>
        )}
        {degradeMode === 'corridor' && (
          <div>
            <label style={{ fontSize: 11 }}>
              Corridor width: <strong>{corridorWidth}m</strong>
            </label>
            <input type="range" min={100} max={5000} step={100} value={corridorWidth}
              onChange={e => onCorridorWidthChange(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
              <span>100m (narrow)</span><span>5000m (wide)</span>
            </div>
            <div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>
              Removes all traffic data within {corridorWidth}m of route line
            </div>
          </div>
        )}
        {degradeMode === 'zone' && (
          <div>
            <label style={{ fontSize: 11 }}>
              Zone radius: <strong>{zoneRadius}m</strong>
            </label>
            <input type="range" min={200} max={8000} step={200} value={zoneRadius}
              onChange={e => onZoneRadiusChange(Number(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
              <span>200m (small)</span><span>8000m (large)</span>
            </div>
            <div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>
              Removes all traffic data within {zoneRadius}m of route midpoint
            </div>
          </div>
        )}
        {degradeMode === 'minor' && (
          <div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>
            Removes traffic data from all residential, service, tertiary, and unclassified roads.
            Keeps primary, secondary, trunk, and motorway data.
          </div>
        )}
      </div>

      <hr style={{ margin: '10px 0' }} />
      <h3 style={{ margin: '0 0 8px' }}>Traffic</h3>

      <div className="info-item" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button onClick={onToggleTraffic} disabled={isLoading}>
          {showTraffic ? 'Hide' : 'Show'} traffic
        </button>
        <button onClick={onToggleIncidents} disabled={isLoading}>
          {showIncidents ? 'Hide' : 'Show'} incidents
        </button>
        <button onClick={onTriggerSnapshot} disabled={isLoading}>
          {isLoading ? 'Fetching...' : 'New snapshot'}
        </button>
      </div>

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
        <div className="info-item" style={{ color: 'red' }}>Traffic fetch failed</div>
      )}

      {trafficLastUpdated && (
        <div className="info-item" style={{ fontSize: '0.75rem', color: '#666' }}>
          {new Date(trafficLastUpdated).toLocaleString()}
          {meta && (
            <span> · {meta.here_flow_segments as number} segments · {meta.here_incidents as number} incidents</span>
          )}
        </div>
      )}

      {trafficStatus === 'loaded' && (
        <div className="info-item" style={{ fontSize: '0.75rem' }}>
          <strong>Legend:</strong><br/>
          <span style={{ color: '#22c55e' }}>●</span> Free{' '}
          <span style={{ color: '#eab308' }}>●</span> Slow{' '}
          <span style={{ color: '#ef4444' }}>●</span> Congested{' '}
          <span style={{ color: '#d1d5db' }}>●</span> No data<br/>
          <span style={{ color: '#2563eb' }}>━ ━</span> Shortest{' '}
          <span style={{ color: '#ffee00' }}>━ ━</span> Traffic{' '}
          <span style={{ color: '#d946ef' }}>━ ━</span> GNN{' '}
          <span style={{ color: '#00fff2' }}>━ ━</span> Classifier
        </div>
      )}
    </div>
  )
}

export default InfoPanel
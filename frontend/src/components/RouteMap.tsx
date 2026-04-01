import { MapContainer, TileLayer, GeoJSON, CircleMarker, Marker, Popup, ZoomControl } from 'react-leaflet'
import type { GeoJsonObject } from 'geojson'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import MapClickHandler from './MapClickHandler'
import { CITIES_COORDS } from '../config/cities'

interface Incident {
  lat: number
  lng: number
  incident_id: string
  incident_type: string
  incident_description: string | null
  incident_criticality: string
  incident_road_closed: boolean
  incident_start_time: string | null
  incident_end_time: string | null
}

interface SimEvent {
  type: string
  center: [number, number]
  radius_m: number
  description: string
  edges_affected?: number
  congestion_factor?: number
}

interface RouteMapProps {
  city: string
  graphData: GeoJsonObject | null
  trafficData: GeoJsonObject | null
  showTraffic: boolean
  showIncidents: boolean
  simData: GeoJsonObject | null
  showSimulation: boolean
  classicRoute: GeoJsonObject | null
  trafficRoute: GeoJsonObject | null
  startPoint: [number, number] | null
  endPoint: [number, number] | null
  onMapClick: (latlng: { lat: number; lng: number }) => void
}

// ── Emoji DivIcon factory ─────────────────────────────────────────────────────
function emojiIcon(emoji: string, size = 28, bgColor = 'white', border = '#ccc') {
  return L.divIcon({
    html: `<div style="
      font-size: ${size}px;
      line-height: 1;
      background: ${bgColor};
      border: 2px solid ${border};
      border-radius: 50%;
      width: ${size + 8}px;
      height: ${size + 8}px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
      cursor: pointer;
    ">${emoji}</div>`,
    className: '',
    iconSize: [size + 8, size + 8],
    iconAnchor: [(size + 8) / 2, (size + 8) / 2],
    popupAnchor: [0, -(size + 8) / 2],
  })
}

// ── Incident helpers ──────────────────────────────────────────────────────────
const INCIDENT_ICONS: Record<string, string> = {
  roadClosure:   '🚫',
  construction:  '🚧',
  accident:      '💥',
  congestion:    '🐢',
  roadHazard:    '⚠️',
  weatherHazard: '🌧️',
}

const INCIDENT_BORDER: Record<string, string> = {
  critical: '#dc2626',
  major:    '#f97316',
  minor:    '#eab308',
}

// ── Simulation helpers ────────────────────────────────────────────────────────
const SIM_ICONS: Record<string, string> = {
  infrastructure_damage: '💥',
  road_closure:          '🚫',
  district_congestion:   '🐢',
}

const SIM_BORDER: Record<string, string> = {
  infrastructure_damage: '#7c3aed',
  road_closure:          '#dc2626',
  district_congestion:   '#f97316',
}

// ── Road style functions ──────────────────────────────────────────────────────
const baseStreetStyle   = { color: '#3388ff', weight: 2, opacity: 0.8 }
const classicRouteStyle = { color: '#3b82f6', weight: 5, opacity: 0.9 }
const trafficRouteStyle = { color: '#f97316', weight: 5, opacity: 0.9, dashArray: '8 4' }

const CONGESTION_COLORS: Record<string, string> = {
  green:   '#22c55e',
  yellow:  '#eab308',
  red:     '#ef4444',
  unknown: '#94a3b8',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function trafficStyle(feature: any) {
  const p = feature?.properties ?? {}
  if (p.road_closure || p.jam_factor === 10)
    return { color: '#dc2626', weight: p.functional_class <= 2 ? 4 : 3, opacity: 1, dashArray: '6 3' }
  const level = p.congestion_level ?? 'unknown'
  const fc = p.functional_class ?? 5
  return { color: CONGESTION_COLORS[level] ?? CONGESTION_COLORS.unknown, weight: fc <= 2 ? 4 : fc <= 3 ? 3 : 2, opacity: 0.85 }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onEachFeature(feature: any, layer: any) {
  const p = feature?.properties
  if (!p) return
  const name    = p.name ?? p.ref ?? 'Unnamed road'
  const speed   = p.current_speed_kph   != null ? `${p.current_speed_kph} kph`   : 'No data'
  const ffSpeed = p.free_flow_speed_kph != null ? `${p.free_flow_speed_kph} kph` : '—'
  const jam     = p.jam_factor          != null ? `${p.jam_factor.toFixed(1)}/10` : '—'
  const conf    = p.confidence          != null ? `${(p.confidence * 100).toFixed(0)}%` : '—'
  const closure = (p.road_closure || p.jam_factor === 10) ? ' 🚫' : ''
  layer.bindTooltip(
    `<strong>${name}</strong>${closure}<br/>Speed: ${speed} / ${ffSpeed}<br/>Jam: ${jam} · Conf: ${conf}`,
    { sticky: true }
  )
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function simStyle(feature: any) {
  const p = feature?.properties ?? {}
  if (!p.simulated) return { opacity: 0, fillOpacity: 0, weight: 0 }
  if (p.infrastructure_damaged) return { color: '#7c3aed', weight: 5, opacity: 0.9 }
  if (p.road_closure)           return { color: '#dc2626', weight: 4, opacity: 0.9, dashArray: '6 3' }
  const jam = p.jam_factor ?? 5
  return { color: jam >= 7 ? '#dc2626' : jam >= 4 ? '#f97316' : '#eab308', weight: 3, opacity: 0.85 }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onEachSimFeature(feature: any, layer: any) {
  const p = feature?.properties
  if (!p?.simulated) return
  const name = p.name ?? p.ref ?? 'Unnamed road'
  const type = p.infrastructure_damaged
    ? '💥 Infrastructure damage'
    : p.road_closure
    ? '🚫 Road closure (simulated)'
    : `🐢 Congestion (simulated) — ${p.current_speed_kph ?? '?'} kph`
  const orig = p.free_flow_speed_kph != null ? `Free-flow: ${p.free_flow_speed_kph} kph` : ''
  layer.bindTooltip(
    `<strong>${name}</strong><br/>${type}<br/><span style="color:#666;font-size:0.85em">${orig}</span>`,
    { sticky: true }
  )
}

function RouteMap({
  city, graphData, trafficData, showTraffic, showIncidents,
  simData, showSimulation,
  classicRoute, trafficRoute,
  startPoint, endPoint, onMapClick,
}: RouteMapProps) {
  const cityCoords       = CITIES_COORDS[city as keyof typeof CITIES_COORDS] || [54.6872, 25.2797]
  const showTrafficLayer = showTraffic && trafficData != null
  const showBaseLayer    = !showTrafficLayer && graphData != null

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const incidents: Incident[] = (trafficData as any)?.metadata?.incidents ?? []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const simEvents: SimEvent[] = showSimulation ? ((simData as any)?.metadata?.events ?? []) : []

  return (
    <MapContainer key={city} center={cityCoords} zoom={13} style={{ height: '100%', width: '100%' }} zoomControl={false}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <ZoomControl position="topright" />

      {showBaseLayer && (
        <GeoJSON key={`graph-${city}`} data={graphData!} style={baseStreetStyle} />
      )}

      {showTrafficLayer && (
        <GeoJSON key={`traffic-${city}`} data={trafficData!} style={trafficStyle} onEachFeature={onEachFeature} />
      )}

      {showSimulation && simData && (
        <GeoJSON key={`sim-${city}-${JSON.stringify(simData).length}`} data={simData} style={simStyle} onEachFeature={onEachSimFeature} />
      )}

      {showIncidents && incidents.map(inc => (
        <Marker
          key={inc.incident_id}
          position={[inc.lat, inc.lng]}
          icon={emojiIcon(INCIDENT_ICONS[inc.incident_type] ?? '⚠️', 20, '#f0f9ff', INCIDENT_BORDER[inc.incident_criticality] ?? '#ccc')}
          zIndexOffset={500}
        >
          <Popup>
            <strong>{INCIDENT_ICONS[inc.incident_type] ?? '⚠️'} {inc.incident_type}</strong>
            {inc.incident_road_closed && <span> 🚫 Road closed</span>}<br />
            {inc.incident_description}<br />
            <span style={{ fontSize: '0.8em', color: '#666' }}>
              Criticality: {inc.incident_criticality}<br />
              {inc.incident_end_time && `Until: ${new Date(inc.incident_end_time).toLocaleString()}`}
            </span>
          </Popup>
        </Marker>
      ))}

      {showSimulation && simEvents.map((ev, i) => (
        <Marker
          key={`sim-event-${i}`}
          position={ev.center}
          icon={emojiIcon(SIM_ICONS[ev.type] ?? '⚠️', 26, '#fef9c3', SIM_BORDER[ev.type] ?? '#f97316')}
          zIndexOffset={1000}
        >
          <Popup>
            <strong>{SIM_ICONS[ev.type]} {ev.type.replace(/_/g, ' ')}</strong><br />
            {ev.description}<br />
            <span style={{ fontSize: '0.8em', color: '#666' }}>
              Radius: {ev.radius_m}m<br />
              {ev.edges_affected != null && `Edges affected: ${ev.edges_affected}`}
              {ev.congestion_factor != null && <><br />Speed reduced to {Math.round(ev.congestion_factor * 100)}% of normal</>}
            </span>
          </Popup>
        </Marker>
      ))}

      {classicRoute && (
        <GeoJSON key={`classic-${JSON.stringify(classicRoute).length}`} data={classicRoute} style={classicRouteStyle} />
      )}
      {trafficRoute && (
        <GeoJSON key={`traffic-route-${JSON.stringify(trafficRoute).length}`} data={trafficRoute} style={trafficRouteStyle} />
      )}

      {startPoint && (
        <CircleMarker center={startPoint} radius={8} color="#00ff00" weight={2} opacity={1}>
          <Popup>Start</Popup>
        </CircleMarker>
      )}
      {endPoint && (
        <CircleMarker center={endPoint} radius={8} color="#ff0000" weight={2} opacity={1}>
          <Popup>End</Popup>
        </CircleMarker>
      )}

      <MapClickHandler onMapClick={onMapClick} />
    </MapContainer>
  )
}

export default RouteMap
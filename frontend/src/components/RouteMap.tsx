import { MapContainer, TileLayer, GeoJSON, CircleMarker, Marker, Popup, ZoomControl, Pane } from 'react-leaflet'
import type { GeoJsonObject } from 'geojson'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import MapClickHandler from './MapClickHandler'
import { CITIES_COORDS } from '../config/cities'

interface Incident {
  lat: number; lng: number; incident_id: string; incident_type: string
  incident_description: string | null; incident_criticality: string; incident_road_closed: boolean
}
interface SimEvent {
  type: string; center: [number, number]; radius_m: number
  speed_reduction_pct?: number; description: string; edges_affected?: number
}
interface RouteMapProps {
  city: string; graphData: GeoJsonObject | null; trafficData: GeoJsonObject | null
  showTraffic: boolean; showIncidents: boolean; simData: GeoJsonObject | null
  showSimulation: boolean; classicRoute: GeoJsonObject | null
  trafficRoute: GeoJsonObject | null; gnnRoute: GeoJsonObject | null
  classifierRoute: GeoJsonObject | null; degradedGeojson: GeoJsonObject | null
  startPoint: [number, number] | null; endPoint: [number, number] | null
  onMapClick: (latlng: { lat: number; lng: number }) => void; placingEvent?: string | null
}

// ── Icon factory ─────────────────────────────────────────────────────────────
function emojiIcon(emoji: string, size: number, bgColor: string, borderColor: string, borderWidth = 2) {
  return L.divIcon({
    html: `<div style="
      font-size:${size}px;line-height:1;background:${bgColor};
      border:${borderWidth}px solid ${borderColor};border-radius:50%;
      width:${size + 8}px;height:${size + 8}px;
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 2px 6px rgba(0,0,0,0.25);cursor:pointer;
    ">${emoji}</div>`,
    className: '',
    iconSize: [size + 8, size + 8],
    iconAnchor: [(size + 8) / 2, (size + 8) / 2],
    popupAnchor: [0, -(size + 8) / 2],
  })
}

// ── HERE incident icons ─────────────────────────────
const INCIDENT_ICONS: Record<string, string> = {
  roadClosure: '🚫', construction: '🚧', accident: '💥',
  congestion: '🐢', roadHazard: '⚠️', weatherHazard: '🌧️',
}
const INCIDENT_BG = '#dbeafe'       // light blue
const INCIDENT_BORDERS: Record<string, string> = {
  critical: '#1d4ed8', major: '#2563eb', minor: '#60a5fa',
}

// ── Simulation event icons  ────────────────────
const SIM_ICONS: Record<string, string> = { damage: '💥', construction: '🚧', congestion: '🐢' }
const SIM_BG = '#fef3c7'            // warm yellow
const SIM_BORDERS: Record<string, string> = {
  damage: '#dc2626', construction: '#ea580c', congestion: '#d97706',
}

// ── Road styles ──────────────────────────────────────────────────────────────
const baseStreetStyle = { color: '#93c5fd', weight: 1.5, opacity: 0.6 }

// Traffic
const TRAFFIC_COLORS: Record<string, string> = {
  green: '#22c55e', yellow: '#eab308', red: '#ef4444', unknown: '#d1d5db',
}

// Simulation
const SIM_ROAD_COLORS = {
  destroyed: '#7c3aed', 
  closed: '#1e1e1e',    
  severe: '#dc2626',    
  moderate: '#ea580c',   
  light: '#f59e0b',     
}

// Routes: thick dashed, high contrast
const classicRouteStyle = { color: '#2563eb', weight: 6, opacity: 0.95, dashArray: '20 10' }
const trafficRouteStyle = { color: '#ffee00', weight: 6, opacity: 0.95, dashArray: '20 10' }
const gnnRouteStyle     = { color: '#d946ef', weight: 6, opacity: 0.95, dashArray: '20 10' }
const classifierStyle   = { color: '#00fff2', weight: 6, opacity: 0.95, dashArray: '20 10' }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function trafficStyle(feature: any) {
  const p = feature?.properties ?? {}
  if (p.current_speed_kph == null && p.jam_factor == null) {
    return { color: '#d1d5db', weight: 1.5, opacity: 0.35 }
  }
  if (p.road_closure || p.impassable || p.jam_factor === 10)
    return { color: '#991b1b', weight: 2.5, opacity: 0.7, dashArray: '3 3' }
  const level = p.congestion_level ?? 'unknown'
  const fc = p.functional_class ?? 5
  return {
    color: TRAFFIC_COLORS[level] ?? TRAFFIC_COLORS.unknown,
    weight: fc <= 2 ? 3 : fc <= 3 ? 2.5 : 1.8,
    opacity: 0.75,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onEachTrafficFeature(feature: any, layer: any) {
  const p = feature?.properties
  if (!p) return
  const name = p.name ?? p.ref ?? 'Unnamed road'
  if (p.current_speed_kph == null && p.jam_factor == null) {
    layer.bindTooltip(`<b>${name}</b><br/><span style="color:#999">No traffic data</span>`, { sticky: true })
    return
  }
  const speed = p.current_speed_kph != null ? `${p.current_speed_kph}` : '—'
  const ff = p.free_flow_speed_kph != null ? `${p.free_flow_speed_kph}` : '—'
  const jam = p.jam_factor != null ? `${p.jam_factor.toFixed(1)}/10` : '—'
  const conf = p.confidence != null ? ` · Conf: ${(p.confidence * 100).toFixed(0)}%` : ''
  layer.bindTooltip(`<b>${name}</b><br/>Speed: ${speed} / ${ff} kph<br/>Jam: ${jam}${conf}`, { sticky: true })
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function simStyle(feature: any) {
  const p = feature?.properties ?? {}
  if (!p.simulated) return { opacity: 0, fillOpacity: 0, weight: 0 }

  if (p.impassable || p.infrastructure_damaged)
    return { color: SIM_ROAD_COLORS.destroyed, weight: 5, opacity: 0.95 }
  if (p.road_closure)
    return { color: SIM_ROAD_COLORS.closed, weight: 4, opacity: 0.9, dashArray: '4 4' }

  const jam = p.jam_factor ?? 5
  const hop = p.propagation_hop ?? 0
  const opacity = Math.max(0.5, 0.95 - hop * 0.07)
  const weight = Math.max(2.5, 4.5 - hop * 0.3)
  const color = jam >= 7 ? SIM_ROAD_COLORS.severe : jam >= 4 ? SIM_ROAD_COLORS.moderate : SIM_ROAD_COLORS.light
  return { color, weight, opacity }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onEachSimFeature(feature: any, layer: any) {
  const p = feature?.properties
  if (!p?.simulated) return
  const name = p.name ?? p.ref ?? 'Unnamed road'
  const hop = p.propagation_hop ?? 0

  let status = ''
  if (p.impassable || p.infrastructure_damaged) status = 'DESTROYED — impassable'
  else if (p.road_closure) status = 'CLOSED — impassable'
  else if (p.construction) status = 'Construction zone'
  else status = 'Congested'

  const speed = p.current_speed_kph != null ? `${p.current_speed_kph}` : '0'
  const ff = p.free_flow_speed_kph != null ? `${p.free_flow_speed_kph}` : '—'
  const jam = p.jam_factor != null ? `${p.jam_factor.toFixed(1)}/10` : '—'

  let reduction = ''
  if (p.impassable || p.road_closure) {
    reduction = '<br/><span style="color:#dc2626;font-weight:600">Road impassable</span>'
  } else if (p.free_flow_speed_kph && p.current_speed_kph != null) {
    const pct = Math.round((1 - p.current_speed_kph / p.free_flow_speed_kph) * 100)
    if (pct > 0) reduction = `<br/><span style="color:#dc2626">${pct}% slower</span>`
  }

  const hopLabel = hop === 0 ? '<span style="color:#7c3aed">Direct impact</span>'
    : `<span style="color:#888">Propagation hop ${hop}</span>`

  layer.bindTooltip(
    `<b>${name}</b> — ${status}<br/>Speed: ${speed} / ${ff} kph · Jam: ${jam}${reduction}<br/>${hopLabel}`,
    { sticky: true }
  )
}

// Generate a stable key from route data so GeoJSON re-renders when route changes
function routeKey(route: GeoJsonObject): string {
  const r = route as unknown as Record<string, unknown>
  const feats = r?.features as Array<Record<string, unknown>> | undefined
  if (!feats || feats.length === 0) return 'empty'
  const props = feats[0]?.properties as Record<string, unknown> | undefined
  const len = props?.total_length_m ?? 0
  const time = props?.estimated_time_s ?? 0
  const weight = props?.weight ?? ''
  return `${weight}-${len}-${time}`
}

function RouteMap({
  city, graphData, trafficData, showTraffic, showIncidents,
  simData, showSimulation, classicRoute, trafficRoute, gnnRoute, classifierRoute,
  degradedGeojson,
  startPoint, endPoint, onMapClick, placingEvent,
}: RouteMapProps) {
  const cityCoords = CITIES_COORDS[city as keyof typeof CITIES_COORDS] || [54.6872, 25.2797]
  const showTrafficLayer = showTraffic && trafficData != null
  const showBaseLayer = !showTrafficLayer && graphData != null

  const trafficAny = trafficData as unknown as Record<string, unknown> | null
  const simAny = simData as unknown as Record<string, unknown> | null
  const incidents: Incident[] = ((trafficAny?.metadata as Record<string, unknown>)?.incidents as Incident[]) ?? []
  const simMeta = (simAny?.metadata as Record<string, unknown>) ?? {}
  const simEvents: SimEvent[] = showSimulation ? ((simMeta.events as SimEvent[]) ?? []) : []
  const simKey = simData ? `sim-${simMeta.timestamp ?? ''}-${((simAny?.features as unknown[]) ?? []).length}` : 'sim-none'

  return (
    <MapContainer key={city} center={cityCoords} zoom={13}
      style={{ height: '100%', width: '100%', cursor: placingEvent ? 'crosshair' : undefined }}
      zoomControl={false}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OSM" />
      <ZoomControl position="topright" />

      {/* Rendering order (bottom → top): base → traffic → sim roads → routes → markers → events */}

      {/* Layer 1: Base street network (when no traffic) */}
      <Pane name="base-streets" style={{ zIndex: 400 }}>
        {showBaseLayer && <GeoJSON key={`graph-${city}`} data={graphData!} style={baseStreetStyle} />}
      </Pane>

      {/* Layer 2: Traffic data */}
      <Pane name="traffic-layer" style={{ zIndex: 410 }}>
        {showTrafficLayer && <GeoJSON key={`traffic-${city}`} data={trafficData!} style={trafficStyle} onEachFeature={onEachTrafficFeature} />}
      </Pane>

      {/* Layer 3: Simulation road effects (on top of traffic) */}
      <Pane name="sim-roads" style={{ zIndex: 420 }}>
        {showSimulation && simData && <GeoJSON key={simKey} data={simData} style={simStyle} onEachFeature={onEachSimFeature} />}
      </Pane>

      {/* Layer 4: Degraded roads visualization (stripped traffic data) */}
      {degradedGeojson && (
        <Pane name="degraded" style={{ zIndex: 415 }}>
          <GeoJSON
            key={`degraded-${routeKey(degradedGeojson)}`}
            data={degradedGeojson}
            style={() => ({
              color: '#ef4444',
              weight: 3,
              opacity: 0.5,
              dashArray: '6,4',
            })}
          />
        </Pane>
      )}

      {/* Layer 5: Routes (on top of everything except markers) */}
      <Pane name="routes" style={{ zIndex: 430 }}>
        {classicRoute && <GeoJSON key={`classic-${routeKey(classicRoute)}`} data={classicRoute} style={classicRouteStyle} />}
        {trafficRoute && <GeoJSON key={`traffic-${routeKey(trafficRoute)}`} data={trafficRoute} style={trafficRouteStyle} />}
        {gnnRoute && <GeoJSON key={`gnn-${routeKey(gnnRoute)}`} data={gnnRoute} style={gnnRouteStyle} />}
        {classifierRoute && <GeoJSON key={`clf-${routeKey(classifierRoute)}`} data={classifierRoute} style={classifierStyle} />}
      </Pane>

      {/* Layer 6: Route start/end markers */}
      <Pane name="route-markers" style={{ zIndex: 440 }}>
        {startPoint && <CircleMarker center={startPoint} radius={9} color="#16a34a" fillColor="#22c55e" fillOpacity={1} weight={3}><Popup>Start</Popup></CircleMarker>}
        {endPoint && <CircleMarker center={endPoint} radius={9} color="#dc2626" fillColor="#ef4444" fillOpacity={1} weight={3}><Popup>End</Popup></CircleMarker>}
      </Pane>

      {/* Layer 6: HERE incident markers (blue-tinted) */}
      {showIncidents && incidents.map((inc: Incident) => (
        <Marker key={inc.incident_id} position={[inc.lat, inc.lng]}
          icon={emojiIcon(INCIDENT_ICONS[inc.incident_type] ?? '⚠️', 18, INCIDENT_BG, INCIDENT_BORDERS[inc.incident_criticality] ?? '#93c5fd', 2)}
          zIndexOffset={500}>
          <Popup>
            <strong>{inc.incident_type}</strong>
            {inc.incident_road_closed && ' — Road closed'}<br />
            {inc.incident_description}<br/>
            <span style={{ fontSize: '0.8em', color: '#2563eb' }}>Source: HERE API</span>
          </Popup>
        </Marker>
      ))}

      {/* Layer 7: Simulation event markers (warm-tinted, larger, thicker border) */}
      {showSimulation && simEvents.map((ev: SimEvent, i: number) => (
        <Marker key={`sim-ev-${i}`} position={ev.center}
          icon={emojiIcon(SIM_ICONS[ev.type] ?? '⚠️', 24, SIM_BG, SIM_BORDERS[ev.type] ?? '#d97706', 3)}
          zIndexOffset={1000}>
          <Popup>
            <strong>{ev.type}</strong><br />
            {ev.description}<br />
            <span style={{ fontSize: '0.8em', color: '#666' }}>
              Speed: {ev.speed_reduction_pct ?? '—'}%
              {ev.edges_affected != null && ` · ${ev.edges_affected} edges`}
            </span><br/>
            <span style={{ fontSize: '0.8em', color: '#d97706' }}>Source: Simulation</span>
          </Popup>
        </Marker>
      ))}

      <MapClickHandler onMapClick={onMapClick} />
    </MapContainer>
  )
}

export default RouteMap
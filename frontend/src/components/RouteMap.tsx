import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup } from 'react-leaflet'
import type { GeoJsonObject } from 'geojson'
import 'leaflet/dist/leaflet.css'
import MapClickHandler from './MapClickHandler'
import { CITIES_COORDS } from '../config/cities'

interface RouteMapProps {
  city: string
  graphData: GeoJsonObject | null
  trafficData: GeoJsonObject | null
  showTraffic: boolean
  classicRoute: GeoJsonObject | null
  trafficRoute: GeoJsonObject | null
  startPoint: [number, number] | null
  endPoint: [number, number] | null
  onMapClick: (latlng: { lat: number; lng: number }) => void
}

const baseStreetStyle = { color: '#3388ff', weight: 2, opacity: 0.8 }

// Blue = shortest distance route
const classicRouteStyle = { color: '#3b82f6', weight: 5, opacity: 0.9 }

// Orange = traffic-aware fastest route
const trafficRouteStyle = { color: '#f97316', weight: 5, opacity: 0.9, dashArray: '12 10' }

const CONGESTION_COLORS: Record<string, string> = {
  green: '#22c55e',
  yellow: '#eab308',
  red: '#ef4444',
  unknown: '#94a3b8',
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function trafficStyle(feature: any) {
  if (feature?.properties?.road_closure) {
    return { color: '#000000', weight: 3, opacity: 0.9 }
  }
  const level = feature?.properties?.congestion_level ?? 'unknown'
  return {
    color: CONGESTION_COLORS[level] ?? CONGESTION_COLORS.unknown,
    weight: 2,
    opacity: 0.85,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onEachFeature(feature: any, layer: any) {
  const p = feature?.properties
  if (!p) return
  const name = p.name ?? p.ref ?? 'Unnamed road'
  const current = p.current_speed_kph != null ? `${p.current_speed_kph} kph` : 'No data'
  const freeFlow = p.free_flow_speed_kph != null ? `${p.free_flow_speed_kph} kph` : '—'
  const confidence = p.confidence != null ? `${(p.confidence * 100).toFixed(0)}%` : '—'
  const delay = p.travel_time_delay_s != null ? `+${p.travel_time_delay_s}s` : '—'
  const closure = p.road_closure ? ' 🚫 CLOSED' : ''
  layer.bindTooltip(
    `<strong>${name}</strong>${closure}<br/>
     Speed: ${current} / ${freeFlow} free-flow<br/>
     Delay: ${delay} · Confidence: ${confidence}`,
    { sticky: true }
  )
}

function RouteMap({
  city, graphData, trafficData, showTraffic,
  classicRoute, trafficRoute,
  startPoint, endPoint, onMapClick,
}: RouteMapProps) {
  const cityCoords = CITIES_COORDS[city as keyof typeof CITIES_COORDS] || [54.6872, 25.2797]
  const showTrafficLayer = showTraffic && trafficData != null
  const showBaseLayer = !showTrafficLayer && graphData != null

  return (
    <MapContainer key={city} center={cityCoords} zoom={13} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />

      {showBaseLayer && (
        <GeoJSON key={`graph-${city}`} data={graphData!} style={baseStreetStyle} />
      )}

      {showTrafficLayer && (
        <GeoJSON key={`traffic-${city}`} data={trafficData!} style={trafficStyle} onEachFeature={onEachFeature} />
      )}

      {/* Classic route — solid blue, drawn first (underneath) */}
      {classicRoute && (
        <GeoJSON key={`classic-${JSON.stringify(classicRoute).length}`} data={classicRoute} style={classicRouteStyle} />
      )}

      {/* Traffic route — dashed orange, drawn on top */}
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
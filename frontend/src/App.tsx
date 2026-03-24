import { useState, useEffect } from 'react'
import './App.css'
import InfoPanel from './components/InfoPanel'
import RouteMap from './components/RouteMap'
import { useGraphData } from './hooks/useGraphData'
import { useRouteData, type RouteMode } from './hooks/useRouteData'
import { useTrafficData } from './hooks/useTrafficData'

function App() {
  const [city, setCity] = useState('Vilnius')
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null)
  const [endPoint, setEndPoint] = useState<[number, number] | null>(null)
  const [showTraffic, setShowTraffic] = useState(false)
  const [routeMode, setRouteMode] = useState<RouteMode>('both')

  const graphData = useGraphData(city)
  const {
    trafficData, status: trafficStatus, lastUpdated: trafficLastUpdated,
    snapshots, selectedSnapshot, fetchSnapshot, triggerSnapshot,
  } = useTrafficData(city)

  const { classicRoute, trafficRoute, clearRoutes } = useRouteData(
    startPoint, endPoint, city, routeMode, selectedSnapshot
  )

  useEffect(() => {
    setStartPoint(null)
    setEndPoint(null)
    clearRoutes()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city])

  const handleMapClick = (latlng: { lat: number; lng: number }) => {
    if (!startPoint) {
      setStartPoint([latlng.lat, latlng.lng])
    } else if (!endPoint) {
      setEndPoint([latlng.lat, latlng.lng])
    } else {
      setStartPoint([latlng.lat, latlng.lng])
      setEndPoint(null)
      clearRoutes()
    }
  }

  const handleClear = () => {
    setStartPoint(null)
    setEndPoint(null)
    clearRoutes()
  }

  return (
    <div className="map-container">
      <InfoPanel
        city={city}
        onCityChange={c => { if (c !== city) setCity(c) }}
        startPoint={startPoint}
        endPoint={endPoint}
        classicRoute={classicRoute}
        trafficRoute={trafficRoute}
        onClear={handleClear}
        showTraffic={showTraffic}
        onToggleTraffic={() => setShowTraffic(v => !v)}
        trafficStatus={trafficStatus}
        trafficLastUpdated={trafficLastUpdated}
        snapshots={snapshots}
        selectedSnapshot={selectedSnapshot}
        onSelectSnapshot={fetchSnapshot}
        onTriggerSnapshot={triggerSnapshot}
        routeMode={routeMode}
        onRouteModeChange={setRouteMode}
      />

      <RouteMap
        key={city}
        city={city}
        graphData={graphData}
        trafficData={trafficData}
        showTraffic={showTraffic}
        classicRoute={classicRoute}
        trafficRoute={trafficRoute}
        startPoint={startPoint}
        endPoint={endPoint}
        onMapClick={handleMapClick}
      />
    </div>
  )
}

export default App
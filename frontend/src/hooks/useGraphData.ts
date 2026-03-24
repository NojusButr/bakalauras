import { useEffect, useState } from 'react'
import type { GeoJsonObject } from 'geojson'

// In-memory cache
const memoryCache = new Map<string, GeoJsonObject>()

export function useGraphData(city: string) {
  const [graphData, setGraphData] = useState<GeoJsonObject | null>(null)

  useEffect(() => {
    let cancelled = false

    // 🔥 Clear previous city data immediately
    setGraphData(null)

    // 1️⃣ Check memory cache
    if (memoryCache.has(city)) {
      const cachedData = memoryCache.get(city)!
      console.log(
        `Loading ${city} from memory cache, features:`,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (cachedData as any).features?.length || 0
      )

      if (!cancelled) {
        setGraphData(cachedData)
      }

      return
    }

    // 2️⃣ Fetch from backend
    async function load() {
      try {
        console.log(`Fetching ${city} from backend...`)
        const encodedCity = encodeURIComponent(city)

        const res = await fetch(
          `http://localhost:8000/cities/${encodedCity}`
        )

        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }

        const data = await res.json()

        if (data?.error) {
          console.error(`Backend error for ${city}:`, data.error)
          return
        }

        if (!data?.features) {
          console.error(`Invalid data structure for ${city}:`, data)
          return
        }

        // Save to memory cache
        memoryCache.set(city, data)

        if (!cancelled) {
          setGraphData(data)
          console.log(
            `${city} data loaded, features:`,
            data.features.length
          )
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Failed to fetch graph:', err)
        }
      }
    }

    load()

    // 🛑 Cancel outdated fetch if city changes
    return () => {
      cancelled = true
    }
  }, [city])

  return graphData
}

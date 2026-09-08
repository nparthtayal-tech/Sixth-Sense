function calculateDirectDistance(a, b) {
  const R = 6371000;
  const lat1 = (a[0] * Math.PI) / 180;
  const lat2 = (b[0] * Math.PI) / 180;
  const dLat = ((b[0] - a[0]) * Math.PI) / 180;
  const dLng = ((b[1] - a[1]) * Math.PI) / 180;
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

export function generateAirCorridor(source, destination, numPoints = 60) {
  const points = [];
  const dist = calculateDirectDistance(source, destination);
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    const lat = source[0] + (destination[0] - source[0]) * t;
    const lng = source[1] + (destination[1] - source[1]) * t;
    points.push([lat, lng]);
  }
  return {
    path: points,
    distance: dist,
    duration: dist / 12 // ~45 km/h flight speed
  };
}

export async function getRoute(source, destination) {
  try {
    const coordinates = [
      `${source[1]},${source[0]}`,
      `${destination[1]},${destination[0]}`
    ].join(";");

    const url =
      `https://router.project-osrm.org/route/v1/driving/${coordinates}` +
      "?overview=full&geometries=geojson&alternatives=true";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);

    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.code === "Ok" && data.routes && data.routes.length > 0) {
        const routes = data.routes.map((item) => ({
          path: item.geometry.coordinates.map(([lng, lat]) => [lat, lng]),
          distance: item.distance,
          duration: item.duration
        }));

        routes.sort((a, b) => {
          const scoreA = a.distance * 0.4 + a.duration * 100 * 0.6;
          const scoreB = b.distance * 0.4 + b.duration * 100 * 0.6;
          return scoreA - scoreB;
        });

        return routes[0];
      }
    }
  } catch (e) {
    // Network or timeout: fallback to direct UAV corridor
  }

  // Graceful direct air flight corridor fallback
  return generateAirCorridor(source, destination);
}

export async function getRecoveryRoute(currentPosition, rejoinPosition) {
  try {
    const coordinates = [
      `${currentPosition[1]},${currentPosition[0]}`,
      `${rejoinPosition[1]},${rejoinPosition[0]}`
    ].join(";");

    const url =
      `https://router.project-osrm.org/route/v1/driving/${coordinates}` +
      "?overview=full&geometries=geojson";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.code === "Ok" && data.routes && data.routes.length > 0) {
        return {
          path: data.routes[0].geometry.coordinates.map(([lng, lat]) => [lat, lng]),
          distance: data.routes[0].distance,
          duration: data.routes[0].duration
        };
      }
    }
  } catch (e) {
    // Fallback
  }

  return generateAirCorridor(currentPosition, rejoinPosition, 25);
}
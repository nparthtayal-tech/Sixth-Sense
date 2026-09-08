// Curated real-world flight / navigation corridor presets in Chennai
export const MISSION_PRESETS = [
  {
    id: "marina-patrol",
    name: "Marina Beach Coastal Patrol",
    description: "Maritime border inspection along Marina coastal highway",
    source: [13.0400, 80.2800], // Near Santhome / Marina South
    destination: [13.0800, 80.2900], // Fort St George area
    nominalAlt: 48, // meters
    nominalSpeed: 45 // km/h
  },
  {
    id: "harbour-transit",
    name: "Chennai Port Security Corridor",
    description: "Automated logistics and maritime perimeter security flight",
    source: [13.0880, 80.2950], // Chennai Port inner basin
    destination: [13.1250, 80.3000], // Kasimedu Fisheries Harbour
    nominalAlt: 55,
    nominalSpeed: 52
  },
  {
    id: "urban-tech-grid",
    name: "Guindy Corridor Inspection",
    description: "Critical infrastructure survey between IIT Madras and Guindy park",
    source: [13.0035, 80.2350], // IIT Madras gate
    destination: [13.0180, 80.2150], // Guindy industrial zone
    nominalAlt: 60,
    nominalSpeed: 38
  },
  {
    id: "adyar-estuary",
    name: "Adyar Estuary Air Patrol",
    description: "Environmental sensor monitoring across Elliot's Beach and Estuary",
    source: [12.9990, 80.2720], // Besant Nagar Beach
    destination: [13.0200, 80.2600], // Adyar Gate
    nominalAlt: 42,
    nominalSpeed: 40
  }
];

export function calculateBearing(from, to) {
  if (!from || !to) return 0;
  const lat1 = (from[0] * Math.PI) / 180;
  const lat2 = (to[0] * Math.PI) / 180;
  const dLng = ((to[1] - from[1]) * Math.PI) / 180;
  const y = Math.sin(dLng) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(dLng);
  const bearing = (Math.atan2(y, x) * 180) / Math.PI;
  return (bearing + 360) % 360;
}

export function getFovCone(center, heading, distanceMeters = 75, fovDeg = 48) {
  if (!center) return [];
  const R = 6371000;
  const rad = (d) => (d * Math.PI) / 180;
  const deg = (r) => (r * 180) / Math.PI;

  const projectPoint = (bearingDeg) => {
    const brng = rad(bearingDeg);
    const lat1 = rad(center[0]);
    const lon1 = rad(center[1]);
    const lat2 = Math.asin(
      Math.sin(lat1) * Math.cos(distanceMeters / R) +
      Math.cos(lat1) * Math.sin(distanceMeters / R) * Math.cos(brng)
    );
    const lon2 = lon1 + Math.atan2(
      Math.sin(brng) * Math.sin(distanceMeters / R) * Math.cos(lat1),
      Math.cos(distanceMeters / R) - Math.sin(lat1) * Math.sin(lat2)
    );
    return [deg(lat2), deg(lon2)];
  };

  const left = projectPoint(heading - fovDeg / 2);
  const mid = projectPoint(heading);
  const right = projectPoint(heading + fovDeg / 2);
  return [center, left, mid, right];
}

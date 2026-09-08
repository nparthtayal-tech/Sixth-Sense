import React, { useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  Polygon,
  Circle,
  useMap,
  useMapEvents
} from "react-leaflet";
import L from "leaflet";
import { getFovCone } from "../services/presets";

const DEFAULT_CENTER = [13.0600, 80.2800];

// Custom tactical SVG icons
function createDroneIcon(heading = 0, quarantined = false, recovery = false, spoofed = false) {
  const color = quarantined
    ? "#facc15"
    : recovery
      ? "#22d3ee"
      : spoofed
        ? "#f43f5e"
        : "#00ff9d";

  const glow = quarantined
    ? "rgba(250, 204, 21, 0.9)"
    : recovery
      ? "rgba(34, 211, 238, 0.9)"
      : spoofed
        ? "rgba(244, 63, 94, 0.9)"
        : "rgba(0, 255, 157, 0.9)";

  return new L.DivIcon({
    className: "drone-leaflet-icon",
    html: `
      <div style="
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        transform: rotate(${heading}deg);
        transition: transform 0.25s ease-out;
      ">
        <div style="
          position: absolute;
          inset: -4px;
          border-radius: 50%;
          border: 1.5px dashed ${color};
          opacity: 0.75;
          animation: spinDroneRing 5s linear infinite;
        "></div>
        <svg width="46" height="46" viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="8" fill="${color}" style="filter: drop-shadow(0 0 10px ${glow});"/>
          <line x1="8" y1="8" x2="40" y2="40" stroke="${color}" stroke-width="3" stroke-linecap="round"/>
          <line x1="40" y1="8" x2="8" y2="40" stroke="${color}" stroke-width="3" stroke-linecap="round"/>
          <circle cx="8" cy="8" r="5" fill="#061224" stroke="${color}" stroke-width="2"/>
          <circle cx="40" cy="8" r="5" fill="#061224" stroke="${color}" stroke-width="2"/>
          <circle cx="8" cy="40" r="5" fill="#061224" stroke="${color}" stroke-width="2"/>
          <circle cx="40" cy="40" r="5" fill="#061224" stroke="${color}" stroke-width="2"/>
          <polygon points="24,8 19,19 29,19" fill="#ffffff"/>
        </svg>
        <div style="
          position: absolute;
          bottom: -19px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(3, 10, 22, 0.95);
          border: 1.5px solid ${color};
          color: ${color};
          font-size: 9.5px;
          font-weight: 900;
          letter-spacing: 0.8px;
          padding: 1px 6px;
          border-radius: 4px;
          white-space: nowrap;
          pointer-events: none;
          box-shadow: 0 0 10px ${glow};
        ">${quarantined ? "EKF (ISOLATED)" : recovery ? "REJOINING" : spoofed ? "HIJACK DRIFT" : "UAV TRUSTED"}</div>
      </div>
    `,
    iconSize: [52, 52],
    iconAnchor: [26, 26]
  });
}

const sourceIcon = new L.DivIcon({
  className: "source-leaflet-icon",
  html: `
    <div style="
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #00ff9d;
      border: 3px solid #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 900;
      color: #03140e;
      box-shadow: 0 0 20px rgba(0, 255, 157, 0.95);
      position: relative;
      cursor: grab;
    ">
      <div style="
        position: absolute;
        inset: -8px;
        border-radius: 50%;
        border: 2px solid #00ff9d;
        animation: pingRipple 1.8s cubic-bezier(0, 0, 0.2, 1) infinite;
      "></div>
      S
      <div style="
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(2, 14, 10, 0.95);
        border: 1px solid #00ff9d;
        color: #00ff9d;
        font-size: 9px;
        font-weight: 800;
        padding: 1px 5px;
        border-radius: 3px;
        white-space: nowrap;
      ">START</div>
    </div>
  `,
  iconSize: [36, 36],
  iconAnchor: [18, 18]
});

const destinationIcon = new L.DivIcon({
  className: "dest-leaflet-icon",
  html: `
    <div style="
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #f43f5e;
      border: 3px solid #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 900;
      color: #ffffff;
      box-shadow: 0 0 20px rgba(244, 63, 94, 0.95);
      position: relative;
      cursor: grab;
    ">
      <div style="
        position: absolute;
        inset: -8px;
        border-radius: 50%;
        border: 2px solid #f43f5e;
        animation: pingRipple 1.8s cubic-bezier(0, 0, 0.2, 1) infinite;
      "></div>
      D
      <div style="
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(28, 4, 10, 0.95);
        border: 1px solid #f43f5e;
        color: #ff6b7d;
        font-size: 9px;
        font-weight: 800;
        padding: 1px 5px;
        border-radius: 3px;
        white-space: nowrap;
      ">TARGET</div>
    </div>
  `,
  iconSize: [36, 36],
  iconAnchor: [18, 18]
});

const gpsIcon = new L.DivIcon({
  className: "gps-leaflet-icon",
  html: `
    <div style="
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: #ff4757;
      border: 2.5px solid #ffffff;
      box-shadow: 0 0 18px rgba(255, 71, 87, 0.95);
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    ">
      <div style="width: 8px; height: 8px; border-radius: 50%; background: #ffffff;"></div>
      <div style="
        position: absolute;
        bottom: -18px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(30, 4, 8, 0.95);
        border: 1px solid #ff4757;
        color: #ff6b7d;
        font-size: 8.5px;
        font-weight: 800;
        padding: 1px 4px;
        border-radius: 3px;
        white-space: nowrap;
      ">RAW GPS</div>
    </div>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 14]
});

const spoofBeaconIcon = new L.DivIcon({
  className: "spoof-beacon-icon",
  html: `
    <div style="
      width: 48px;
      height: 48px;
      border-radius: 50%;
      border: 3px solid #f43f5e;
      background: rgba(244, 63, 94, 0.28);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #f43f5e;
      font-size: 24px;
      font-weight: 900;
      box-shadow: 0 0 28px rgba(244, 63, 94, 0.9);
      position: relative;
      cursor: pointer;
    ">
      <div style="
        position: absolute;
        inset: -14px;
        border-radius: 50%;
        border: 2px solid rgba(244, 63, 94, 0.7);
        animation: pingRipple 1.3s ease-out infinite;
      "></div>
      ⚡
      <div style="
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(30, 4, 8, 0.95);
        border: 1px solid #f43f5e;
        color: #ff6b7d;
        font-size: 9px;
        font-weight: 900;
        padding: 1px 5px;
        border-radius: 3px;
        white-space: nowrap;
      ">SPOOF TARGET</div>
    </div>
  `,
  iconSize: [48, 48],
  iconAnchor: [24, 24]
});

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click(e) {
      onMapClick([e.latlng.lat, e.latlng.lng]);
    }
  });
  return null;
}

function MapController({ route, vehiclePosition, followVehicle, source, destination }) {
  const map = useMap();

  // Fit bounds when a new route is loaded
  useEffect(() => {
    if (route && route.length > 1) {
      const bounds = L.latLngBounds(route);
      map.fitBounds(bounds, {
        padding: [80, 80],
        maxZoom: 16
      });
    } else if (source && destination) {
      const bounds = L.latLngBounds([source, destination]);
      map.fitBounds(bounds, {
        padding: [80, 80],
        maxZoom: 15
      });
    }
  }, [route, source, destination, map]);

  // Smoothly follow vehicle when active
  useEffect(() => {
    if (followVehicle && vehiclePosition) {
      map.panTo(vehiclePosition, {
        animate: true,
        duration: 0.35,
        easeLinearity: 0.5
      });
    }
  }, [vehiclePosition, followVehicle, map]);

  return null;
}

export default function SensorMap({
  source,
  destination,
  route = [],
  vehiclePosition,
  vehicleHeading = 0,
  gpsPosition,
  spoofing = false,
  spoofTarget,
  recoveryRoute = [],
  breadcrumbTrail = [],
  selectionMode = null, // 'start' | 'dest' | null
  missionRunning = false,
  recoveryActive = false,
  quarantined = false,
  followVehicle = true,
  onMapClick,
  onSourceDrag,
  onDestDrag,
  onSpoofDrag
}) {
  const center = useMemo(() => {
    return vehiclePosition || source || destination || DEFAULT_CENTER;
  }, [vehiclePosition, source, destination]);

  const droneIcon = useMemo(() => {
    return createDroneIcon(vehicleHeading, quarantined, recoveryActive, spoofing);
  }, [vehicleHeading, quarantined, recoveryActive, spoofing]);

  const fovPolygon = useMemo(() => {
    if (!vehiclePosition) return [];
    return getFovCone(vehiclePosition, vehicleHeading, 85, 50);
  }, [vehiclePosition, vehicleHeading]);

  // Guidance banner message
  let promptBanner = null;
  if (selectionMode === "start") {
    promptBanner = (
      <div className="onmap-guidance-pill start">
        📍 CLICK ANYWHERE ON THE MAP TO SET INITIAL DEPARTURE (START)
      </div>
    );
  } else if (selectionMode === "dest" || (!destination && source)) {
    promptBanner = (
      <div className="onmap-guidance-pill dest">
        🎯 CLICK ANYWHERE ON THE MAP TO SET FINAL DESTINATION
      </div>
    );
  } else if (!source && !destination) {
    promptBanner = (
      <div className="onmap-guidance-pill start">
        📍 1. CLICK MAP TO SET INITIAL START POINT
      </div>
    );
  } else if (spoofing) {
    promptBanner = (
      <div className="onmap-guidance-pill spoof">
        ⚡ SPOOF ACTIVE: CLICK ANYWHERE ON MAP TO REDIRECT SPOOF TARGET
      </div>
    );
  } else if (missionRunning) {
    promptBanner = (
      <div className="onmap-guidance-pill run">
        ⚡ CLICK ANYWHERE ON MAP TO INJECT SPOOF ATTACK TARGET
      </div>
    );
  }

  return (
    <div className="tactical-map-viewport">
      {promptBanner}

      <MapContainer
        center={center}
        zoom={14}
        scrollWheelZoom={true}
        zoomControl={false}
        className="full-leaflet-canvas"
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a> | &copy; OpenStreetMap'
          url="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png"
          subdomains="abcd"
          maxZoom={20}
        />

        <MapClickHandler onMapClick={onMapClick} />

        <MapController
          route={route}
          source={source}
          destination={destination}
          vehiclePosition={vehiclePosition}
          followVehicle={followVehicle}
        />

        {/* Source Pin (Draggable when not flying) */}
        {source && (
          <Marker
            position={source}
            icon={sourceIcon}
            draggable={!missionRunning}
            eventHandlers={{
              dragend(e) {
                const ll = e.target.getLatLng();
                if (onSourceDrag) onSourceDrag([ll.lat, ll.lng]);
              }
            }}
          >
            <Popup className="tactical-popup">
              <strong>INITIAL START POINT</strong><br />
              {source[0].toFixed(5)}, {source[1].toFixed(5)}<br />
              <small>Drag to adjust start location</small>
            </Popup>
          </Marker>
        )}

        {/* Destination Pin (Draggable when not flying) */}
        {destination && (
          <Marker
            position={destination}
            icon={destinationIcon}
            draggable={!missionRunning}
            eventHandlers={{
              dragend(e) {
                const ll = e.target.getLatLng();
                if (onDestDrag) onDestDrag([ll.lat, ll.lng]);
              }
            }}
          >
            <Popup className="tactical-popup">
              <strong>FINAL DESTINATION</strong><br />
              {destination[0].toFixed(5)}, {destination[1].toFixed(5)}<br />
              <small>Drag to adjust destination</small>
            </Popup>
          </Marker>
        )}

        {/* Authorized Route Polyline */}
        {route && route.length > 1 && (
          <>
            <Polyline
              positions={route}
              pathOptions={{
                color: "#00ff9d",
                weight: 12,
                opacity: 0.22,
                lineCap: "round",
                lineJoin: "round"
              }}
            />
            <Polyline
              positions={route}
              pathOptions={{
                color: "#00ff9d",
                weight: 4.5,
                opacity: 0.95,
                lineCap: "round",
                lineJoin: "round"
              }}
            />
          </>
        )}

        {/* Breadcrumb Trail of drone */}
        {breadcrumbTrail && breadcrumbTrail.length > 1 && (
          <Polyline
            positions={breadcrumbTrail}
            pathOptions={{
              color: "#38bdf8",
              weight: 3,
              opacity: 0.65,
              dashArray: "4 4"
            }}
          />
        )}

        {/* Autonomous Recovery Corridor */}
        {recoveryRoute && recoveryRoute.length > 1 && (
          <>
            <Polyline
              positions={recoveryRoute}
              pathOptions={{
                color: "#22d3ee",
                weight: 10,
                opacity: 0.3,
                lineCap: "round"
              }}
            />
            <Polyline
              positions={recoveryRoute}
              pathOptions={{
                color: "#22d3ee",
                weight: 5,
                dashArray: "12 8",
                opacity: 0.98
              }}
            />
          </>
        )}

        {/* LiDAR Radar Sweep Circle */}
        {vehiclePosition && (
          <Circle
            center={vehiclePosition}
            radius={60}
            pathOptions={{
              color: quarantined ? "#facc15" : spoofing ? "#f43f5e" : "#00ff9d",
              fillColor: quarantined ? "#facc15" : spoofing ? "#f43f5e" : "#00ff9d",
              fillOpacity: 0.08,
              weight: 1.5,
              dashArray: "5 5"
            }}
          />
        )}

        {/* Forward Camera Optical Cone */}
        {fovPolygon.length > 2 && (
          <Polygon
            positions={fovPolygon}
            pathOptions={{
              color: "#22d3ee",
              fillColor: "#22d3ee",
              fillOpacity: 0.16,
              weight: 1.5
            }}
          />
        )}

        {/* Drone Marker */}
        {vehiclePosition && (
          <Marker position={vehiclePosition} icon={droneIcon} zIndexOffset={1000}>
            <Popup className="tactical-popup">
              <strong>UAV SENSORSENTRY</strong><br />
              Heading: {Math.round(vehicleHeading)}°<br />
              Status: {quarantined ? "GPS QUARANTINED (EKF DEAD RECKONING)" : recoveryActive ? "AUTONOMOUS RECOVERY" : spoofing ? "HIJACK DRIFT" : "NOMINAL FLIGHT"}<br />
              Coords: {vehiclePosition[0].toFixed(5)}, {vehiclePosition[1].toFixed(5)}
            </Popup>
          </Marker>
        )}

        {/* Rogue Spoofing Target Beacon on Map (Draggable) */}
        {spoofTarget && (
          <Marker
            position={spoofTarget}
            icon={spoofBeaconIcon}
            zIndexOffset={850}
            draggable={true}
            eventHandlers={{
              dragend(e) {
                const ll = e.target.getLatLng();
                if (onSpoofDrag) onSpoofDrag([ll.lat, ll.lng]);
              }
            }}
          >
            <Popup className="tactical-popup">
              <strong style={{ color: "#f43f5e" }}>⚡ ROGUE GNSS TRANSMITTER</strong><br />
              Attack Target: {spoofTarget[0].toFixed(5)}, {spoofTarget[1].toFixed(5)}<br />
              <small>Click map or drag to redirect attack</small>
            </Popup>
          </Marker>
        )}

        {/* Raw / Spoofed GPS Marker */}
        {gpsPosition && (
          <Marker position={gpsPosition} icon={gpsIcon} zIndexOffset={900}>
            <Popup className="tactical-popup">
              <strong style={{ color: spoofing ? "#f43f5e" : "#00ff9d" }}>
                {spoofing ? "⚠ SPOOFED GPS FIX" : "GNSS RECEIVER"}
              </strong><br />
              {spoofing ? "Corrupted by false pseudoranges" : "Signal authenticated"}
            </Popup>
          </Marker>
        )}

        {/* Dynamic Vector Line Connecting Drone to Spoof Target when active */}
        {spoofing && vehiclePosition && spoofTarget && (
          <Polyline
            positions={[vehiclePosition, spoofTarget]}
            pathOptions={{
              color: "#f43f5e",
              weight: 2.5,
              dashArray: "6 6",
              opacity: 0.6
            }}
          />
        )}

        {/* Tether line between True Vehicle & Fake GPS */}
        {spoofing && vehiclePosition && gpsPosition && (
          <Polyline
            positions={[vehiclePosition, gpsPosition]}
            pathOptions={{
              color: "#ff4757",
              weight: 3.5,
              dashArray: "8 6",
              opacity: 0.95
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
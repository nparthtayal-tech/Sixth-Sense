import React, { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  Marker,
  Popup,
  Polyline,
  Polygon,
  Circle,
  CircleMarker,
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

function createCobotIcon(label = "AGV-COBOT", heading = 0, isSpoofed = false, isCompromised = false) {
  const color = isCompromised || isSpoofed ? "#f43f5e" : "#f59e0b";
  const glow = isCompromised || isSpoofed ? "rgba(244, 63, 94, 0.9)" : "rgba(245, 158, 11, 0.85)";

  return new L.DivIcon({
    className: "cobot-leaflet-icon",
    html: `
      <div style="
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        transform: rotate(${heading}deg);
        transition: transform 0.25s ease-out;
      ">
        <div style="
          position: absolute;
          inset: -3px;
          border-radius: 8px;
          border: 1.5px dashed ${color};
          opacity: 0.8;
          animation: spinDroneRing 6s linear infinite;
        "></div>
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <rect x="8" y="10" width="24" height="20" rx="4" fill="#0f172a" stroke="${color}" stroke-width="2.5" style="filter: drop-shadow(0 0 8px ${glow});" />
          <rect x="4" y="13" width="4" height="14" rx="2" fill="${color}" />
          <rect x="32" y="13" width="4" height="14" rx="2" fill="${color}" />
          <polygon points="20,13 15,21 25,21" fill="${color}" />
          <circle cx="20" cy="24" r="3.5" fill="#ffffff" />
        </svg>
        <div style="
          position: absolute;
          bottom: -18px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(15, 23, 42, 0.95);
          border: 1px solid ${color};
          color: ${color};
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.6px;
          padding: 1px 5px;
          border-radius: 3px;
          white-space: nowrap;
          box-shadow: 0 0 8px ${glow};
        ">${label}</div>
      </div>
    `,
    iconSize: [48, 48],
    iconAnchor: [24, 24]
  });
}

function createSwarmDroneIcon(label = "UAV-SWARM", heading = 0, isSpoofed = false, isCompromised = false) {
  const color = isCompromised ? "#f43f5e" : isSpoofed ? "#f43f5e" : "#22d3ee";
  const glow = isCompromised || isSpoofed ? "rgba(244, 63, 94, 0.9)" : "rgba(34, 211, 238, 0.85)";

  return new L.DivIcon({
    className: "swarm-drone-leaflet-icon",
    html: `
      <div style="
        width: 48px;
        height: 48px;
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
        <svg width="42" height="42" viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="7" fill="${color}" style="filter: drop-shadow(0 0 8px ${glow});"/>
          <line x1="10" y1="10" x2="38" y2="38" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="38" y1="10" x2="10" y2="38" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="10" cy="10" r="4" fill="#061224" stroke="${color}" stroke-width="1.8"/>
          <circle cx="38" cy="10" r="4" fill="#061224" stroke="${color}" stroke-width="1.8"/>
          <circle cx="10" cy="38" r="4" fill="#061224" stroke="${color}" stroke-width="1.8"/>
          <circle cx="38" cy="38" r="4" fill="#061224" stroke="${color}" stroke-width="1.8"/>
          <polygon points="24,10 20,19 28,19" fill="#ffffff"/>
        </svg>
        <div style="
          position: absolute;
          bottom: -18px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(3, 10, 22, 0.95);
          border: 1.2px solid ${color};
          color: ${color};
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.6px;
          padding: 1px 5px;
          border-radius: 3px;
          white-space: nowrap;
          box-shadow: 0 0 8px ${glow};
        ">${label}</div>
      </div>
    `,
    iconSize: [48, 48],
    iconAnchor: [24, 24]
  });
}

function createHazardSiteIcon(type = "CH4_GAS_LEAK", label = "HAZARD") {
  const isGas = type.includes("GAS") || type.includes("CH4");
  const color = isGas ? "#f59e0b" : "#ef4444";
  const iconEmoji = isGas ? "☣️" : "🔥";
  return new L.DivIcon({
    className: "hazard-site-icon",
    html: `
      <div style="
        width: 42px;
        height: 48px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        cursor: pointer;
      ">
        <div style="
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: rgba(15, 23, 42, 0.92);
          border: 2px solid ${color};
          box-shadow: 0 0 14px ${color};
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 15px;
          animation: pulseGlowRed 1.8s ease-in-out infinite;
        ">${iconEmoji}</div>
        <div style="
          margin-top: 2px;
          background: rgba(15, 23, 42, 0.95);
          border: 1px solid ${color};
          color: ${color};
          font-size: 7.5px;
          font-weight: 800;
          padding: 1px 4px;
          border-radius: 3px;
          white-space: nowrap;
          letter-spacing: 0.5px;
          box-shadow: 0 0 6px rgba(0,0,0,0.8);
        ">${label}</div>
      </div>
    `,
    iconSize: [42, 48],
    iconAnchor: [21, 24]
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

function MapController({ route, vehiclePosition, followVehicle, source, destination, envMode, floorPlanUrl, floorPlanBounds }) {
  const map = useMap();

  // Fit bounds to Floor Plan and lock the camera
  useEffect(() => {
    if (envMode === "factory" && floorPlanUrl && floorPlanBounds) {
      const b = L.latLngBounds(floorPlanBounds);
      map.fitBounds(b, { padding: [30, 30] });
      // Allow slight panning around the floor plan with 35% padding
      map.setMaxBounds(b.pad(0.35));
      const targetZoom = Math.max(map.getBoundsZoom(b, false), 17);
      map.setMinZoom(Math.max(15, targetZoom - 2));
    } else {
      map.setMaxBounds(null);
      map.setMinZoom(0);
    }
  }, [envMode, floorPlanUrl, floorPlanBounds, map]);

  // Fit bounds when a new route is loaded
  useEffect(() => {
    if (route && route.length > 1) {
      const bounds = L.latLngBounds(route);
      map.fitBounds(bounds, {
        padding: [60, 60],
        maxZoom: envMode === "factory" ? 22 : 16
      });
    } else if (source && destination) {
      const bounds = L.latLngBounds([source, destination]);
      map.fitBounds(bounds, {
        padding: [60, 60],
        maxZoom: envMode === "factory" ? 21 : 15
      });
    }
  }, [route, source, destination, map, envMode]);

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
  overrideToStart = false,
  followVehicle = true,
  envMode = "outdoor",
  floorPlanUrl = null,
  floorPlanScale = 100,
  customNodes = [],
  customEdges = [],
  manualWaypoints = [],
  selectedDotForEdge = null,
  onMapClick,
  onSourceDrag,
  onDestDrag,
  onSpoofDrag,
  onFloorPlanUpload,
  onLoadSampleBlueprint,
  fileInputRef,
  swarmMode = false,
  swarmRobots = [],
  meshLinks = [],
  sharedHazards = [],
  environmentalHazards = []
}) {
  const [basemapStyle, setBasemapStyle] = useState("tactical"); // 'tactical' | 'satellite'

  const mapStyle = {
    height: "100%",
    width: "100%",
    background: envMode === "factory" ? "#0f172a" : "#030a14",
    cursor: selectionMode ? "crosshair" : "grab"
  };
  const center = useMemo(() => {
    if (envMode === "factory") {
      return vehiclePosition || source || destination || DEFAULT_CENTER;
    }
    return vehiclePosition || source || destination || DEFAULT_CENTER;
  }, [envMode, vehiclePosition, source, destination]);

  const droneIcon = useMemo(() => {
    return createDroneIcon(vehicleHeading, quarantined, recoveryActive, spoofing);
  }, [vehicleHeading, quarantined, recoveryActive, spoofing]);

  const fovPolygon = useMemo(() => {
    if (!vehiclePosition) return [];
    return getFovCone(vehiclePosition, vehicleHeading, 85, 50);
  }, [vehiclePosition, vehicleHeading]);

  // Guidance banner message
  let promptBanner = null;
  if (overrideToStart) {
    promptBanner = (
      <div className="onmap-guidance-pill spoof" style={{ background: "rgba(244, 63, 94, 0.95)", color: "#ffffff", borderColor: "#ffffff", fontWeight: 800 }}>
        🛑 HIGH-FREQUENCY CYBER ATTACK: DESTINATION OVERRIDDEN WITH STARTING POINT — RETURNING TO START
      </div>
    );
  } else if (selectionMode === "start") {
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

  // Calculate Floor Plan Bounds
  const floorPlanBounds = useMemo(() => {
    if (!floorPlanScale) return [[0,0], [0,0]];
    const lat = DEFAULT_CENTER[0];
    const lng = DEFAULT_CENTER[1];
    
    // 1 degree latitude = ~111,320 meters
    const lat_diff = floorPlanScale / 111320;
    // 1 degree longitude = ~111,320 * cos(lat) meters
    const lng_diff = floorPlanScale / (111320 * Math.cos(lat * Math.PI / 180));
    
    // Center the image exactly around DEFAULT_CENTER
    return [
      [lat - lat_diff / 2, lng - lng_diff / 2],
      [lat + lat_diff / 2, lng + lng_diff / 2]
    ];
  }, [floorPlanScale]);

  return (
    <div 
      className="tactical-map-viewport"
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (onFloorPlanUpload) onFloorPlanUpload(e);
      }}
    >
      {promptBanner}

      {envMode === "outdoor" && (
        <div className="tactical-layer-control">
          <button
            type="button"
            className={`layer-btn ${basemapStyle === "tactical" ? "active" : ""}`}
            onClick={() => setBasemapStyle("tactical")}
            title="Dark Cyber Tactical Mode (Watermark-Free)"
          >
            🛰️ Tactical Dark
          </button>
          <button
            type="button"
            className={`layer-btn ${basemapStyle === "satellite" ? "active" : ""}`}
            onClick={() => setBasemapStyle("satellite")}
            title="High-Res Aerial Satellite (Watermark-Free)"
          >
            🌍 Satellite
          </button>
        </div>
      )}

      <MapContainer
        center={center}
        zoom={14}
        maxZoom={22}
        scrollWheelZoom={true}
        zoomControl={false}
        className="full-leaflet-canvas"
        style={mapStyle}
      >
        {envMode === "outdoor" && basemapStyle === "tactical" && (
          <TileLayer
            key="osm-tactical"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            className="tactical-dark-tiles"
            maxZoom={19}
          />
        )}

        {envMode === "outdoor" && basemapStyle === "satellite" && (
          <TileLayer
            key="esri-satellite"
            attribution='Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            maxZoom={19}
          />
        )}

        {envMode === "factory" && floorPlanUrl && (
          <ImageOverlay
            url={floorPlanUrl}
            bounds={floorPlanBounds}
            opacity={0.88}
          />
        )}

        {envMode === "factory" && !floorPlanUrl && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1000,
            background: 'rgba(15,23,42,0.94)',
            padding: '28px 36px',
            borderRadius: '12px',
            color: '#fff',
            border: '1.5px solid #00ff9d',
            boxShadow: '0 12px 40px rgba(0,255,157,0.25)',
            textAlign: 'center',
            maxWidth: '460px'
          }}>
             <div style={{ fontSize: '32px', marginBottom: '8px' }}>🏭</div>
             <h3 style={{ color: '#00ff9d', margin: '0 0 8px 0', fontSize: '18px' }}>Factory Indoor Flight Arena</h3>
             <p style={{ color: '#cbd5e1', fontSize: '13px', margin: '0 0 16px 0', lineHeight: 1.5 }}>
               Drag & drop any floor plan image anywhere onto this map, browse a file, or load the pre-configured 100m factory blueprint.
             </p>
             <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
               <button 
                 type="button"
                 onClick={() => fileInputRef?.current?.click()}
                 style={{ background: '#00ff9d', color: '#090d16', border: 'none', padding: '9px 18px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}
               >
                 📥 Upload Floor Plan
               </button>
               <button 
                 type="button"
                 onClick={onLoadSampleBlueprint}
                 style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #38bdf8', padding: '9px 18px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', fontSize: '12px' }}
               >
                 ⚡ Load Sample Plan
               </button>
             </div>
          </div>
        )}

        <MapClickHandler onMapClick={onMapClick} />

        <MapController
          route={route}
          source={source}
          destination={destination}
          vehiclePosition={vehiclePosition}
          followVehicle={followVehicle}
          envMode={envMode}
          floorPlanUrl={floorPlanUrl}
          floorPlanBounds={floorPlanBounds}
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

        {/* Destination Pin (Draggable when not flying and not overridden) */}
        {destination && (
          <Marker
            position={destination}
            icon={destinationIcon}
            draggable={!missionRunning && !overrideToStart}
            eventHandlers={{
              dragend(e) {
                const ll = e.target.getLatLng();
                if (onDestDrag) onDestDrag([ll.lat, ll.lng]);
              }
            }}
          >
            <Popup className="tactical-popup">
              <strong>{overrideToStart ? "DESTINATION OVERRIDDEN TO START" : "FINAL DESTINATION"}</strong><br />
              {destination[0].toFixed(5)}, {destination[1].toFixed(5)}<br />
              <small>{overrideToStart ? "Target locked to initial start point under cyber attack" : "Drag to adjust destination"}</small>
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

        {/* Manual Dots Rendering */}
        {envMode === "factory" && customNodes && customNodes.map((node, idx) => (
           <Circle
             key={`custom-node-${idx}`}
             center={node}
             radius={selectedDotForEdge === idx ? 1.5 : 0.8}
             pathOptions={{ 
               color: selectedDotForEdge === idx ? "#ef4444" : "#facc15", 
               fillColor: selectedDotForEdge === idx ? "#ef4444" : "#facc15", 
               fillOpacity: 0.9 
             }}
             interactive={false}
           />
        ))}

        {/* Manual Edges Rendering */}
        {envMode === "factory" && customEdges && customEdges.map((edge, idx) => {
           const n1 = customNodes[edge[0]];
           const n2 = customNodes[edge[1]];
           if (!n1 || !n2) return null;
           return (
             <Polyline
               key={`custom-edge-${idx}`}
               positions={[n1, n2]}
               color="rgba(250, 204, 21, 0.4)"
               weight={1.5}
               dashArray="3 3"
               interactive={false}
             />
           );
        })}

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

        {/* ── Active Environmental Hazard Zones (Plumes & Spikes) ── */}
        {swarmMode && environmentalHazards && environmentalHazards.map((hz) => {
          const isGas = hz.type.includes("GAS") || hz.type.includes("CH4");
          const plumeColor = isGas ? "#f59e0b" : "#ef4444";
          return (
            <React.Fragment key={hz.id}>
              {/* Outer Dissipation Plume */}
              <Circle
                center={hz.pos}
                radius={hz.radius}
                pathOptions={{
                  color: plumeColor,
                  fillColor: plumeColor,
                  fillOpacity: 0.12,
                  weight: 1.5,
                  dashArray: "6 6"
                }}
              />
              {/* Inner High-Density Core */}
              <Circle
                center={hz.pos}
                radius={Math.round(hz.radius * 0.45)}
                pathOptions={{
                  color: plumeColor,
                  fillColor: plumeColor,
                  fillOpacity: 0.28,
                  weight: 2
                }}
              />
              {/* Hazard Warning Pin */}
              <Marker position={hz.pos} icon={createHazardSiteIcon(hz.type, hz.title)}>
                <Popup className="tactical-popup">
                  <strong style={{ color: plumeColor }}>⚠️ {hz.title}</strong><br />
                  Severity: <strong style={{ color: "#f43f5e" }}>{hz.severity}</strong><br />
                  {hz.basePpm && <>Peak Concentration: <strong>{hz.basePpm} PPM</strong><br /></>}
                  {hz.tempC && <>Hotspot Temp: <strong>{hz.tempC} °C</strong><br /></>}
                  Effective Radius: <strong>{hz.radius}m</strong><br />
                  <small style={{ color: "#38bdf8" }}>Swarm cobots entering perimeter actively detect and broadcast telemetry alerts.</small>
                </Popup>
              </Marker>
            </React.Fragment>
          );
        })}

        {/* ── Autonomous Robot Dynamic Sensor Scan Cones & Footprints ── */}
        {swarmMode && swarmRobots && swarmRobots.map((robot) => {
          if (robot.role === "cobot" && robot.position) {
            const isSpike = (robot.sensors?.gas || 0) > 40;
            return (
              <Circle
                key={`sniffer-${robot.id}`}
                center={robot.position}
                radius={isSpike ? 24 : 18}
                pathOptions={{
                  color: isSpike ? "#f43f5e" : "#10b981",
                  fillColor: isSpike ? "#f43f5e" : "#10b981",
                  fillOpacity: isSpike ? 0.28 : 0.08,
                  weight: isSpike ? 2.5 : 1.2,
                  dashArray: isSpike ? "3 3" : "4 4"
                }}
              />
            );
          }
          if (robot.role === "drone" && robot.id === "UAV-BETA" && robot.position) {
            return (
              <Circle
                key={`radar-${robot.id}`}
                center={robot.position}
                radius={48}
                pathOptions={{
                  color: "#22d3ee",
                  fillColor: "#22d3ee",
                  fillOpacity: 0.07,
                  weight: 1.2,
                  dashArray: "6 6"
                }}
              />
            );
          }
          return null;
        })}

        {/* ── IoT Multi-Robot Swarm Mesh Links ────────────────────── */}
        {swarmMode && meshLinks && meshLinks.map((link, idx) => (
          <Polyline
            key={`mesh-link-${idx}`}
            positions={[link.fromPos, link.toPos]}
            pathOptions={{
              color: link.status === "threat" ? "#f43f5e" : link.status === "warning" ? "#f59e0b" : "#22d3ee",
              weight: link.status === "threat" ? 3.5 : 2.5,
              dashArray: "6 8",
              opacity: 0.85,
            }}
          >
            <Popup className="tactical-popup">
              <strong style={{ color: link.status === "threat" ? "#f43f5e" : "#22d3ee" }}>
                📡 P2P IoT MESH LINK
              </strong><br />
              {link.from} ⟷ {link.to}<br />
              Signal: <strong>{link.rssi} dBm</strong> (Dist: {Math.round(link.distanceM)}m)<br />
              Status: <strong>{link.status.toUpperCase()}</strong>
            </Popup>
          </Polyline>
        ))}

        {/* ── IoT Swarm Collective Hazard Markers ───────────────────── */}
        {swarmMode && sharedHazards && sharedHazards.map((hazard, idx) => (
          <CircleMarker
            key={`hazard-${idx}`}
            center={hazard.position}
            radius={10}
            pathOptions={{
              color: hazard.hazard_type === "GPS_SPOOFING" ? "#f43f5e" : "#f59e0b",
              fillColor: hazard.hazard_type === "GPS_SPOOFING" ? "#f43f5e" : "#f59e0b",
              fillOpacity: 0.7,
              weight: 2,
            }}
          >
            <Popup className="tactical-popup">
              <strong style={{ color: "#f43f5e" }}>🚨 COLLECTIVE HAZARD DETECTED</strong><br />
              Type: {hazard.hazard_type}<br />
              Reported By: {hazard.reported_by}<br />
              Details: {hazard.details || "Discrepancy reported"}
            </Popup>
          </CircleMarker>
        ))}

        {/* ── IoT Swarm Multi-Robot Markers ────────────────────────── */}
        {swarmMode && swarmRobots && swarmRobots.map((robot) => {
          const icon = robot.role === "cobot"
            ? createCobotIcon(robot.name, robot.heading || 0, robot.isSpoofed, robot.isCompromised)
            : createSwarmDroneIcon(robot.name, robot.heading || 0, robot.isSpoofed, robot.isCompromised);

          return (
            <Marker key={robot.id} position={robot.position} icon={icon} zIndexOffset={950}>
              <Popup className="tactical-popup">
                <strong style={{ color: robot.isCompromised ? "#f43f5e" : "#00ff9d" }}>
                  {robot.name} [{robot.role.toUpperCase()}]
                </strong><br />
                Battery: <strong>{robot.battery}%</strong><br />
                {robot.sensors?.gas !== undefined && <>Gas PPM: <strong>{robot.sensors.gas}</strong><br /></>}
                {robot.sensors?.obstacle !== undefined && <>Nearest Obstacle: <strong>{robot.sensors.obstacle}m</strong><br /></>}
                GPS Trust: <strong>{((robot.sensors?.gpsTrust ?? 1) * 100).toFixed(0)}%</strong><br />
                Swarm Status: <span style={{ color: robot.isCompromised ? "#f43f5e" : "#22d3ee", fontWeight: "bold" }}>{robot.status}</span>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
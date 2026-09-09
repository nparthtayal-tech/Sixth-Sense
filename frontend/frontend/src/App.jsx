import { useEffect, useRef, useState, useCallback } from "react";
import {
  ShieldCheck,
  Navigation,
  Target,
  Route,
  Activity,
  AlertTriangle,
  Lock,
  CheckCircle2,
  MapPin,
  Play,
  Pause,
  RotateCcw,
  RadioTower,
  Eye,
  EyeOff,
  Compass,
  Zap,
  ChevronUp,
  ChevronDown,
  Trash2,
  Crosshair,
  Flame,
  Terminal,
  Wifi
} from "lucide-react";

import "./App.css";
import SensorMap from "./Components/SensorMap";
import SensorCards from "./Components/SensorCards";
import { getRoute, getRecoveryRoute, generateAirCorridor } from "./services/routing";
import { findPathAStar, findGeoAStarPath } from "./services/astar";
import { MISSION_PRESETS, calculateBearing } from "./services/presets";
import {
  checkPixelWall
} from "./services/floorPlanProcessor";

const DETECTION_THRESHOLD = 95; // meters (Chi-Square / CUSUM gate)

// Dynamic Ground Patrol Waypoints for AGV-OMEGA (patrolling ground sector corridors)
const OMEGA_PATROL_WAYPOINTS = [
  [13.0594, 80.2796],
  [13.0601, 80.2809],
  [13.06035, 80.28185], // Intersects Methane Plume zone!
  [13.0598, 80.2824],
  [13.0591, 80.2811]
];

// Active Environmental Hazards on the tactical map
const INITIAL_ENVIRONMENTAL_HAZARDS = [
  {
    id: "hz-ch4-plume",
    type: "CH4_GAS_LEAK",
    title: "CH₄ METHANE PLUME",
    pos: [13.0603, 80.2818],
    radius: 65,
    severity: "CRITICAL",
    basePpm: 110.0
  },
  {
    id: "hz-thermal-spot",
    type: "THERMAL_HOTSPOT",
    title: "THERMAL ANOMALY",
    pos: [13.0593, 80.2787],
    radius: 48,
    severity: "HIGH",
    tempC: 78.4
  }
];

function distanceBetween(a, b) {
  if (!a || !b) return 0;
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

function moveTowards(current, target, amount) {
  if (!current || !target) return current;
  const latDiff = target[0] - current[0];
  const lngDiff = target[1] - current[1];
  const dist = Math.sqrt(latDiff ** 2 + lngDiff ** 2);
  if (dist === 0) return target;
  const ratio = Math.min(amount / dist, 1);
  return [current[0] + latDiff * ratio, current[1] + lngDiff * ratio];
}

export default function App() {
  // Navigation & Flight State: Unprecoded (User chooses on map!)
  const [source, setSource] = useState(null);
  const [destination, setDestination] = useState(null);
  const [route, setRoute] = useState([]);
  const [recoveryRoute, setRecoveryRoute] = useState([]);
  const [breadcrumbTrail, setBreadcrumbTrail] = useState([]);
  const [distance, setDistance] = useState(null);
  const [duration, setDuration] = useState(null);

  // Selection mode: 'start' | 'dest' | null
  const [selectionMode, setSelectionMode] = useState("start");

  // Real-time Vehicle & Sensor Positions
  const [vehiclePosition, setVehiclePosition] = useState(null);
  const [vehicleHeading, setVehicleHeading] = useState(0);
  const [gpsPosition, setGpsPosition] = useState(null);
  const [spoofTarget, setSpoofTarget] = useState(null);

  // Simulation Controls & Status
  const [missionRunning, setMissionRunning] = useState(false);
  const [missionPaused, setMissionPaused] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [followVehicle, setFollowVehicle] = useState(true);

  // Defense & Attack State
  const [spoofing, setSpoofing] = useState(false);
  const [quarantined, setQuarantined] = useState(false);
  const [recoveryActive, setRecoveryActive] = useState(false);
  const [recoveryProgress, setRecoveryProgress] = useState(0);
  const [securityState, setSecurityState] = useState("SELECT WAYPOINTS");
  const [riskScore, setRiskScore] = useState(0);
  const [gpsTrust, setGpsTrust] = useState(100);
  const [gpsDivergence, setGpsDivergence] = useState(0);
  const [spoofCount, setSpoofCount] = useState(0);

  // Environment Mode (Outdoor vs Factory Floor Plan)
  const [envMode, setEnvMode] = useState("outdoor");
  const [floorPlanUrl, setFloorPlanUrl] = useState(null);
  const [floorPlanScale, setFloorPlanScale] = useState(100);

  // Telemetry & UI Drawers
  const [routeLoading, setRouteLoading] = useState(false);
  const [alerts, setAlerts] = useState([]);

  // Audit Log Helper (defined early to prevent TDZ ReferenceError in hooks)
  const addAlert = useCallback((message, type = "info") => {
    setAlerts((prev) => [
      {
        id: Date.now() + Math.random(),
        message,
        type,
        time: new Date().toLocaleTimeString()
      },
      ...prev
    ].slice(0, 30));
  }, []);

  const [logExpanded, setLogExpanded] = useState(false);
  const [leftDockOpen, setLeftDockOpen] = useState(true);
  const [rightDockOpen, setRightDockOpen] = useState(true);

  // Fast animation refs
  const routeIndexRef = useRef(0);
  const vehiclePosRef = useRef(null);
  const gpsPosRef = useRef(null);
  const spoofTargetRef = useRef(null);
  const spoofingRef = useRef(false);
  const recoveryRouteRef = useRef([]);
  const recoveryIndexRef = useRef(0);
  const recoveryActiveRef = useRef(false);
  const missionRunningRef = useRef(false);
  const missionPausedRef = useRef(false);
  const quarantinedRef = useRef(false);
  const spoofTimestampsRef = useRef([]);
  const overrideToStartRef = useRef(false);
  const [overrideToStart, setOverrideToStart] = useState(false);
  const spoofStartTimeRef = useRef(0);

  // Auto-Generated Graph Network State
  const [customNodes, setCustomNodes] = useState([]); // array of [lat, lng]
  const [customEdges, setCustomEdges] = useState([]); // array of [idxA, idxB]
  const [manualWaypoints, setManualWaypoints] = useState([]); // array of user-clicked manual waypoints
  const [factoryPhase, setFactoryPhase] = useState("draw_dots"); // "draw_dots", "connect_dots", or "ready"
  const [selectedDotForEdge, setSelectedDotForEdge] = useState(null);
  const [aiMessage, setAiMessage] = useState("");
  const [isAiProcessing, setIsAiProcessing] = useState(false);

  // ── IoT Swarm Mesh State ─────────────────────────────────────────
  const [swarmMode, setSwarmMode] = useState(true);
  const [swarmPatrolActive, setSwarmPatrolActive] = useState(true);
  const [swarmRobots, setSwarmRobots] = useState([]);
  const [meshLinks, setMeshLinks] = useState([]);
  const [sharedHazards, setSharedHazards] = useState([]);
  const [environmentalHazards, setEnvironmentalHazards] = useState(INITIAL_ENVIRONMENTAL_HAZARDS);
  const [swarmConsensusAlert, setSwarmConsensusAlert] = useState(null);
  const [swarmDockOpen, setSwarmDockOpen] = useState(true);
  const [iotPacketLog, setIotPacketLog] = useState([]);
  const [cobotGasPpm, setCobotGasPpm] = useState(14.2);
  const [cobotObstacleDist, setCobotObstacleDist] = useState(4.2);

  // High-frequency autonomous swarm motion & telemetry refs
  const betaAngleRef = useRef(0);
  const betaPosRef = useRef([13.0608, 80.2809]);
  const betaHeadingRef = useRef(45);
  const omegaWpIdxRef = useRef(0);
  const omegaPosRef = useRef(OMEGA_PATROL_WAYPOINTS[0]);
  const omegaHeadingRef = useRef(90);
  const packetSeqRef = useRef(1001);
  const lastHazardBroadcastRef = useRef(0);
  const lastPacketLogTimeRef = useRef(0);

  const collisionImageWidthRef = useRef(0);
  const collisionImageHeightRef = useRef(0);
  const floorPlanImageDataRef = useRef(null);
  const floorPlanDarkIsWallRef = useRef(true);
  const fileInputRef = useRef(null);

  // Load Image Data for UI
  useEffect(() => {
    if (floorPlanUrl && envMode === "factory") {
      const img = new Image();
      img.onload = () => {
        collisionImageWidthRef.current = img.width;
        collisionImageHeightRef.current = img.height;
      };
      img.src = floorPlanUrl;
    }
  }, [floorPlanUrl, envMode]);

  // Swarm Interactive Action Triggers
  const triggerSimulatedGasSpike = () => {
    const curOmega = omegaPosRef.current || [13.0601, 80.2809];
    // Immediately center the methane plume directly onto AGV-OMEGA's current location!
    setEnvironmentalHazards((prev) =>
      prev.map((h) =>
        h.id === "hz-ch4-plume"
          ? { ...h, pos: [curOmega[0] + 0.00004, curOmega[1] + 0.00004], basePpm: 128.0 }
          : h
      )
    );
    addAlert("☣️ GAS LEAK INJECTED: Plume centered on AGV-OMEGA. Sniffer PPM will spike immediately!", "warning");
  };

  const triggerSwarmSpoofAttack = () => {
    const alphaPos = vehiclePosition || source || [13.0600, 80.2800];
    injectOrUpdateSpoofTarget([alphaPos[0] + 0.0016, alphaPos[1] + 0.0018]);
    addAlert("⚡ SWARM BYZANTINE TEST: UAV-ALPHA spoofed! UAV-BETA & AGV-OMEGA will cross-check and isolate it.", "danger");
  };

  // ── Multi-Robot IoT Swarm Autonomous Simulation Engine ──
  useEffect(() => {
    if (!swarmMode) {
      setSwarmRobots([]);
      setMeshLinks([]);
      setSharedHazards([]);
      setSwarmConsensusAlert(null);
      return;
    }

    const timer = setInterval(() => {
      // 1. Advance UAV-BETA Orbital Patrol
      if (swarmPatrolActive) {
        betaAngleRef.current += 0.035 * simulationSpeed;
        const centerLat = 13.0604;
        const centerLng = 80.2806;
        const nextBetaLat = centerLat + 0.00062 * Math.sin(betaAngleRef.current);
        const nextBetaLng = centerLng + 0.00078 * Math.cos(betaAngleRef.current);
        const prevBeta = betaPosRef.current || [centerLat, centerLng];
        const nextBeta = [nextBetaLat, nextBetaLng];
        betaHeadingRef.current = calculateBearing(prevBeta, nextBeta);
        betaPosRef.current = nextBeta;
      }

      // 2. Advance AGV-OMEGA Ground Corridors
      if (swarmPatrolActive) {
        const targetWp = OMEGA_PATROL_WAYPOINTS[omegaWpIdxRef.current];
        const curOmega = omegaPosRef.current || OMEGA_PATROL_WAYPOINTS[0];
        const nextOmega = moveTowards(curOmega, targetWp, 0.00010 * simulationSpeed);
        omegaHeadingRef.current = calculateBearing(curOmega, targetWp);
        omegaPosRef.current = nextOmega;

        if (distanceBetween(nextOmega, targetWp) < 14) {
          omegaWpIdxRef.current = (omegaWpIdxRef.current + 1) % OMEGA_PATROL_WAYPOINTS.length;
        }
      }

      // 3. Dynamic Environmental Hazard Sensing
      const methaneHazard = environmentalHazards.find((h) => h.id === "hz-ch4-plume");
      let currentGas = 12.0 + Math.sin(Date.now() / 1500) * 1.8;
      if (methaneHazard && omegaPosRef.current) {
        const distToPlume = distanceBetween(omegaPosRef.current, methaneHazard.pos);
        if (distToPlume <= methaneHazard.radius) {
          const intensity = 1.0 - distToPlume / methaneHazard.radius;
          currentGas = Math.round(14 + intensity * 98 + (Math.random() * 4 - 2));

          // Threshold Alert (> 45 PPM)
          if (currentGas > 45 && Date.now() - lastHazardBroadcastRef.current > 4500) {
            lastHazardBroadcastRef.current = Date.now();
            const timeStr = new Date().toLocaleTimeString();
            addAlert(
              `🚨 AGV-OMEGA SENSOR ALERT: Methane Gas Plume Detected (${currentGas} PPM) at Sector B! Broadcast sent to swarm.`,
              "danger"
            );

            setSharedHazards((prev) => {
              if (prev.some((h) => h.hazard_type === "CH4_GAS_LEAK")) return prev;
              return [
                ...prev,
                {
                  hazard_type: "CH4_GAS_LEAK",
                  reported_by: "AGV-OMEGA",
                  position: methaneHazard.pos,
                  details: `Toxic CH4 gas concentration: ${currentGas} PPM (Threshold: 45 PPM)`
                }
              ];
            });

            // Add immediate high-priority packet
            setIotPacketLog((prev) => [
              {
                id: `pkt-${Date.now()}`,
                time: timeStr,
                type: "ALERT_BROADCAST",
                sender: "AGV-OMEGA",
                target: "SWARM_BROADCAST",
                channel: "CH_11_802.15.4",
                seq: packetSeqRef.current++,
                rssi: -52,
                payload: `HAZARD_DISCOVERY: CH4_GAS=${currentGas}ppm, LAT=${omegaPosRef.current[0].toFixed(5)}, LNG=${omegaPosRef.current[1].toFixed(5)}`
              },
              ...prev.slice(0, 24)
            ]);
          }
        }
      }
      setCobotGasPpm(typeof currentGas === "number" ? Math.round(currentGas) : currentGas);
      const curObstacle = +(3.8 + Math.sin(Date.now() / 1800) * 1.6).toFixed(1);
      setCobotObstacleDist(curObstacle);

      // 4. Update Swarm Fleet Array
      const alphaPos = vehiclePosition || source || [13.0600, 80.2800];
      const alphaReportedPos = spoofing && gpsPosition ? gpsPosition : alphaPos;
      const isAlphaSpoofed = spoofing && (gpsDivergence > 15 || quarantined);

      const robots = [
        {
          id: "UAV-ALPHA",
          name: "UAV-ALPHA",
          role: "drone",
          position: alphaReportedPos,
          heading: vehicleHeading,
          battery: 88,
          isSpoofed: isAlphaSpoofed,
          isCompromised: isAlphaSpoofed,
          status: isAlphaSpoofed
            ? "CONSENSUS QUARANTINED"
            : missionRunning
            ? "CORRIDOR CRUISE"
            : "HOVER PATROL",
          sensors: {
            gas: 12.4,
            obstacle: 35.0,
            gpsTrust: isAlphaSpoofed ? 0.15 : gpsTrust / 100
          }
        },
        {
          id: "UAV-BETA",
          name: "UAV-BETA",
          role: "drone",
          position: betaPosRef.current || [13.0608, 80.2809],
          heading: betaHeadingRef.current || 45,
          battery: 93,
          isSpoofed: false,
          isCompromised: false,
          status: isAlphaSpoofed
            ? "CROSS-VALIDATING LEADER"
            : swarmPatrolActive
            ? "ORBITAL SCAN PATROL"
            : "LOITER HOLD",
          sensors: {
            gas: 15.1,
            obstacle: 19.5,
            gpsTrust: 0.99
          }
        },
        {
          id: "AGV-OMEGA",
          name: "AGV-OMEGA",
          role: "cobot",
          position: omegaPosRef.current || OMEGA_PATROL_WAYPOINTS[0],
          heading: omegaHeadingRef.current || 90,
          battery: 82,
          isSpoofed: false,
          isCompromised: false,
          status:
            currentGas > 45
              ? "TOXIC PLUME DETECTED"
              : isAlphaSpoofed
              ? "GROUND ISOLATION ANCHOR"
              : swarmPatrolActive
              ? "GROUND AISLE RECON"
              : "STATIONARY HOLD",
          sensors: {
            gas: typeof currentGas === "number" ? Math.round(currentGas) : currentGas,
            obstacle: curObstacle,
            gpsTrust: 1.0
          }
        }
      ];
      setSwarmRobots(robots);

      // 5. Dynamic IoT Wireless Mesh Links (< 260m range)
      const links = [];
      for (let i = 0; i < robots.length; i++) {
        for (let j = i + 1; j < robots.length; j++) {
          const r1 = robots[i];
          const r2 = robots[j];
          const dist = distanceBetween(r1.position, r2.position);
          if (dist <= 260) {
            const isThreatLink = r1.isCompromised || r2.isCompromised;
            const rssi = Math.round(-38.0 - 25 * Math.log10(Math.max(1, dist)));
            links.push({
              from: r1.name,
              to: r2.name,
              fromPos: r1.position,
              toPos: r2.position,
              distanceM: dist,
              rssi: rssi,
              status: isThreatLink ? "threat" : dist > 180 ? "warning" : "nominal"
            });
          }
        }
      }
      setMeshLinks(links);

      // 6. Byzantine Consensus State
      if (isAlphaSpoofed) {
        setSwarmConsensusAlert(
          "🚨 SWARM CONSENSUS: UAV-ALPHA ISOLATED (CROSS-RANGING DISCREPANCY > 40m)"
        );
        setSharedHazards((prev) => {
          if (prev.some((h) => h.hazard_type === "GPS_SPOOFING")) return prev;
          return [
            ...prev,
            {
              hazard_type: "GPS_SPOOFING",
              reported_by: "UAV-BETA & AGV-OMEGA",
              position: alphaReportedPos,
              details: "Byzantine cross-check rejected UAV-ALPHA false coordinates"
            }
          ];
        });
      } else {
        setSwarmConsensusAlert(null);
      }

      // 7. Periodic Live IoT P2P Packet Generation (every ~1.4s)
      if (Date.now() - lastPacketLogTimeRef.current > 1400) {
        lastPacketLogTimeRef.current = Date.now();
        const timeStr = new Date().toLocaleTimeString();
        const randNode = Math.random() > 0.5 ? "AGV-OMEGA" : "UAV-BETA";
        const targetNode = randNode === "AGV-OMEGA" ? "UAV-ALPHA" : "AGV-OMEGA";
        const payloadStr =
          randNode === "AGV-OMEGA"
            ? `GAS_PPM=${currentGas}, SONAR_DIST=${curObstacle}m, BATT=82%`
            : `FLIR_TEMP=28.4C, LIDAR_ALT=38m, BATT=93%`;
        const rssiVal = Math.round(-48 - Math.random() * 14);

        setIotPacketLog((prev) => [
          {
            id: `pkt-${Date.now()}-${Math.random()}`,
            time: timeStr,
            type: "P2P_TELEMETRY",
            sender: randNode,
            target: targetNode,
            channel: "CH_11_802.15.4",
            seq: packetSeqRef.current++,
            rssi: rssiVal,
            payload: payloadStr
          },
          ...prev.slice(0, 24)
        ]);
      }
    }, 140);

    return () => clearInterval(timer);
  }, [
    swarmMode,
    swarmPatrolActive,
    simulationSpeed,
    vehiclePosition,
    vehicleHeading,
    source,
    spoofing,
    gpsPosition,
    gpsDivergence,
    quarantined,
    gpsTrust,
    missionRunning,
    environmentalHazards,
    addAlert
  ]);

  // Wall Boundary Check Function (Only strictly checks image boundaries now, AI handles paths)
  const isWall = useCallback((lat, lng) => {
    if (envMode !== "factory") return false;

    const centerLat = 13.0600;
    const centerLng = 80.2800;
    const lat_diff = floorPlanScale / 111320;
    const lng_diff = floorPlanScale / (111320 * Math.cos(centerLat * Math.PI / 180));

    const minLat = centerLat - lat_diff / 2;
    const maxLat = centerLat + lat_diff / 2;
    const minLng = centerLng - lng_diff / 2;
    const maxLng = centerLng + lng_diff / 2;

    const normalizedX = (lng - minLng) / lng_diff;
    const normalizedY = 1.0 - ((lat - minLat) / lat_diff);

    // 1. If outside the image bounds, it is a boundary wall
    if (normalizedX < 0 || normalizedX >= 1 || normalizedY < 0 || normalizedY >= 1) return true;
    
    // 2. If pixel-level floor plan data is loaded, check interior obstacle collision
    if (floorPlanImageDataRef.current && collisionImageWidthRef.current && collisionImageHeightRef.current) {
      const px = Math.floor(normalizedX * collisionImageWidthRef.current);
      const py = Math.floor(normalizedY * collisionImageHeightRef.current);
      return checkPixelWall(
        px,
        py,
        floorPlanImageDataRef.current,
        collisionImageWidthRef.current,
        collisionImageHeightRef.current,
        floorPlanDarkIsWallRef.current
      );
    }

    return false;
  }, [envMode, floorPlanScale]);

  // Compute a path along the Custom Graph (A*)
  const computeGraphPath = useCallback((src, dst, nodes, edges) => {
    if (!src || !dst || nodes.length < 2 || edges.length === 0) return null;

    let closestSrcIdx = 0; let minSrcDist = Infinity;
    nodes.forEach((node, idx) => {
      const d = distanceBetween(src, node);
      if (d < minSrcDist) { minSrcDist = d; closestSrcIdx = idx; }
    });

    let closestDstIdx = 0; let minDstDist = Infinity;
    nodes.forEach((node, idx) => {
      const d = distanceBetween(dst, node);
      if (d < minDstDist) { minDstDist = d; closestDstIdx = idx; }
    });

    const adj = Array.from({ length: nodes.length }, () => []);
    edges.forEach(([u, v]) => {
      // Ignore malformed edges rather than allowing an invalid map click to
      // break route generation. A route must use only the drawn connections.
      if (
        !Number.isInteger(u) ||
        !Number.isInteger(v) ||
        u === v ||
        !nodes[u] ||
        !nodes[v]
      ) return;

      const dist = distanceBetween(nodes[u], nodes[v]);
      adj[u].push({ target: v, weight: dist });
      adj[v].push({ target: u, weight: dist });
    });

    const gScore = Array(nodes.length).fill(Infinity);
    const fScore = Array(nodes.length).fill(Infinity);
    const prev = Array(nodes.length).fill(null);
    gScore[closestSrcIdx] = 0;
    fScore[closestSrcIdx] = distanceBetween(nodes[closestSrcIdx], nodes[closestDstIdx]);

    const pq = [{ node: closestSrcIdx, f: fScore[closestSrcIdx] }];

    while (pq.length > 0) {
      pq.sort((a, b) => a.f - b.f);
      const { node: u } = pq.shift();

      if (u === closestDstIdx) break;

      adj[u].forEach(edge => {
        const v = edge.target;
        const tentativeG = gScore[u] + edge.weight;
        if (tentativeG < gScore[v]) {
          gScore[v] = tentativeG;
          fScore[v] = tentativeG + distanceBetween(nodes[v], nodes[closestDstIdx]);
          prev[v] = u;
          pq.push({ node: v, f: fScore[v] });
        }
      });
    }

    if (gScore[closestDstIdx] === Infinity) return null;

    const pathIdxs = [];
    let u = closestDstIdx;
    while (u !== null) {
      pathIdxs.unshift(u);
      u = prev[u];
    }

    const rawPath = [];
    pathIdxs.forEach(idx => rawPath.push(nodes[idx]));

    // Interpolate the path so the robot moves smoothly
    let densePath = [];
    let totalDist = 0;
    for (let i = 0; i < rawPath.length - 1; i++) {
      const segment = generateAirCorridor(rawPath[i], rawPath[i + 1], 40);
      let p = segment.path;
      if (i > 0) p.shift(); // Avoid duplicating connection points
      densePath = densePath.concat(p);
      totalDist += segment.distance;
    }

    // A start and destination on the same dot still need a valid route value.
    if (densePath.length === 0) densePath = [rawPath[0]];

    return { path: densePath, distance: totalDist, duration: totalDist / 12 };
  }, []);

  // Compute Route between source and destination
  const computeRoute = useCallback(async (src, dst) => {
    if (!src || !dst) return;
    setRouteLoading(true);
    try {
      addAlert("Calculating flight path...", "info");

      let result;
      if (envMode === "factory") {
        addAlert("Indoor Factory Mode: Computing route through the manual dotted network...", "info");
        const graphResult = computeGraphPath(src, dst, customNodes, customEdges);
        if (graphResult) {
          result = graphResult;
          addAlert("✅ Manual graph path found. The UAV will follow the connected yellow segments.", "success");
        } else {
          setRoute([]);
          setDistance(null);
          setDuration(null);
          setSecurityState("CONNECT NAVMESH");
          addAlert("❌ No connected dotted path exists between Start and Destination. Connect the selected dots, then try again.", "warning");
          return;
        }
      } else {
        result = await getRoute(src, dst);
      }

      setRoute(result.path);
      setDistance(result.distance);
      setDuration(result.duration);

      const startPt = result.path[0];
      setVehiclePosition(startPt);
      vehiclePosRef.current = startPt;
      setGpsPosition(startPt);
      gpsPosRef.current = startPt;

      if (result.path.length > 1) {
        setVehicleHeading(calculateBearing(result.path[0], result.path[1]));
      }

      setBreadcrumbTrail([startPt]);
      routeIndexRef.current = 0;
      setRecoveryRoute([]);
      recoveryRouteRef.current = [];
      setSecurityState("ROUTE READY");
      setRiskScore(5);
      setGpsTrust(100);
      setGpsDivergence(0);
      setSelectionMode(null);
      
      const distFormatted = envMode === "factory" 
        ? `${result.distance.toFixed(1)} m`
        : `${(result.distance / 1000).toFixed(2)} km`;
      addAlert(`Flight path ready: ${distFormatted}. Ready to Launch!`, "success");
    } catch (err) {
      console.error(err);
      addAlert("Could not compute route. Try selecting another point.", "danger");
    } finally {
      setRouteLoading(false);
    }
  }, [addAlert, computeGraphPath, customEdges, customNodes, envMode]);

  // Initial welcome message
  useEffect(() => {
    addAlert("Ready: Click anywhere on the map to place your INITIAL START point.", "info");
  }, [addAlert]);

  // Handle File Upload for Floor Plan
  const handleFloorPlanUpload = (e) => {
    if (e.preventDefault) e.preventDefault();
    if (e.stopPropagation) e.stopPropagation();

    const file = e.target?.files?.[0] || e.dataTransfer?.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setFloorPlanUrl(objectUrl);
    setEnvMode("factory");
    setFactoryPhase("draw_dots");
    setSelectedDotForEdge(null);
    setCustomNodes([]);
    setCustomEdges([]);
    setManualWaypoints([]);
    setRoute([]);
    setSource(null);
    setDestination(null);
    setSelectionMode("none");
    addAlert("📥 Floor plan uploaded! Phase 1: Click anywhere on the floor plan to place your yellow dots.", "info");
  };

  // Environment Mode Switcher Handler
  const handleEnvModeChange = (newMode) => {
    setEnvMode(newMode);
    if (newMode === "factory") {
      setFactoryPhase("draw_dots");
      addAlert("Switched to Factory Indoor Mode. Click map to place yellow dots or upload your floor plan.", "info");
    } else {
      addAlert("Switched to Outdoor Drone GPS Mode.", "info");
      const p = MISSION_PRESETS[0];
      setSource(p.source);
      setDestination(p.destination);
      setVehiclePosition(p.source);
      vehiclePosRef.current = p.source;
      setGpsPosition(p.source);
      gpsPosRef.current = p.source;
      computeRoute(p.source, p.destination);
    }
  };

  // Master Map Click Handler: Supports setting start, destination, and spoofing anytime!
  const handleMapClick = (pos) => {
    // If high-frequency override is active, strictly reject new clicks/spoofs
    if (overrideToStartRef.current) {
      addAlert("🛑 DESTINATION LOCKED TO START: High-frequency cyber attack confirmed (>5 spoofing in 5s). Drone goes ONLY towards starting point!", "danger");
      return;
    }

    // 🔥 Factory Mode Phase 1: Manual Waypoint Drawing 🔥
    if (envMode === "factory" && factoryPhase === "draw_dots") {
      setCustomNodes((prev) => [...prev, pos]);
      addAlert(`Dot placed. Total dots: ${customNodes.length + 1}.`, "success");
      return;
    }

    // 🔥 Factory Mode Phase 2: Connecting Dots 🔥
    if (envMode === "factory" && factoryPhase === "connect_dots") {
      if (customNodes.length < 2) return;

      // Find the closest dot
      let minDist = Infinity;
      let closestIdx = -1;
      customNodes.forEach((node, idx) => {
        const d = distanceBetween(pos, node);
        if (d < minDist) { minDist = d; closestIdx = idx; }
      });

      if (closestIdx === -1) return;

      if (selectedDotForEdge === null) {
        setSelectedDotForEdge(closestIdx);
        addAlert("Dot selected. Click the next dot to connect them.", "info");
      } else if (selectedDotForEdge === closestIdx) {
        setSelectedDotForEdge(null);
        addAlert("Dot deselected.", "info");
      } else {
        setCustomEdges((prev) => [...prev, [selectedDotForEdge, closestIdx]]);
        setSelectedDotForEdge(closestIdx); // Automatically chain to the new dot!
        addAlert("Connected! Click another dot to continue the chain.", "success");
      }
      return;
    }

    // Helper: Snap position to nearest custom node in factory mode Phase 2
    let snappedPos = pos;
    if (envMode === "factory" && factoryPhase === "ready" && customNodes.length > 0) {
      let minDist = Infinity;
      customNodes.forEach(node => {
        const d = distanceBetween(pos, node);
        if (d < minDist) { minDist = d; snappedPos = node; }
      });
    }

    // 1. If currently choosing Start:
    if (selectionMode === "start" || !source) {
      setSource(snappedPos);
      setVehiclePosition(snappedPos);
      vehiclePosRef.current = snappedPos;
      setGpsPosition(snappedPos);
      gpsPosRef.current = snappedPos;
      setRoute([]);
      setRecoveryRoute([]);
      setBreadcrumbTrail([snappedPos]);
      setSelectionMode("dest");
      setSecurityState("SELECT DESTINATION");
      addAlert(`Start point snapped to node. Now click for Destination.`, "success");
      return;
    }

    // 2. If currently choosing Destination:
    if (selectionMode === "dest" || (!destination && source)) {
      setDestination(snappedPos);
      setSelectionMode(null);
      addAlert(`Final Destination snapped to node.`, "success");
      computeRoute(source, snappedPos);
      return;
    }

    // 3. If route already exists (at ANY time of launch or flight):
    // Clicking anywhere on the map sets or updates the Spoof Target location!
    injectOrUpdateSpoofTarget(pos);
  };

  // Drag handlers for interactive fine-tuning
  const handleSourceDrag = (newPos) => {
    setSource(newPos);
    setVehiclePosition(newPos);
    vehiclePosRef.current = newPos;
    setGpsPosition(newPos);
    gpsPosRef.current = newPos;
    if (destination) {
      computeRoute(newPos, destination);
    }
  };

  const handleDestDrag = (newPos) => {
    if (overrideToStartRef.current) {
      addAlert("🛑 DESTINATION LOCKED: Overridden to starting point under high-frequency cyber attack.", "danger");
      return;
    }
    setDestination(newPos);
    if (source) {
      computeRoute(source, newPos);
    }
  };

  const handleSpoofDrag = (newPos) => {
    injectOrUpdateSpoofTarget(newPos);
  };

  // Inject or Update Spoof Target at ANY time (like demo_animated_sentry)
  function injectOrUpdateSpoofTarget(targetPos) {
    // 0. If return-to-start override is already active, reject any new targets
    if (overrideToStartRef.current) {
      addAlert("🛑 DESTINATION LOCKED TO START: High-frequency cyber attack confirmed (>5 spoofing in 5s). Drone goes ONLY towards starting point!", "danger");
      return;
    }

    const now = Date.now();
    spoofTimestampsRef.current.push(now);
    // Filter to attacks within the last 5 seconds (5000 ms)
    spoofTimestampsRef.current = spoofTimestampsRef.current.filter((t) => now - t <= 5000);

    // CRITICAL USER REQUIREMENT: More than 5 spoofing found in 5 seconds!
    if (spoofTimestampsRef.current.length > 5) {
      overrideToStartRef.current = true;
      setOverrideToStart(true);
      spoofingRef.current = false;
      setSpoofing(false);
      quarantinedRef.current = true;
      setQuarantined(true);
      setSpoofTarget(null);
      spoofTargetRef.current = null;

      const currentVeh = vehiclePosRef.current || vehiclePosition;
      const startPt = source || (route.length > 0 ? route[0] : null);

      // OVERRIDE THE DESTINATION WITH STARTING POINT:
      // No matter what the destination is, drone will go ONLY towards starting point!
      if (startPt) {
        setDestination(startPt);
      }
      setSecurityState("HIGH-FREQ HIJACK // RETURN TO START");
      setRiskScore(100);
      setGpsTrust(0);

      addAlert("🚨 HIGH-FREQUENCY CYBER ATTACK CONFIRMED (>5 spoofing attacks in 5s)!", "danger");
      addAlert("🛑 EMERGENCY OVERRIDE ENGAGED: Destination REPLACED with STARTING POINT.", "danger");
      addAlert("✈ DRONE ENFORCED: Navigating ONLY towards initial starting point!", "warning");

      if (currentVeh && startPt) {
        triggerReturnToStartRecovery(currentVeh, startPt);
      }
      return;
    }

    // 1. Immediately cancel any running recovery so new spoof attack takes full effect
    recoveryActiveRef.current = false;
    setRecoveryActive(false);
    recoveryRouteRef.current = [];
    setRecoveryRoute([]);
    recoveryIndexRef.current = 0;

    // 2. Clear quarantine to allow fresh spoof drift
    quarantinedRef.current = false;
    setQuarantined(false);

    // 3. Set the new spoof target
    setSpoofTarget(targetPos);
    spoofTargetRef.current = targetPos;
    setSpoofing(true);
    spoofingRef.current = true;
    spoofStartTimeRef.current = Date.now();

    // 4. Reset GPS position to where vehicle is right now so divergence builds afresh from 0m
    const currentVeh = vehiclePosRef.current || vehiclePosition || (route.length > 0 ? route[0] : null);
    if (currentVeh) {
      gpsPosRef.current = [currentVeh[0], currentVeh[1]];
      setGpsPosition([currentVeh[0], currentVeh[1]]);
      // Immediately rotate drone towards the new target
      setVehicleHeading(calculateBearing(currentVeh, targetPos));
    }
    setGpsDivergence(0);
    setGpsTrust(90);
    setRiskScore(35);

    // 5. If mission wasn't running, start it automatically so user sees immediate motion!
    if (!missionRunningRef.current) {
      setMissionRunning(true);
      missionRunningRef.current = true;
      setMissionPaused(false);
      missionPausedRef.current = false;
    }

    setSecurityState("ATTACK ACTIVE // HIJACK DRIFT");
    setSpoofCount((c) => c + 1);

    addAlert(`⚡ SPOOF TARGET UPDATED to [${targetPos[0].toFixed(4)}, ${targetPos[1].toFixed(4)}]!`, "danger");
    addAlert("Autopilot deceived: UAV redirected and actively flying toward spoof location.", "danger");
  };

  // Clear & Pick New Custom Route from scratch
  const clearAndPickNewRoute = () => {
    resetSimulation();
    setSource(null);
    setDestination(null);
    setRoute([]);
    setVehiclePosition(null);
    vehiclePosRef.current = null;
    setGpsPosition(null);
    gpsPosRef.current = null;
    setSelectionMode("start");
    setSecurityState("CLICK MAP FOR START");
    addAlert("Route cleared. Click anywhere on the map to set your INITIAL START point.", "info");
  };

  // Quick Preset Loader (optional convenience)
  const handleQuickPreset = (presetId) => {
    const preset = MISSION_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSource(preset.source);
    setDestination(preset.destination);
    setSelectionMode(null);
    resetSimulation();
    computeRoute(preset.source, preset.destination);
  };

  // Mission Launch / Pause
  const toggleMission = () => {
    if (!route.length) {
      addAlert("Please pick Initial and Final points first.", "warning");
      return;
    }
    if (!missionRunning) {
      setMissionRunning(true);
      missionRunningRef.current = true;
      setMissionPaused(false);
      missionPausedRef.current = false;
      setSecurityState(spoofingRef.current ? "ATTACK ACTIVE // HIJACK DRIFT" : "FLIGHT NOMINAL");
      addAlert("🚀 MISSION LAUNCHED: UAV airborne and navigating with real-time EKF fusion.", "success");
    } else {
      const nextPaused = !missionPaused;
      setMissionPaused(nextPaused);
      missionPausedRef.current = nextPaused;
      addAlert(nextPaused ? "Flight paused." : "Flight resumed.", "info");
    }
  };

  // Reset Simulation
  const resetSimulation = () => {
    setMissionRunning(false);
    missionRunningRef.current = false;
    setMissionPaused(false);
    missionPausedRef.current = false;
    setSpoofing(false);
    spoofingRef.current = false;
    setQuarantined(false);
    quarantinedRef.current = false;
    setRecoveryActive(false);
    recoveryActiveRef.current = false;
    setRecoveryProgress(0);
    setSpoofTarget(null);
    spoofTargetRef.current = null;
    setRecoveryRoute([]);
    recoveryRouteRef.current = [];
    setManualWaypoints([]);
    routeIndexRef.current = 0;
    recoveryIndexRef.current = 0;
    setRiskScore(4);
    setGpsTrust(100);
    setGpsDivergence(0);
    overrideToStartRef.current = false;
    setOverrideToStart(false);
    spoofTimestampsRef.current = [];
    setSecurityState(route.length ? "ROUTE READY" : "SELECT WAYPOINTS");

    if (route.length > 0) {
      const startPt = route[0];
      setVehiclePosition(startPt);
      vehiclePosRef.current = startPt;
      setGpsPosition(startPt);
      gpsPosRef.current = startPt;
      setBreadcrumbTrail([startPt]);
      if (route.length > 1) {
        setVehicleHeading(calculateBearing(route[0], route[1]));
      }
    }
  };

  // Stop current attack
  const stopAttack = () => {
    setSpoofing(false);
    spoofingRef.current = false;
    setSpoofTarget(null);
    spoofTargetRef.current = null;
    if (!quarantinedRef.current && !overrideToStartRef.current) {
      setSecurityState("FLIGHT NOMINAL");
      setRiskScore(8);
      setGpsTrust(100);
      setGpsDivergence(0);
      if (vehiclePosRef.current) {
        setGpsPosition(vehiclePosRef.current);
        gpsPosRef.current = vehiclePosRef.current;
      }
      addAlert("Attack beacon deactivated. Autopilot resumed nominal corridor cruise.", "success");
    }
  };

  // Return-to-Start Recovery Corridor Generation (Overrides destination with starting point)
  const triggerReturnToStartRecovery = useCallback((currentPos, startPt) => {
    if (!startPt || !currentPos) return;
    addAlert("🛑 HIGH-FREQUENCY HIJACKING: Destination OVERRIDDEN with STARTING POINT.", "danger");
    addAlert("Direct emergency air corridor locked directly to initial launch coordinates.", "info");

    let returnCorridor;
    if (envMode === "factory") {
      // Just draw a straight line but the physics engine isWall check will prevent it from going out of bounds
      returnCorridor = generateAirCorridor(currentPos, startPt, 50);
    } else {
      returnCorridor = generateAirCorridor(currentPos, startPt, 50);
    }

    setDestination(startPt);
    setRoute(returnCorridor.path);
    routeIndexRef.current = 0;
    setRecoveryRoute(returnCorridor.path);
    recoveryRouteRef.current = returnCorridor.path;
    recoveryIndexRef.current = 0;
    setRecoveryActive(true);
    recoveryActiveRef.current = true;
    setSecurityState("RETURNING TO START POINT");
    setRiskScore(95);
    setGpsTrust(0);
    setGpsDivergence(0);
    addAlert("✈ Drone is now flying ONLY towards starting point. All new destinations blocked.", "warning");
  }, [addAlert]);

  // Autonomous Recovery Rejoin Route Generation
  const triggerAutonomousRecovery = useCallback(async (currentPos) => {
    if (!route.length) return;

    // If high-frequency override is active (>5 spoofs in 5s): ALWAYS return to starting point!
    if (overrideToStartRef.current || spoofTimestampsRef.current.length > 5) {
      overrideToStartRef.current = true;
      setOverrideToStart(true);
      const startPt = source || route[0];
      if (startPt) {
        setDestination(startPt);
        triggerReturnToStartRecovery(currentPos, startPt);
      }
      return;
    }

    try {
      addAlert("🛡 EKF QUARANTINE ENGAGED: GNSS fix isolated. Holding safe course.", "danger");
      addAlert("HMAC-SHA256 Server Validation: Calculating dynamic rejoin corridor...", "info");

      // Find nearest waypoint on authorized route
      let nearestIdx = 0;
      let minDst = Infinity;
      route.forEach((pt, idx) => {
        const dst = distanceBetween(currentPos, pt);
        if (dst < minDst) {
          minDst = dst;
          nearestIdx = idx;
        }
      });

      // Target a rejoin point safely ahead on the planned route
      const rejoinIdx = Math.min(nearestIdx + 12, route.length - 1);
      const rejoinPoint = route[rejoinIdx];

      let recoveryResult;
      if (envMode === "factory") {
        // Fallback to straight line for recovery, but physics engine will block walls
        recoveryResult = { path: [currentPos, rejoinPoint] };
      } else {
        recoveryResult = await getRecoveryRoute(currentPos, rejoinPoint);
      }

      const fullRecoveryPath = [...recoveryResult.path, ...route.slice(rejoinIdx)];

      recoveryRouteRef.current = fullRecoveryPath;
      recoveryIndexRef.current = 0;
      setRecoveryRoute(fullRecoveryPath);
      setRecoveryActive(true);
      recoveryActiveRef.current = true;
      setSecurityState("AUTONOMOUS RECOVERY");
      addAlert(`HMAC authorization verified! Rejoining planned route at waypoint ${rejoinIdx + 1}.`, "success");
    } catch (err) {
      console.error(err);
      const fallback = [
        currentPos,
        route[Math.min(routeIndexRef.current + 10, route.length - 1)]
      ];
      recoveryRouteRef.current = fallback;
      recoveryIndexRef.current = 0;
      setRecoveryRoute(fallback);
      setRecoveryActive(true);
      recoveryActiveRef.current = true;
    }
  }, [route, addAlert]);

  // Main Motion Physics Loop (Precisely matches demo_animated_sentry.py)
  useEffect(() => {
    if (!missionRunning || missionPaused || !route.length) return;

    const intervalMs = Math.max(70, Math.round(340 / simulationSpeed));
    const timer = setInterval(() => {
      const curPos = vehiclePosRef.current || vehiclePosition;
      if (!curPos) return;

      // ── Priority 0: High-Frequency Cyber Attack Return-to-Start Override ──
      // "no matter what the destination is, drone will go only towards starting point"
      if (overrideToStartRef.current) {
        const startTarget = source || (route.length > 0 ? route[0] : null);
        if (startTarget) {
          let target = startTarget;
          const rPath = recoveryRouteRef.current;
          const rIdx = recoveryIndexRef.current;
          if (rPath && rPath.length > 0 && rIdx < rPath.length - 1) {
            target = rPath[rIdx + 1];
          }

          const next = moveTowards(curPos, target, 0.00060 * simulationSpeed);
          vehiclePosRef.current = next;
          setVehiclePosition(next);
          setVehicleHeading(calculateBearing(curPos, target));
          setBreadcrumbTrail((prev) => [...prev.slice(-100), next]);

          if (rPath && rPath.length > 0) {
            if (distanceBetween(next, target) < 22) {
              recoveryIndexRef.current = Math.min(rIdx + 1, rPath.length - 1);
            }
            setRecoveryProgress(recoveryIndexRef.current / Math.max(1, rPath.length - 1));
          }

          const distToStart = distanceBetween(next, startTarget);
          if (distToStart < 22) {
            vehiclePosRef.current = startTarget;
            setVehiclePosition(startTarget);
            setMissionRunning(false);
            missionRunningRef.current = false;
            setRecoveryActive(false);
            recoveryActiveRef.current = false;
            setRecoveryProgress(1);
            setSecurityState("SECURED AT STARTING POINT");
            setRiskScore(0);
            setGpsTrust(100);
            setGpsDivergence(0);
            addAlert("✅ SAFE AT STARTING POINT: Return-to-start completed under high-frequency cyber attack override.", "success");
            addAlert("High-frequency attack neutralized. Drone secured at initial departure location.", "success");
            return;
          }
        }
        return; // CRITICAL: NEVER proceed to spoof drift or normal forward flight!
      }

      // ── Scenario A: Recovery in Progress ──────────────────────────────
      if (recoveryActiveRef.current && recoveryRouteRef.current.length > 0) {
        const rPath = recoveryRouteRef.current;
        const rIdx = recoveryIndexRef.current;

        if (rIdx >= rPath.length - 1) {
          if (overrideToStartRef.current) {
            setRecoveryActive(false);
            recoveryActiveRef.current = false;
            setRecoveryProgress(1);
            setMissionRunning(false);
            missionRunningRef.current = false;
            setSecurityState("SECURED AT STARTING POINT");
            setRiskScore(0);
            setGpsTrust(100);
            setGpsDivergence(0);
            addAlert("✅ MISSION TERMINATED SAFELY: Drone returned and secured at STARTING POINT.", "success");
            addAlert("High-frequency attack fully neutralized. Vehicle secured at launch coordinates.", "success");
            return;
          }

          // Rejoin Complete!
          setRecoveryActive(false);
          recoveryActiveRef.current = false;
          setRecoveryProgress(1);
          setSecurityState("REJOINED // NOMINAL");
          setRiskScore(12);
          setRecoveryRoute([]);
          recoveryRouteRef.current = [];
          addAlert("✅ UAV REJOINED AUTHORIZED FLIGHT CORRIDOR.", "success");
          addAlert("GNSS quarantined. Optical/LiDAR odometry continuing to destination.", "success");

          // Sync main route index to closest point
          let bestIdx = route.length - 1;
          let bestDist = Infinity;
          route.forEach((p, idx) => {
            const d = distanceBetween(curPos, p);
            if (d < bestDist) {
              bestDist = d;
              bestIdx = idx;
            }
          });
          routeIndexRef.current = bestIdx;
          return;
        }

        const target = rPath[rIdx + 1];
        const next = moveTowards(curPos, target, 0.00045 * simulationSpeed);
        vehiclePosRef.current = next;
        setVehiclePosition(next);
        setVehicleHeading(calculateBearing(curPos, target));
        setBreadcrumbTrail((prev) => [...prev.slice(-100), next]);

        if (distanceBetween(next, target) < 22) {
          recoveryIndexRef.current = rIdx + 1;
        }
        setRecoveryProgress(recoveryIndexRef.current / Math.max(1, rPath.length - 1));
        return;
      }

      // ── Scenario B: Under Active GNSS Spoofing (Autopilot deceived) ───
      if (spoofingRef.current && spoofTargetRef.current && !quarantinedRef.current) {
        const target = spoofTargetRef.current;
        const deceivedStep = moveTowards(curPos, target, 0.00040 * simulationSpeed);

        // Pixel-Level Wall Collision Check for Spoofed Drifting
        if (isWall(deceivedStep[0], deceivedStep[1])) {
          addAlert("💥 SPOOF COLLISION ALERT: Hijack drift blocked by wall or boundary!", "warning");
          // Drone is physically blocked, but GPS still drifts!
        } else {
          vehiclePosRef.current = deceivedStep;
          setVehiclePosition(deceivedStep);
        }

        setVehicleHeading(calculateBearing(curPos, target));
        setBreadcrumbTrail((prev) => [...prev.slice(-100), deceivedStep]);

        // Fake GPS drifts even faster toward the spoof target
        const curGps = gpsPosRef.current || curPos;
        const fakeGps = moveTowards(curGps, target, 0.00075 * simulationSpeed);
        gpsPosRef.current = fakeGps;
        setGpsPosition(fakeGps);

        // Compute cross-sensor divergence
        const divergence = distanceBetween(deceivedStep, fakeGps);
        setGpsDivergence(divergence);

        // Dynamic risk & trust evaluation
        const currentRisk = Math.min(95, Math.round(25 + divergence * 0.85));
        const currentTrust = Math.max(5, Math.round(100 - divergence * 0.95));
        setRiskScore(currentRisk);
        setGpsTrust(currentTrust);

        // Chi-Square & CUSUM Gate Alarm Trigger
        if (divergence >= DETECTION_THRESHOLD) {
          setQuarantined(true);
          quarantinedRef.current = true;
          setSpoofing(false);
          spoofingRef.current = false;
          setGpsTrust(0);
          setRiskScore(100);
          setSecurityState(overrideToStartRef.current ? "RETURNING TO START POINT" : "GPS QUARANTINED");

          addAlert(`🚨 ANOMALY: GPS Divergence = ${Math.round(divergence)}m (Limit: ${DETECTION_THRESHOLD}m)`, "danger");
          addAlert("Chi-Square NIS Gate: EXCEEDED (p < 0.0001). Innovation rejected.", "danger");
          addAlert("CUSUM Drift Monitor: PERSISTENT DRIFT CONFIRMED.", "danger");
          addAlert("SAARM Bank: GNSS Receiver Isolated. Primary flight control switched to EKF.", "warning");

          if (overrideToStartRef.current) {
            const startPt = source || route[0];
            triggerReturnToStartRecovery(deceivedStep, startPt);
          } else {
            triggerAutonomousRecovery(deceivedStep);
          }
        }
        return;
      }

      // ── Scenario C: Normal Flight along Authorized Corridor ───────────
      const cIndex = routeIndexRef.current;
      if (cIndex >= route.length - 1) {
        setMissionRunning(false);
        missionRunningRef.current = false;
        setSecurityState("MISSION COMPLETE");
        setRiskScore(0);
        addAlert("🎯 TARGET DESTINATION REACHED SAFELY.", "success");
        return;
      }

      const nextIndex = Math.min(cIndex + 1, route.length - 1);
      const nextPos = route[nextIndex];

      // Pixel-Level Wall Collision & Boundary Check
      if (isWall(nextPos[0], nextPos[1])) {
        addAlert("💥 COLLISION ALERT: Flight path blocked by factory wall!", "danger");
        setMissionRunning(false);
        missionRunningRef.current = false;
        setSecurityState("CRASH: WALL COLLISION");
        return;
      }

      routeIndexRef.current = nextIndex;
      vehiclePosRef.current = nextPos;
      setVehiclePosition(nextPos);
      setVehicleHeading(calculateBearing(curPos, nextPos));
      setBreadcrumbTrail((prev) => [...prev.slice(-100), nextPos]);

      // If not spoofing, raw GPS accurately tracks trusted position
      if (!spoofingRef.current && !quarantinedRef.current) {
        gpsPosRef.current = nextPos;
        setGpsPosition(nextPos);
        setGpsDivergence(0);
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [missionRunning, missionPaused, route, simulationSpeed, triggerAutonomousRecovery, addAlert]);

  const riskStatus =
    riskScore >= 80 ? "CRITICAL" : riskScore >= 55 ? "HIGH" : riskScore >= 25 ? "ELEVATED" : "NOMINAL";

  return (
    <div className="sentry-root-canvas">
      {/* ── 1. The Interactive Real Map ────────────────────────────────── */}
      <SensorMap
        source={source}
        destination={destination}
        route={route}
        vehiclePosition={vehiclePosition}
        vehicleHeading={vehicleHeading}
        gpsPosition={gpsPosition}
        spoofing={spoofing}
        spoofTarget={spoofTarget}
        recoveryRoute={recoveryRoute}
        breadcrumbTrail={breadcrumbTrail}
        selectionMode={selectionMode}
        missionRunning={missionRunning}
        recoveryActive={recoveryActive}
        quarantined={quarantined}
        overrideToStart={overrideToStart}
        followVehicle={followVehicle}
        onMapClick={handleMapClick}
        onSourceDrag={handleSourceDrag}
        onDestDrag={handleDestDrag}
        onSpoofDrag={handleSpoofDrag}
        envMode={envMode}
        floorPlanUrl={floorPlanUrl}
        floorPlanScale={floorPlanScale}
        customNodes={customNodes}
        customEdges={customEdges}
        onFloorPlanUpload={handleFloorPlanUpload}
        fileInputRef={fileInputRef}
        manualWaypoints={manualWaypoints}
        selectedDotForEdge={selectedDotForEdge}
        swarmMode={swarmMode}
        swarmRobots={swarmRobots}
        meshLinks={meshLinks}
        sharedHazards={sharedHazards}
        environmentalHazards={environmentalHazards}
      />

      {/* ── 2. Floating Top Flight HUD Bar ─────────────────────────────── */}
      <header className="hud-top-bar">
        <div className="hud-brand">
          <div className="hud-logo-pulse">
            <ShieldCheck size={24} />
          </div>
          <div className="hud-titles">
            <h1>SENSORSENTRY</h1>
            <span className="hud-subtitle">ANTI-SPOOFING TACTICAL FLIGHT HUD</span>
          </div>
        </div>

        {/* Environment Mode Switcher */}
        <div style={{ display: "flex", gap: "8px", alignItems: "center", background: "rgba(15,23,42,0.75)", padding: "4px 8px", borderRadius: "6px", border: "1px solid rgba(0,255,157,0.3)" }}>
          <select 
            value={envMode} 
            onChange={(e) => handleEnvModeChange(e.target.value)}
            style={{ background: "#0f172a", color: "#00ff9d", border: "1px solid #00ff9d", padding: "4px 8px", borderRadius: "4px", fontWeight: "bold", fontSize: "11px", cursor: "pointer" }}
          >
            <option value="outdoor">🌍 Drone (Outdoor GPS)</option>
            <option value="factory">🏭 Factory (Indoor Floor Plan)</option>
          </select>

          {envMode === "factory" && (
            <>
              {/* Reliable Button Trigger with ref */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="Click or drop custom floor plan image (PNG, JPG, WebP)"
                style={{ 
                  border: floorPlanUrl ? "1px solid #00ff9d" : "1.5px dashed #00ff9d", 
                  padding: "4px 10px", 
                  borderRadius: "4px", 
                  color: "#00ff9d", 
                  fontSize: "11px",
                  background: floorPlanUrl ? "rgba(0,255,157,0.15)" : "rgba(0,255,157,0.06)",
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                  fontWeight: 500
                }}
              >
                <span>{floorPlanUrl ? "✅ Floor Plan Loaded" : "📥 Upload Plan"}</span>
              </button>
              <input 
                ref={fileInputRef}
                type="file" 
                accept="image/*"
                style={{ display: "none" }}
                onClick={(e) => { e.target.value = null; }}
                onChange={handleFloorPlanUpload}
              />

              {/* Floor Plan Real-World Scale */}
              <div style={{ display: "flex", alignItems: "center", gap: "3px" }}>
                <span style={{ fontSize: "10px", color: "#94a3b8" }}>Scale:</span>
                <input 
                  type="number" 
                  value={floorPlanScale} 
                  onChange={(e) => setFloorPlanScale(Math.max(10, Number(e.target.value)))}
                  style={{ width: "45px", background: "#090d16", color: "#fff", border: "1px solid #334155", padding: "2px 4px", borderRadius: "3px", fontSize: "11px", textAlign: "center" }}
                  title="Width in meters"
                />
                <span style={{ fontSize: "10px", color: "#64748b" }}>m</span>
              </div>

              {/* Phase 1: Confirm Dots & Draw Lines */}
              {factoryPhase === "draw_dots" && (
                <button
                  type="button"
                  style={{ background: "#facc15", color: "#000", border: "1px solid #facc15", padding: "4px 10px", borderRadius: "4px", fontWeight: "bold", fontSize: "11px", cursor: "pointer" }}
                  onClick={() => {
                    if (customNodes.length < 2) {
                      addAlert("Please place at least 2 yellow dots on the map.", "warning");
                      return;
                    }
                    setFactoryPhase("connect_dots");
                    setSelectedDotForEdge(null);
                    addAlert("Phase 2: Click on any yellow dot (turns red), then click another dot to draw a connecting line.", "info");
                  }}
                >
                  Confirm Dots & Draw Lines ({customNodes.length})
                </button>
              )}

              {/* Phase 2: Finish Network */}
              {factoryPhase === "connect_dots" && (
                <button
                  type="button"
                  style={{ background: "#00ff9d", color: "#000", border: "1px solid #00ff9d", padding: "4px 10px", borderRadius: "4px", fontWeight: "bold", fontSize: "11px", cursor: "pointer" }}
                  onClick={() => {
                    if (customEdges.length === 0) {
                      addAlert("Connect at least two dots before finishing the network.", "warning");
                      return;
                    }
                    setFactoryPhase("ready");
                    setSelectionMode("start");
                    setSelectedDotForEdge(null);
                    addAlert("NavMesh network confirmed! Now select your STARTING point (S).", "success");
                  }}
                >
                  Finish Network ({customEdges.length} lines)
                </button>
              )}

              {/* Return to Edit Mode */}
              {factoryPhase === "ready" && (
                <button
                  type="button"
                  style={{ background: "rgba(250, 204, 21, 0.15)", color: "#facc15", border: "1px solid #facc15", padding: "4px 8px", borderRadius: "4px", fontWeight: "bold", fontSize: "11px", cursor: "pointer" }}
                  onClick={() => {
                    setFactoryPhase("draw_dots");
                    addAlert("Edit Mode: Click map to place more yellow dots.", "info");
                  }}
                >
                  ✏️ Edit Dots
                </button>
              )}

              {/* Clear Dots button */}
              {customNodes.length > 0 && (
                <button
                  type="button"
                  style={{ background: "rgba(239, 68, 68, 0.15)", color: "#ef4444", border: "1px solid #ef4444", padding: "4px 8px", borderRadius: "4px", fontWeight: "bold", fontSize: "11px", cursor: "pointer" }}
                  onClick={() => {
                    setCustomNodes([]);
                    setCustomEdges([]);
                    setSelectedDotForEdge(null);
                    setFactoryPhase("draw_dots");
                    setRoute([]);
                    setSource(null);
                    setDestination(null);
                    addAlert("Yellow dots and connections cleared. Click map to place new dots.", "info");
                  }}
                >
                  🗑️ Clear Dots
                </button>
              )}
            </>
          )}
        </div>

        {/* Live Security Phase Chip */}
        <div className={`hud-phase-chip ${securityState.toLowerCase().replace(/\s+/g, "-")}`}>
          <span className="phase-dot"></span>
          <strong>{securityState}</strong>
        </div>

        {/* Real-Time Flight Telemetry Chips */}
        <div className="hud-telemetry-cluster">
          <div className="telemetry-chip">
            <span className="chip-label">SPEED</span>
            <strong className="chip-val">{missionRunning && !missionPaused ? (44 * simulationSpeed).toFixed(0) : "0"} km/h</strong>
          </div>
          <div className="telemetry-chip">
            <span className="chip-label">ALTITUDE</span>
            <strong className="chip-val">{vehiclePosition ? "45.0 m" : "--"}</strong>
          </div>
          <div className="telemetry-chip">
            <span className="chip-label">HEADING</span>
            <strong className="chip-val">{Math.round(vehicleHeading)}°</strong>
          </div>
          <div className={`telemetry-chip ${gpsDivergence > 40 ? "danger" : ""}`}>
            <span className="chip-label">GPS DIVERGENCE</span>
            <strong className="chip-val">{gpsDivergence < 1 ? "0 m" : `${Math.round(gpsDivergence)} m`}</strong>
          </div>
        </div>

        {/* Top Action Controls */}
        <div className="hud-top-actions">
          <button
            className={`hud-icon-btn ${swarmMode ? "active swarm-btn-active" : ""}`}
            onClick={() => setSwarmMode(!swarmMode)}
            title="Toggle Decentralized IoT Swarm Mesh (Drones + Cobots Collaborative Perception)"
          >
            <RadioTower size={18} />
            <span>{swarmMode ? "🌐 SWARM MESH [ON]" : "🌐 SWARM MESH"}</span>
          </button>

          <button
            className={`hud-icon-btn ${followVehicle ? "active" : ""}`}
            onClick={() => setFollowVehicle(!followVehicle)}
            title="Lock Camera on Drone vs Free Pan"
          >
            {followVehicle ? <Eye size={18} /> : <EyeOff size={18} />}
            <span>{followVehicle ? "LOCK CAM" : "FREE PAN"}</span>
          </button>

          <button
            className="hud-icon-btn reset"
            onClick={resetSimulation}
            title="Reset Simulation"
          >
            <RotateCcw size={18} />
          </button>
        </div>
      </header>

      {/* ── Swarm Byzantine Consensus Threat Banner ─────────────────────── */}
      {swarmMode && swarmConsensusAlert && (
        <div className="hud-swarm-consensus-banner">
          <AlertTriangle size={20} />
          <div className="banner-text">
            <strong>BYZANTINE SWARM CONSENSUS TRIGGERED</strong>
            <span>{swarmConsensusAlert}</span>
          </div>
        </div>
      )}

      {/* ── 3. Floating Left Mission & Attack Dock ──────────────────────── */}
      <aside className={`hud-floating-dock left ${leftDockOpen ? "open" : "collapsed"}`}>
        <div className="dock-header">
          <div className="dock-title">
            <Compass size={18} />
            <span>MISSION CONTROLS</span>
          </div>
          <button
            className="dock-toggle-btn"
            onClick={() => setLeftDockOpen(!leftDockOpen)}
          >
            {leftDockOpen ? "◀" : "▶"}
          </button>
        </div>

        {leftDockOpen && (
          <div className="dock-content">
            {/* Custom On-Map Waypoint Pickers (Unprecoded!) */}
            <div className="dock-section">
              <label className="section-caption">CHOOSE INITIAL & FINAL DESTINATION</label>

              <div className="waypoint-pick-card">
                <div className={`pick-row ${selectionMode === "start" ? "active" : ""}`}>
                  <div className="pick-indicator start">S</div>
                  <div className="pick-meta">
                    <span className="pick-label">INITIAL DEPARTURE (START)</span>
                    <span className="pick-coords">
                      {source ? `${source[0].toFixed(4)}, ${source[1].toFixed(4)}` : "Click map to choose"}
                    </span>
                  </div>
                  <button
                    className={`pick-action-btn ${selectionMode === "start" ? "active" : ""}`}
                    onClick={() => setSelectionMode("start")}
                    disabled={missionRunning}
                  >
                    {selectionMode === "start" ? "CLICK MAP..." : "SET"}
                  </button>
                </div>

                <div className={`pick-row ${selectionMode === "dest" ? "active" : ""}`}>
                  <div className="pick-indicator dest">D</div>
                  <div className="pick-meta">
                    <span className="pick-label">
                      FINAL DESTINATION (TARGET)
                      {overrideToStart && (
                        <span style={{ color: "#fb7185", marginLeft: "6px", fontWeight: "bold" }}>
                          [OVERRIDDEN TO START]
                        </span>
                      )}
                    </span>
                    <span className="pick-coords" style={overrideToStart ? { color: "#fb7185", fontWeight: "bold" } : {}}>
                      {destination ? `${destination[0].toFixed(4)}, ${destination[1].toFixed(4)}` : "Click map to choose"}
                      {overrideToStart && " (LOCKED TO START POINT)"}
                    </span>
                  </div>
                  <button
                    className={`pick-action-btn ${selectionMode === "dest" ? "active" : ""}`}
                    onClick={() => setSelectionMode("dest")}
                    disabled={missionRunning || overrideToStart}
                  >
                    {overrideToStart ? "LOCKED" : selectionMode === "dest" ? "CLICK MAP..." : "SET"}
                  </button>
                </div>
              </div>

              <div className="waypoint-tool-row">
                <button
                  className="hud-btn danger-subtle"
                  onClick={clearAndPickNewRoute}
                  disabled={missionRunning}
                  title="Clear waypoints and choose from scratch"
                >
                  <Trash2 size={14} />
                  <span>CLEAR / NEW ROUTE</span>
                </button>

                {/* Optional Quick Preset selector */}
                <select
                  className="hud-dropdown mini"
                  onChange={(e) => handleQuickPreset(e.target.value)}
                  disabled={missionRunning}
                  defaultValue=""
                >
                  <option value="" disabled>Load preset corridor...</option>
                  {MISSION_PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Flight Execution */}
            <div className="dock-section">
              <label className="section-caption">FLIGHT EXECUTION</label>
              <button
                className={`flight-launch-btn ${missionRunning ? (missionPaused ? "paused" : "running") : "ready"}`}
                onClick={toggleMission}
                disabled={!route.length}
              >
                {missionRunning ? (
                  missionPaused ? <Play size={18} /> : <Pause size={18} />
                ) : (
                  <Play size={18} />
                )}
                <span>
                  {missionRunning
                    ? (missionPaused ? "RESUME FLIGHT" : "PAUSE FLIGHT")
                    : "LAUNCH MISSION"}
                </span>
              </button>

              {/* Simulation Speed */}
              <div className="speed-selector-row">
                <span className="speed-label">SPEED:</span>
                {[1, 2, 4].map((spd) => (
                  <button
                    key={spd}
                    className={`speed-pill ${simulationSpeed === spd ? "active" : ""}`}
                    onClick={() => setSimulationSpeed(spd)}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Cyber Threat & Spoof Suite (Inject at any time!) */}
            <div className="dock-section attack-section">
              <div className="attack-header">
                <RadioTower size={16} />
                <label className="section-caption danger">GPS SPOOFING (CLICK MAP ANYTIME)</label>
              </div>

              <p className="attack-hint">
                ⚡ <strong>Click anywhere directly on the map</strong> at any time of launch or flight to pull the drone toward that location!
              </p>

              <div className="spoof-target-status">
                <span>ACTIVE SPOOF TARGET:</span>
                <strong>
                  {spoofTarget
                    ? `${spoofTarget[0].toFixed(4)}, ${spoofTarget[1].toFixed(4)}`
                    : "None (Click map to set)"}
                </strong>
              </div>

              {spoofing && (
                <button className="attack-stop-btn" onClick={stopAttack}>
                  STOP ATTACK
                </button>
              )}
            </div>
          </div>
        )}
      </aside>

      {/* ── 4. Floating Right Defense Matrix Dock ───────────────────────── */}
      <aside className={`hud-floating-dock right ${rightDockOpen ? "open" : "collapsed"}`}>
        <div className="dock-header">
          <button
            className="dock-toggle-btn"
            onClick={() => setRightDockOpen(!rightDockOpen)}
          >
            {rightDockOpen ? "▶" : "◀"}
          </button>
          <div className="dock-title">
            <Activity size={18} />
            <span>SECURITY MATRIX</span>
          </div>
        </div>

        {rightDockOpen && (
          <div className="dock-content">
            {/* IoT Swarm P2P Fleet Mesh Section */}
            {swarmMode && (
              <div className="dock-section swarm-fleet-section">
                <div className="swarm-header-row">
                  <label className="section-caption">
                    <span className="live-dot-cyan"></span> IOT SWARM P2P MESH ({swarmRobots.length} NODES)
                  </label>
                  <span className={`swarm-patrol-status-pill ${swarmPatrolActive ? "active" : "paused"}`}>
                    {swarmPatrolActive ? "AUTONOMOUS PATROL: ON" : "PATROL: PAUSED"}
                  </span>
                </div>

                <div className="swarm-topology-bar">
                  <span>TOPOLOGY: <strong>P2P 802.15.4 MESH</strong></span>
                  <span>LINKS: <strong style={{ color: "#00ff9d" }}>{meshLinks.length}/3 ACTIVE</strong></span>
                </div>

                {/* Swarm Quick Interactive Action Controls */}
                <div className="swarm-action-buttons-row">
                  <button
                    className={`swarm-act-btn ${swarmPatrolActive ? "active" : ""}`}
                    onClick={() => setSwarmPatrolActive(!swarmPatrolActive)}
                    title="Toggle autonomous movement for all robots"
                  >
                    {swarmPatrolActive ? <Pause size={12} /> : <Play size={12} />}
                    <span>{swarmPatrolActive ? "PAUSE PATROL" : "START PATROL"}</span>
                  </button>

                  <button
                    className="swarm-act-btn hazard-btn"
                    onClick={triggerSimulatedGasSpike}
                    title="Center methane gas plume onto AGV-OMEGA to immediately test sensor spike & broadcast"
                  >
                    <Flame size={12} />
                    <span>SIMULATE GAS SPIKE</span>
                  </button>

                  <button
                    className="swarm-act-btn spoof-btn"
                    onClick={triggerSwarmSpoofAttack}
                    title="Spoof UAV-ALPHA to trigger peer Byzantine cross-check"
                  >
                    <Zap size={12} />
                    <span>SPOOF UAV-ALPHA</span>
                  </button>
                </div>

                {/* Swarm Dynamic Robots Cards */}
                <div className="swarm-nodes-container">
                  {swarmRobots.map((robot) => (
                    <div key={robot.id} className={`swarm-node-item ${robot.isCompromised ? "compromised" : robot.sensors.gas > 40 ? "hazard-alert" : ""}`}>
                      <div className="node-item-header">
                        <div className="node-title">
                          <span className={`node-badge ${robot.role}`}>{robot.role === "cobot" ? "🤖 COBOT" : "🚁 UAV"}</span>
                          <strong>{robot.name}</strong>
                        </div>
                        <span className={`node-status-tag ${robot.isCompromised ? "danger" : robot.sensors.gas > 40 ? "hazard" : "ok"}`}>
                          {robot.isCompromised ? "SPOOF QUARANTINED" : robot.sensors.gas > 40 ? "GAS ALERT" : "ACTIVE"}
                        </span>
                      </div>
                      <div className="node-mini-metrics">
                        <div>BATTERY: <strong>{robot.battery}%</strong></div>
                        <div>GPS TRUST: <strong style={{ color: robot.sensors.gpsTrust < 0.5 ? '#f43f5e' : '#00ff9d' }}>{(robot.sensors.gpsTrust * 100).toFixed(0)}%</strong></div>
                        {robot.sensors.gas !== undefined && (
                          <div>
                            GAS: <strong style={{ color: robot.sensors.gas > 40 ? '#f43f5e' : '#f59e0b', fontWeight: 'bold' }}>
                              {robot.sensors.gas} ppm {robot.sensors.gas > 40 ? "⚠️" : ""}
                            </strong>
                          </div>
                        )}
                        {robot.sensors.obstacle !== undefined && <div>OBSTACLE: <strong>{robot.sensors.obstacle} m</strong></div>}
                      </div>
                      <div className="node-mission-status">
                        ROLE: <span>{robot.status}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Live IoT P2P Serialized Packet Stream Terminal */}
                <div className="iot-packet-terminal">
                  <div className="terminal-header">
                    <div className="terminal-title">
                      <Terminal size={12} />
                      <span>LIVE IOT P2P PACKET STREAM (802.15.4)</span>
                    </div>
                    <span className="terminal-stats">TX: 3.8 kB/s | LOSS: 0%</span>
                  </div>
                  <div className="terminal-body" id="iot-packet-stream">
                    {iotPacketLog.length === 0 ? (
                      <div className="term-line info">Waiting for P2P mesh handshake packets...</div>
                    ) : (
                      iotPacketLog.map((pkt) => (
                        <div key={pkt.id} className={`term-packet-line ${pkt.type.toLowerCase()}`}>
                          <span className="pkt-time">[{pkt.time}]</span>
                          <span className="pkt-seq">#{pkt.seq}</span>
                          <span className="pkt-route">{pkt.sender} ➔ {pkt.target}</span>
                          <span className="pkt-rssi">{pkt.rssi}dBm</span>
                          <span className="pkt-payload">{pkt.payload}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Shared Swarm Hazard Registry */}
                {sharedHazards.length > 0 && (
                  <div className="swarm-hazards-registry">
                    <div className="registry-title">
                      <AlertTriangle size={13} color="#f43f5e" />
                      <span>SHARED HAZARD LEDGER ({sharedHazards.length})</span>
                    </div>
                    {sharedHazards.map((hz, idx) => (
                      <div key={`hz-reg-${idx}`} className="hazard-ledger-item">
                        <div className="hz-reg-head">
                          <strong>{hz.hazard_type}</strong>
                          <span>By: {hz.reported_by}</span>
                        </div>
                        <div className="hz-reg-details">{hz.details}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Risk & Threat Gauge */}
            <div className="threat-gauge-card">
              <div className="gauge-header">
                <span>SYSTEM RISK LEVEL</span>
                <span className={`status-badge ${riskStatus.toLowerCase()}`}>{riskStatus}</span>
              </div>
              <div className="gauge-meter-wrapper">
                <div className="gauge-ring">
                  <span className="gauge-score">{riskScore}</span>
                  <span className="gauge-total">/100</span>
                </div>
                <div className="gauge-details">
                  <div className="gauge-stat">
                    <span>GPS TRUST:</span>
                    <strong style={{ color: gpsTrust > 50 ? "#00ff9d" : "#f43f5e" }}>{gpsTrust}%</strong>
                  </div>
                  <div className="gauge-stat">
                    <span>GATE THRESHOLD:</span>
                    <strong>{DETECTION_THRESHOLD} m</strong>
                  </div>
                  <div className="gauge-stat">
                    <span>ATTACKS SURVIVED:</span>
                    <strong style={{ color: "#22d3ee" }}>{spoofCount}</strong>
                  </div>
                </div>
              </div>
              <div className="risk-progress-bar">
                <div
                  className={`risk-fill ${riskStatus.toLowerCase()}`}
                  style={{ width: `${Math.min(riskScore, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Sensor Trust Matrix */}
            <div className="dock-section">
              <label className="section-caption">MULTI-SENSOR ARBITRATION</label>
              <SensorCards gpsTrust={gpsTrust} quarantined={quarantined} />
            </div>

            {/* Autonomous Recovery Stepper */}
            <div className="dock-section">
              <label className="section-caption">AUTONOMOUS RECOVERY PIPELINE</label>
              <div className="recovery-stepper">
                <div className={`step-node ${gpsDivergence >= DETECTION_THRESHOLD ? "done" : ""}`}>
                  <span className="step-num">1</span>
                  <span className="step-text">Chi-Square & CUSUM Anomaly Detection</span>
                </div>
                <div className={`step-node ${quarantined ? "done" : ""}`}>
                  <span className="step-num">2</span>
                  <span className="step-text">SAARM Quarantine (GNSS Cut & Hover)</span>
                </div>
                <div className={`step-node ${recoveryActive || securityState === "REJOINED // NOMINAL" ? "done" : ""}`}>
                  <span className="step-num">3</span>
                  <span className="step-text">HMAC Server Corridor Calculation</span>
                </div>
                <div className={`step-node ${securityState === "REJOINED // NOMINAL" || securityState === "MISSION COMPLETE" ? "done" : ""}`}>
                  <span className="step-num">4</span>
                  <span className="step-text">Corridor Rejoin & Mission Completion</span>
                </div>
              </div>

              {recoveryActive && (
                <div className="recovery-progress-box">
                  <div className="progress-label">
                    <span>REJOIN PROGRESS</span>
                    <strong>{Math.round(recoveryProgress * 100)}%</strong>
                  </div>
                  <div className="recovery-bar">
                    <div className="recovery-fill" style={{ width: `${recoveryProgress * 100}%` }}></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </aside>

      {/* ── 5. Floating Bottom Incident Ticker & Legend ─────────────────── */}
      <footer className={`hud-bottom-bar ${logExpanded ? "expanded" : ""}`}>
        <div className="bottom-bar-header">
          <div className="ticker-lead" onClick={() => setLogExpanded(!logExpanded)}>
            <Activity size={16} />
            <span className="ticker-title">LIVE EVENT AUDIT</span>
            <span className="ticker-latest">
              {alerts.length > 0 ? alerts[0].message : "System nominal. Ready for flight dispatch."}
            </span>
          </div>

          <div className="bottom-controls">
            <div className="map-legend-pills">
              <span className="legend-item"><span className="legend-dot drone"></span> Trusted UAV</span>
              <span className="legend-item"><span className="legend-dot gps"></span> Raw GPS</span>
              <span className="legend-item"><span className="legend-dot route"></span> Authorized Route</span>
              <span className="legend-item"><span className="legend-dot recovery"></span> Rejoin Path</span>
              <span className="legend-item"><span className="legend-dot beacon"></span> Rogue Beacon</span>
              {swarmMode && (
                <>
                  <span className="legend-item"><span className="legend-dot swarm-link"></span> IoT Mesh Link</span>
                  <span className="legend-item"><span className="legend-dot cobot"></span> Ground Cobot</span>
                </>
              )}
            </div>

            <button className="expand-log-btn" onClick={() => setLogExpanded(!logExpanded)}>
              {logExpanded ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
            </button>
          </div>
        </div>

        {logExpanded && (
          <div className="audit-log-stream">
            {alerts.length === 0 ? (
              <div className="log-empty">No security incidents logged yet.</div>
            ) : (
              alerts.map((item) => (
                <div key={item.id} className={`log-entry ${item.type}`}>
                  <span className="log-time">{item.time}</span>
                  <span className="log-msg">{item.message}</span>
                </div>
              ))
            )}
          </div>
        )}
      </footer>
    </div>
  );
}

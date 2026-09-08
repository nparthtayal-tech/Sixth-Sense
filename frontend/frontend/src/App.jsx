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
  Crosshair
} from "lucide-react";

import "./App.css";
import SensorMap from "./Components/SensorMap";
import SensorCards from "./Components/SensorCards";
import { getRoute, getRecoveryRoute, generateAirCorridor } from "./services/routing";
import { MISSION_PRESETS, calculateBearing } from "./services/presets";

const DETECTION_THRESHOLD = 95; // meters (Chi-Square / CUSUM gate)

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

  // Telemetry & UI Drawers
  const [routeLoading, setRouteLoading] = useState(false);
  const [alerts, setAlerts] = useState([]);
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

  // Audit Log Helper
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

  // Compute Route between source and destination
  const computeRoute = useCallback(async (src, dst) => {
    if (!src || !dst) return;
    setRouteLoading(true);
    try {
      addAlert("Calculating flight path for custom waypoints...", "info");
      const result = await getRoute(src, dst);
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
      addAlert(`Flight path ready: ${(result.distance / 1000).toFixed(2)} km. Ready to Launch!`, "success");
    } catch (err) {
      console.error(err);
      addAlert("Could not compute route. Try selecting another point.", "danger");
    } finally {
      setRouteLoading(false);
    }
  }, [addAlert]);

  // Initial welcome message
  useEffect(() => {
    addAlert("Ready: Click anywhere on the map to place your INITIAL START point.", "info");
  }, [addAlert]);

  // Master Map Click Handler: Supports setting start, destination, and spoofing anytime!
  const handleMapClick = (pos) => {
    // If high-frequency override is active, strictly reject new clicks/spoofs
    if (overrideToStartRef.current) {
      addAlert("🛑 DESTINATION LOCKED TO START: High-frequency cyber attack confirmed (>5 spoofing in 5s). Drone goes ONLY towards starting point!", "danger");
      return;
    }

    // 1. If currently choosing Start:
    if (selectionMode === "start" || !source) {
      setSource(pos);
      setVehiclePosition(pos);
      vehiclePosRef.current = pos;
      setGpsPosition(pos);
      gpsPosRef.current = pos;
      setRoute([]);
      setRecoveryRoute([]);
      setBreadcrumbTrail([pos]);
      setSelectionMode("dest");
      setSecurityState("SELECT DESTINATION");
      addAlert(`Initial Start point set: [${pos[0].toFixed(4)}, ${pos[1].toFixed(4)}]. Now click for Final Destination.`, "success");
      return;
    }

    // 2. If currently choosing Destination:
    if (selectionMode === "dest" || (!destination && source)) {
      setDestination(pos);
      setSelectionMode(null);
      addAlert(`Final Destination set: [${pos[0].toFixed(4)}, ${pos[1].toFixed(4)}].`, "success");
      computeRoute(source, pos);
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
  const injectOrUpdateSpoofTarget = (targetPos) => {
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

    const returnCorridor = generateAirCorridor(currentPos, startPt, 50);
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

      const recoveryResult = await getRecoveryRoute(currentPos, rejoinPoint);
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
        // Just like demo_animated_sentry: autopilot moves toward spoof target!
        const target = spoofTargetRef.current;
        const deceivedStep = moveTowards(curPos, target, 0.00040 * simulationSpeed);

        vehiclePosRef.current = deceivedStep;
        setVehiclePosition(deceivedStep);
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
      routeIndexRef.current = nextIndex;
      const nextPos = route[nextIndex];

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
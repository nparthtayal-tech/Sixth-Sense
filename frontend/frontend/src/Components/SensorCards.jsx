import {
  Satellite,
  Activity,
  RotateCw,
  ScanLine,
  Thermometer,
  Gauge
} from "lucide-react";

const sensors = [
  ["GPS", Satellite],
  ["Accelerometer", Activity],
  ["Gyroscope", RotateCw],
  ["LiDAR", ScanLine],
  ["Temperature", Thermometer],
  ["Pressure", Gauge]
];

export default function SensorCards({
  gpsTrust,
  quarantined
}) {
  const values = {
    GPS: quarantined ? 0 : gpsTrust,

    Accelerometer: 100,

    Gyroscope: 100,

    LiDAR: 100,

    Temperature: 100,

    Pressure: 100
  };

  return (
    <div className="sensor-grid">

      {sensors.map(([name, Icon]) => {

        const trust = values[name];

        return (
          <div
            className={`sensor-card ${name === "GPS" && quarantined
                ? "sensor-danger"
                : ""
              }`}
            key={name}
          >

            <div className="sensor-card-top">

              <Icon size={20} />

              <span>
                {name}
              </span>

            </div>

            <div className="trust-value">
              {Math.round(trust)}%
            </div>

            <div className="trust-bar">

              <div
                style={{
                  width: `${trust}%`
                }}
              />

            </div>

            <small>

              {name === "GPS" && quarantined
                ? "QUARANTINED"
                : trust >= 80
                  ? "TRUSTED"
                  : "DEGRADED"}

            </small>

          </div>
        );
      })}

    </div>
  );
}
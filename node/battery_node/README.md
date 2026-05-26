# battery_node

Owns the canonical battery state for a node and exposes it on a small HTTP
service. Backend is pluggable: the same module powers scout (where battery
voltage rides on the rover's UART) and any node with a UPS HAT (INA219 over
I2C).

## Layout

```
battery_node/
  __init__.py
  commands.py         # NL handler + selftest hook used by luhkas_node
  service.py          # HTTP server (run as systemd unit)
  backends/
    base.py           # BatteryBackend / BatteryReading
    uart_proxy.py     # Reads /run/luhkas/battery_raw.json (scout)
    ina219.py         # Reads INA219 over I2C (UPS HAT)
```

## Service

Default port `5003`. Endpoints:

- `GET /health` — `{ok, backend, battery: {...}}`
- `GET /battery` — `{ok, stale, voltage, percent, source, timestamp, ...}`

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `BATTERY_BACKEND` | `uart_proxy` | `uart_proxy`, `max17040` (Geekworm X120x), or `ina219` |
| `BATTERY_HOST` | `0.0.0.0` | bind host |
| `BATTERY_PORT` | `5003` | bind port |
| `BATTERY_POLL_S` | `1.0` | seconds between backend reads |
| `BATTERY_STALE_S` | `5.0` | reading is `stale` after this many seconds without a fresh one |

UART proxy backend:

| Env var | Default | Meaning |
|---|---|---|
| `BATTERY_UART_PROXY_PATH` | `${XDG_RUNTIME_DIR}/luhkas-battery.json` (per-user tmpfs) | file written by `robot_api`'s serial reader |
| `BATTERY_UART_PROXY_MAX_AGE` | `10` | seconds before a file reading is considered stale |

MAX17040 backend (Geekworm X1200/X1201/X1202):

| Env var | Default | Meaning |
|---|---|---|
| `BATTERY_I2C_BUS` | `1` | I2C bus id |
| `BATTERY_MAX17040_ADDR` | `0x36` | MAX17040 default address |

The MAX17040 fuel gauge reports voltage and state-of-charge (SOC) directly,
so no `V_EMPTY`/`V_FULL` calibration is needed.

INA219 backend (generic):

| Env var | Default | Meaning |
|---|---|---|
| `BATTERY_I2C_BUS` | `1` | I2C bus id |
| `BATTERY_INA219_ADDR` | `0x42` | I2C address (Waveshare Pi 5 UPS default; common alternates: `0x40`, `0x41`, `0x43`) |
| `BATTERY_V_EMPTY` | `6.0` | voltage considered 0% (2S Li-Ion default) |
| `BATTERY_V_FULL` | `8.4` | voltage considered 100% |

## Scout integration

Scout's battery telemetry arrives on the same UART that carries motor + IMU
data, owned by `services/robot_api.py`. To avoid splitting the serial port,
the serial reader writes each `T:1001` packet's voltage/percent to
`${XDG_RUNTIME_DIR}/luhkas-battery.json` (typically `/run/user/<uid>/luhkas-battery.json`
— per-user tmpfs, no root needed). The `uart_proxy` backend tails that file.

The `oled_updater` in `robot_api` fetches the latest reading from
`http://127.0.0.1:5003/battery` instead of holding the value in-process.

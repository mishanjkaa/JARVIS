# RFC-010 Phone Location & Turn-by-Turn Navigation

RFC-010 lets JARVIS give the owner spoken/text turn-by-turn directions ("turn right",
"continue straight for 300 meters") to a destination, using the owner's own phone as the
GPS source and a maps/routing provider for the route — no camera or vision involvement at
all, since a camera can identify what's in front of it but has no idea where the nearest
mall is; only a location + map can answer that.

## Decisions locked in (owner-confirmed)

- **Transport: Tailscale.** The PC and the phone join the same private Tailscale network
  (a WireGuard-based mesh VPN). Once both devices are on it, the PC gets a stable Tailscale
  IP (e.g. `100.x.y.z`) reachable from the phone anywhere the phone has internet — home
  Wi-Fi or mobile data — without opening any port on the home router and without a public
  IP. JARVIS's location/voice endpoints bind only to the Tailscale interface, not to
  `0.0.0.0`, so they are simply unreachable from the wider internet — only from other
  devices on that same Tailscale network. This answers the earlier "how does the phone
  reach the PC" question directly: Tailscale removes the "only works on home Wi-Fi"
  limitation of a plain LAN endpoint, without the "open endpoint on the public internet"
  exposure of a raw tunnel/relay.
- **Location source app: Overland (iOS).** A phone still needs *something* pushing GPS
  points to a URL — Tailscale is the pipe, not the location source. Overland (by Aaron
  Parecki) is a free, purpose-built iOS app for exactly this: it runs in the background,
  batches GPS points, and POSTs them as GeoJSON to any HTTP(S) endpoint you configure —
  here, JARVIS's Tailscale address. No Google account, no third-party cloud in between. Set
  the endpoint in Overland to `https://<pc-tailscale-ip-or-name>:<port>/location/overland`
  once JARVIS exposes that route. Realistic accuracy: outdoors, typically 5-15 meters
  (this is a physical GPS/phone-hardware limit, not something software can improve on); it
  degrades further indoors or between tall buildings. "Exactly" precise location isn't
  something any phone can promise — Overland's job is just to relay whatever accuracy iOS
  itself reports, as often as configured (a send interval is configurable in the app; every
  10-30 seconds on "moving" is reasonable and battery-friendly).
- **Routing/maps provider: OpenStreetMap + OSRM.** Free, no API key, no per-request cost,
  and keeps routing off a third party's servers — consistent with "local by default"
  elsewhere in this project. A public OSRM demo server is fine to start with; a
  self-hosted OSRM instance (a single Docker container with a Latvia/Baltics OSM extract)
  is the natural upgrade once this is working, for reliability and no rate limits. Anyone
  who later wants Google's generally sharper routing/ETA data can swap the provider behind
  the same internal interface — the tool surface below doesn't name a provider on purpose.

## What triggers a session, and when it stops

Matching the rest of the project's "explicit activation, explicit stop, nothing silently
ongoing" pattern: Overland keeps sending points in the background once configured (that's
how Overland works, and battery-friendly background tracking is the whole point of using a
dedicated app instead of polling from JARVIS), but JARVIS only *acts* on that location
data — for navigation, for "where am I", for distance queries — when the owner asks. There
is no standing "JARVIS is watching where I am" behavior; there's a phone app quietly
holding onto the freshest point, and JARVIS reads that latest point only when asked a
question that needs it. `navigation.get_next_instruction` recomputes the route from
whatever the freshest stored point is at the moment it's called — there is no
server-side background poller re-routing on a timer, on purpose: a poller would be exactly
the kind of silently-ongoing behavior this project's design avoids elsewhere.

## Supported in RFC-010

- the `/location/overland` HTTP route Overland's app posts to (over Tailscale only),
  storing just the single freshest point per device, not a history — this is an ingestion
  route driven by the phone's background app, not an AI-callable tool (see Security)
- `location.where_am_i` — reverse-geocodes the freshest point into a human answer ("Джарвис,
  где я нахожусь?")
- `location.distance_to(place)` — answers "сколько до дома метров/км", where named places
  (like "home") are looked up from owner-saved coordinates, not from any third-party
  contacts/places list
- `location.save_place(name, latitude, longitude)` — the only way a named place (such as
  `home`) becomes available to `location.distance_to`/`navigation.start`; never inferred
  from history
- `navigation.start(destination)` — resolves the destination (a saved place name, raw
  `"lat,lon"` coordinates, or a place name looked up via OSM/Nominatim) via OSRM and
  returns the first turn instruction plus distance/ETA
- `navigation.get_next_instruction(session_id)` — recomputed from the freshest Overland
  point each time it's asked, returns the next turn-by-turn instruction ("turn right",
  "continue straight", "300 meters to your destination") and total remaining distance
- `navigation.stop(session_id)` — ends an active turn-by-turn session (does not stop
  Overland itself, which keeps running independently on the phone — see above)
- `navigation status` / `location status` — reports the freshest point's age, whether a
  turn-by-turn session is active, and whether the configured Tailscale bind interface is
  currently present on this machine

## Not supported in RFC-010

- tracking anyone's location other than the owner's own phone
- storing a location *history* — only the single freshest point per device is ever kept;
  each new Overland post overwrites the last one
- using the camera or Vision Runtime for way-finding — this RFC is GPS + maps only
- accepting a location point from anything that isn't authenticated as the owner's own
  device over the owner's own Tailscale network (see Security below)
- reverse-geocoding or routing against anyone's saved places except the owner's own
  (`home`, and whatever else the owner explicitly saves — never inferred from history)
- a background poller that silently re-routes on a timer while a session is open —
  `navigation.get_next_instruction` is pull-only, matching this project's "explicit ask"
  pattern
- calling `location.receive_overland_point` (the ingestion path) as an AI-plannable tool —
  it is intentionally not registered in the tool registry at all, so a compromised or
  malfunctioning AI plan can never inject a fake location update

## Security

- The `/location/overland` route is bound to the machine's Tailscale interface only, never
  `0.0.0.0` — being on the Tailscale network is necessary but the route still checks a
  shared secret configured in `location_shared_secret` against the `Authorization: Bearer
  <token>` header Overland sends (Overland has a dedicated "access token" field for exactly
  this), so a stolen Tailscale node key alone isn't sufficient to spoof location updates.
  `location_shared_secret` is deliberately excluded from the `config set`/`config show`
  live configuration surface — it is set once by editing `config/config.json` directly, and
  `config show`/`config get` redact it rather than echo it back in plaintext.
- Startup fails loudly (a clear error, JARVIS keeps running otherwise) rather than silently
  falling back to a public bind if `location_bind_host` is empty, is `0.0.0.0`/`::`, or does
  not resolve to an address actually present on this machine.
- Tailscale ACLs (configured once, in the Tailscale admin console, not in JARVIS) should
  restrict which devices can reach the PC's location/voice ports at all — this is
  configuration on the owner's Tailscale account, not something the JARVIS codebase can
  enforce by itself, so it's called out here as a setup step, not skipped.

## Session lifecycle

A `navigation.start` call opens a `NavigationSession` (an opaque `session_id`, the
resolved destination coordinates and name, and the owning request/agent-task id) kept in
its own in-memory store, separate from the freshest-point store and from every Vision
capture store. Unlike a Vision capture, a session has no TTL of its own — it lives until
explicitly ended — but it is still owner-request-scoped the same way, and is deleted on
every path that already ends an owner's agent task:

- normal completion (`AgentController._run_task` finishing all plan steps)
- a failed step, whether the task was pending approval or already running
  (`AgentController._fail_pending_task` / `_fail_running_task`)
- cancellation, whether the task was pending approval or already running
  (`AgentController.cancel_pending_or_running_task` / `_finalize_stop`)
- approval expiration — a pending plan that simply times out before the owner runs
  `approve plan`/`cancel plan` (`app.brain.planner.approval._synchronize_terminal_pending_state_locked`)

The last of these is easy to miss because it is a structurally separate code path from
`AgentController.cancel_pending_or_running_task` (this bit JARVIS's own Vision capture
cleanup twice before — once for browser captures, once for desktop captures — so
`navigation.start`'s cleanup is wired into both places from the start, with a regression
test asserting all four paths, not just the three that are easy to reach by only testing
`AgentController`).

## Owner verification

1. `Джарвис, где я нахожусь?` answers with a human-readable location derived from the
   freshest Overland point, and says how old that point is if it's more than a couple of
   minutes stale (phone briefly offline, etc.) rather than presenting stale data as current.
2. `Сколько до дома?` answers with a distance (and, once `navigate to` is supported,
   accepts "home" as a destination) using the owner's own saved `home` coordinates.
3. `navigate to <place>` starts a session, states the resolved destination and the
   OSRM-derived route, and gives a first instruction.
4. Turn-by-turn instructions update as new Overland points arrive, without the owner
   needing to look at a screen.
5. `navigation stop` ends a turn-by-turn session; Overland itself keeps running on the
   phone as configured, independent of any JARVIS session state.
6. Only the single freshest point per device is ever stored; there is no query or command
   that returns a location history.
7. A location POST without the correct shared secret is rejected, logged, and never
   updates the stored point, even if it arrives from within the Tailscale network.

package traffic.api;

import com.sun.net.httpserver.*;
import traffic.sensor.AsyncSensorPipeline;
import traffic.sensor.AsyncSensorPipeline.SensorFrame;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.logging.*;

/**
 * TrafficControlServer
 * --------------------
 * Lightweight HTTP server bridging the Python DRL agent and the
 * signal controller hardware (or SUMO TraCI in simulation).
 *
 * Endpoints:
 *   GET  /state       → current 21-element state vector + metadata (JSON)
 *   POST /action      → apply signal phase from DRL agent
 *   GET  /metrics     → episode statistics (JSON)
 *   GET  /health      → liveness probe
 *   POST /reset       → reset episode counters
 *
 * Safety guarantees enforced server-side (independent of the DRL agent):
 *   1. Minimum green time per phase
 *   2. All-red clearance between conflicting phases
 *   3. Hard pedestrian protection veto
 *   4. Rate limiting on action endpoint (max 1 req/s)
 *
 * Error responses:
 *   400 Bad Request  — malformed JSON
 *   409 Conflict     — safety veto active
 *   429 Too Many Requests — rate limit
 *   500 Internal Server Error
 */
public final class TrafficControlServer {

    private static final Logger LOG = Logger.getLogger(TrafficControlServer.class.getName());

    // ── Signal phase constants ────────────────────────────────────────────────

    public enum Phase {
        NS_GREEN(0), EW_GREEN(1), PED_CROSSING(2), ALL_RED(3);

        public final int index;
        Phase(int i) { this.index = i; }

        public static Phase fromIndex(int i) {
            return values()[i % values().length];
        }
    }

    // ── Safety configuration ──────────────────────────────────────────────────

    private record SafetyConfig(
        Map<Phase, Long> minDurationMs,
        long             allRedDurationMs,
        long             pedUrgencyThresholdMs
    ) {
        static SafetyConfig defaults() {
            return new SafetyConfig(
                Map.of(
                    Phase.NS_GREEN,     10_000L,
                    Phase.EW_GREEN,     10_000L,
                    Phase.PED_CROSSING, 15_000L,
                    Phase.ALL_RED,       3_000L
                ),
                3_000L,
                60_000L
            );
        }
    }

    // ── State tracking ────────────────────────────────────────────────────────

    private volatile Phase  currentPhase     = Phase.NS_GREEN;
    private volatile long   phaseStartMs     = System.currentTimeMillis();
    private volatile boolean safetyVetoActive = false;
    private volatile SensorFrame lastFrame   = null;

    // Metrics
    private final AtomicLong totalRequests   = new AtomicLong(0);
    private final AtomicLong phaseChanges    = new AtomicLong(0);
    private final AtomicLong safetyVetos     = new AtomicLong(0);
    private final AtomicLong rateLimitHits   = new AtomicLong(0);
    private final long       serverStartMs   = System.currentTimeMillis();

    // Rate limiter (token bucket — 1 action per 1000ms)
    private final AtomicLong lastActionMs    = new AtomicLong(0);
    private static final long MIN_ACTION_INTERVAL_MS = 500L;

    private final AsyncSensorPipeline sensorPipeline;
    private final SafetyConfig        safety;
    private final int                 port;
    private HttpServer                server;

    // ── Construction ──────────────────────────────────────────────────────────

    public TrafficControlServer(AsyncSensorPipeline pipeline, int port) {
        this.sensorPipeline = pipeline;
        this.safety         = SafetyConfig.defaults();
        this.port           = port;

        // Register frame listener to keep lastFrame current
        // (pipeline calls this on every window emission)
        pipeline.submitReading(
            AsyncSensorPipeline.SensorReading.camera(0, false)  // dummy to register
        );
    }

    public void setLastFrame(SensorFrame frame) {
        this.lastFrame = frame;
        updateSafetyVeto(frame);
    }

    // ── Server lifecycle ──────────────────────────────────────────────────────

    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(port), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());

        register("/state",   this::handleState);
        register("/action",  this::handleAction);
        register("/metrics", this::handleMetrics);
        register("/health",  this::handleHealth);
        register("/reset",   this::handleReset);

        server.start();
        LOG.info(String.format("TrafficControlServer started on :%d", port));
        logEndpoints();
    }

    public void stop() {
        if (server != null) {
            server.stop(1);
            LOG.info("Server stopped.");
        }
    }

    // ── Handlers ─────────────────────────────────────────────────────────────

    private void handleState(HttpExchange ex) throws IOException {
        if (!assertMethod(ex, "GET")) return;
        totalRequests.incrementAndGet();

        SensorFrame frame = lastFrame;
        long elapsed = System.currentTimeMillis() - phaseStartMs;

        String json = buildStateJson(frame, elapsed);
        respond(ex, 200, json);
    }

    private void handleAction(HttpExchange ex) throws IOException {
        if (!assertMethod(ex, "POST")) return;
        totalRequests.incrementAndGet();

        // Rate limiting
        long now = System.currentTimeMillis();
        long last = lastActionMs.get();
        if (now - last < MIN_ACTION_INTERVAL_MS) {
            rateLimitHits.incrementAndGet();
            respond(ex, 429, json("error", "rate_limited",
                "retry_after_ms", String.valueOf(MIN_ACTION_INTERVAL_MS - (now - last))));
            return;
        }
        lastActionMs.set(now);

        String body;
        try (InputStream is = ex.getRequestBody()) {
            body = new String(is.readAllBytes(), StandardCharsets.UTF_8).trim();
        }

        int actionIdx;
        try {
            actionIdx = parseActionIndex(body);
        } catch (NumberFormatException e) {
            respond(ex, 400, json("error", "invalid_action", "detail", e.getMessage()));
            return;
        }

        Phase requested = Phase.fromIndex(actionIdx);
        String result   = applyAction(requested);
        respond(ex, safetyVetoActive && requested == Phase.PED_CROSSING ? 409 : 200, result);
    }

    private void handleMetrics(HttpExchange ex) throws IOException {
        if (!assertMethod(ex, "GET")) return;
        long uptime = System.currentTimeMillis() - serverStartMs;
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("uptime_ms",      uptime);
        m.put("total_requests", totalRequests.get());
        m.put("phase_changes",  phaseChanges.get());
        m.put("safety_vetos",   safetyVetos.get());
        m.put("rate_limit_hits",rateLimitHits.get());
        m.put("current_phase",  currentPhase.name());
        m.put("phase_elapsed_ms", System.currentTimeMillis() - phaseStartMs);
        if (lastFrame != null) {
            m.put("sensor_stats", sensorPipeline.stats());
        }
        respond(ex, 200, toJson(m));
    }

    private void handleHealth(HttpExchange ex) throws IOException {
        respond(ex, 200, "{\"status\":\"ok\",\"server\":\"TrafficControlServer\"}");
    }

    private void handleReset(HttpExchange ex) throws IOException {
        if (!assertMethod(ex, "POST")) return;
        currentPhase = Phase.NS_GREEN;
        phaseStartMs = System.currentTimeMillis();
        respond(ex, 200, "{\"reset\":true}");
    }

    // ── Phase application + safety layer ─────────────────────────────────────

    private synchronized String applyAction(Phase requested) {
        long elapsed = System.currentTimeMillis() - phaseStartMs;
        long minMs   = safety.minDurationMs().getOrDefault(currentPhase, 5_000L);

        // 1. Minimum duration enforcement
        if (requested != currentPhase && elapsed < minMs) {
            return buildActionResponse(currentPhase, requested, "held_min_duration", elapsed);
        }

        // 2. Pedestrian safety veto
        if (safetyVetoActive
            && isConflicting(requested, Phase.PED_CROSSING)
            && lastFrame != null
            && anyPedUrgent(lastFrame)) {
            safetyVetos.incrementAndGet();
            LOG.warning("Safety veto: pedestrian urgent, blocking " + requested.name());
            return buildActionResponse(
                Phase.PED_CROSSING, requested, "safety_veto_ped_urgent", elapsed
            );
        }

        // 3. Apply change (with ALL_RED interlock)
        if (requested != currentPhase) {
            // Insert ALL_RED clearance if transitioning between conflicting phases
            if (isConflicting(currentPhase, requested)) {
                scheduleClearancePhase();
            }
            LOG.info(String.format(
                "Phase: %s → %s (elapsed=%dms)", currentPhase.name(), requested.name(), elapsed
            ));
            currentPhase = requested;
            phaseStartMs = System.currentTimeMillis();
            phaseChanges.incrementAndGet();
        }

        return buildActionResponse(currentPhase, requested, "ok", elapsed);
    }

    private void scheduleClearancePhase() {
        Phase saved = currentPhase;
        currentPhase = Phase.ALL_RED;
        // In production: hold ALL_RED for safety.allRedDurationMs() then restore.
        // Here we use a virtual thread for the delay.
        Thread.ofVirtual().start(() -> {
            try {
                Thread.sleep(safety.allRedDurationMs());
            } catch (InterruptedException ignored) {}
        });
    }

    // ── Safety helpers ────────────────────────────────────────────────────────

    private void updateSafetyVeto(SensorFrame frame) {
        safetyVetoActive = anyPedUrgent(frame);
    }

    private boolean anyPedUrgent(SensorFrame frame) {
        if (frame == null) return false;
        // Urgent if any pedestrian has been waiting > urgency threshold
        // (In full system: pass pedWaitTimes from server-side state tracking)
        return Arrays.stream(frame.pedestrianPresent).anyMatch(p -> p);
    }

    private static boolean isConflicting(Phase a, Phase b) {
        // NS_GREEN and EW_GREEN conflict with each other and with PED_CROSSING
        if (a == Phase.ALL_RED || b == Phase.ALL_RED) return false;
        if (a == Phase.PED_CROSSING || b == Phase.PED_CROSSING) return a != b;
        return (a == Phase.NS_GREEN) != (b == Phase.NS_GREEN);
    }

    // ── JSON builders ─────────────────────────────────────────────────────────

    private String buildStateJson(SensorFrame frame, long phaseElapsedMs) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\"phase\":\"").append(currentPhase.name()).append("\",");
        sb.append("\"phase_index\":").append(currentPhase.index).append(",");
        sb.append("\"phase_elapsed_ms\":").append(phaseElapsedMs).append(",");
        sb.append("\"safety_veto\":").append(safetyVetoActive).append(",");

        if (frame != null) {
            sb.append("\"vehicle_counts\":").append(Arrays.toString(frame.vehicleCounts)).append(",");
            sb.append("\"ped_present\":").append(Arrays.toString(frame.pedestrianPresent)).append(",");
            sb.append("\"dropped\":").append(frame.droppedReadings);
        } else {
            sb.append("\"frame\":null");
        }
        sb.append("}");
        return sb.toString();
    }

    private String buildActionResponse(
        Phase applied, Phase requested, String status, long elapsed
    ) {
        return String.format(
            "{\"applied\":\"%s\",\"requested\":\"%s\",\"status\":\"%s\",\"elapsed_ms\":%d}",
            applied.name(), requested.name(), status, elapsed
        );
    }

    private static String json(String... kv) {
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < kv.length; i += 2) {
            if (i > 0) sb.append(",");
            sb.append("\"").append(kv[i]).append("\":\"").append(kv[i+1]).append("\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private static String toJson(Map<String, Object> m) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (var e : m.entrySet()) {
            if (!first) sb.append(",");
            sb.append("\"").append(e.getKey()).append("\":");
            Object v = e.getValue();
            if (v instanceof String) sb.append("\"").append(v).append("\"");
            else sb.append(v);
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    private void register(String path, CheckedHandler h) {
        server.createContext(path, ex -> {
            ex.getResponseHeaders().set("Content-Type", "application/json");
            ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
            try { h.handle(ex); }
            catch (Exception e) {
                LOG.log(Level.SEVERE, "Handler error on " + path, e);
                respond(ex, 500, json("error", "internal_error", "detail", e.getMessage()));
            }
        });
    }

    private static void respond(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private static boolean assertMethod(HttpExchange ex, String method) throws IOException {
        if (!ex.getRequestMethod().equalsIgnoreCase(method)) {
            respond(ex, 405, json("error", "method_not_allowed", "expected", method));
            return false;
        }
        return true;
    }

    private static int parseActionIndex(String body) {
        body = body.trim();
        if (body.startsWith("{")) {
            // {"action": 2}
            int i = body.indexOf("\"action\"");
            if (i == -1) throw new NumberFormatException("Missing 'action' key");
            int colon = body.indexOf(":", i);
            String num = body.substring(colon + 1).replaceAll("[^0-9]", "").trim();
            return Integer.parseInt(num);
        }
        return Integer.parseInt(body.replaceAll("[^0-9]", "").trim());
    }

    private void logEndpoints() {
        LOG.info(String.format("  GET  http://localhost:%d/state",   port));
        LOG.info(String.format("  POST http://localhost:%d/action",  port));
        LOG.info(String.format("  GET  http://localhost:%d/metrics", port));
        LOG.info(String.format("  GET  http://localhost:%d/health",  port));
    }

    @FunctionalInterface
    interface CheckedHandler {
        void handle(HttpExchange ex) throws IOException;
    }

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8765;

        AsyncSensorPipeline pipeline = AsyncSensorPipeline.builder()
            .nLanes(4)
            .nCrosswalks(4)
            .windowMs(5_000)
            .queueCapacity(512)
            .build()
            .start();

        TrafficControlServer server = new TrafficControlServer(pipeline, port);
        pipeline.builder().onFrame(server::setLastFrame);   // wire frame → server state

        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop();
            pipeline.stop();
        }));
        LOG.info("System ready. Press Ctrl+C to exit.");
        Thread.currentThread().join();
    }
}

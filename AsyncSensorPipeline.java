package traffic.sensor;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.Consumer;
import java.util.logging.*;

/**
 * AsyncSensorPipeline
 * -------------------
 * High-throughput, non-blocking sensor data ingestion for an urban intersection.
 *
 * Architecture:
 *   Camera / loop-detector threads push {@link SensorReading} objects into a
 *   bounded {@link LinkedBlockingQueue}. A single dedicated consumer thread
 *   drains the queue, aggregates readings within a configurable time window,
 *   and emits a consolidated {@link SensorFrame} to registered listeners.
 *
 * Design patterns:
 *   - Producer–consumer via BlockingQueue (back-pressure by bounding capacity)
 *   - Atomic accumulators for lock-free per-lane state
 *   - ScheduledExecutorService for periodic frame emission
 *   - Builder pattern for FluentConfig
 *
 * Usage:
 * <pre>
 *   AsyncSensorPipeline pipeline = AsyncSensorPipeline.builder()
 *       .windowMs(5_000)
 *       .queueCapacity(512)
 *       .onFrame(frame -> agent.observe(frame.toStateVector()))
 *       .build()
 *       .start();
 *
 *   // From any camera thread:
 *   pipeline.submitReading(SensorReading.camera(crosswalkId, pedestriansDetected));
 *   pipeline.submitReading(SensorReading.loop(laneId, vehicleCount, avgSpeedKmh));
 * </pre>
 */
public final class AsyncSensorPipeline {

    private static final Logger LOG = Logger.getLogger(AsyncSensorPipeline.class.getName());

    // ── Sensor Reading ────────────────────────────────────────────────────────

    public enum SensorType { LOOP_DETECTOR, CAMERA_VEHICLE, CAMERA_PEDESTRIAN }

    public static final class SensorReading {
        public final SensorType type;
        public final int        channelId;   // lane index or crosswalk index
        public final double     value1;      // count / presence
        public final double     value2;      // speed (loop) or confidence (camera)
        public final long       timestampNs;

        private SensorReading(SensorType t, int ch, double v1, double v2) {
            this.type        = t;
            this.channelId   = ch;
            this.value1      = v1;
            this.value2      = v2;
            this.timestampNs = System.nanoTime();
        }

        public static SensorReading loop(int laneId, int vehicleCount, double speedKmh) {
            return new SensorReading(SensorType.LOOP_DETECTOR, laneId, vehicleCount, speedKmh);
        }

        public static SensorReading camera(int crosswalkId, boolean pedestriansPresent) {
            return new SensorReading(
                SensorType.CAMERA_PEDESTRIAN, crosswalkId,
                pedestriansPresent ? 1.0 : 0.0, 1.0
            );
        }

        public static SensorReading cameraVehicle(int laneId, int count) {
            return new SensorReading(SensorType.CAMERA_VEHICLE, laneId, count, 0.0);
        }
    }

    // ── Sensor Frame (consolidated per window) ────────────────────────────────

    public static final class SensorFrame {
        public final long   windowStartNs;
        public final long   windowEndNs;
        public final double[] vehicleCounts;    // per lane, averaged over window
        public final double[] avgSpeedsKmh;     // per lane
        public final boolean[]pedestrianPresent;// per crosswalk (OR over window)
        public final double[] pedConfidence;    // per crosswalk (max over window)
        public final int    droppedReadings;    // back-pressure drops this window

        public SensorFrame(
            long start, long end,
            double[] counts, double[] speeds,
            boolean[] peds, double[] pedConf, int dropped
        ) {
            this.windowStartNs     = start;
            this.windowEndNs       = end;
            this.vehicleCounts     = Arrays.copyOf(counts, counts.length);
            this.avgSpeedsKmh      = Arrays.copyOf(speeds, speeds.length);
            this.pedestrianPresent = Arrays.copyOf(peds,   peds.length);
            this.pedConfidence     = Arrays.copyOf(pedConf,pedConf.length);
            this.droppedReadings   = dropped;
        }

        /** 21-element state vector compatible with Python PPO observation. */
        public double[] toStateVector(
            double maxQueue, double maxWait, double maxPedWait,
            double[] queueLengths, double[] waitTimes, double[] pedWaitTimes,
            int currentPhase, int phaseElapsed, int phaseMaxDur
        ) {
            double[] s = new double[21];
            int n = Math.min(queueLengths.length, 4);
            for (int i = 0; i < n; i++) {
                s[i]     = Math.min(queueLengths[i] / maxQueue, 1.0);
                s[4 + i] = Math.min(waitTimes[i]    / maxWait,  1.0);
            }
            for (int i = 0; i < Math.min(pedestrianPresent.length, 4); i++) {
                s[8  + i] = pedestrianPresent[i] ? 1.0 : 0.0;
                s[12 + i] = Math.min(pedWaitTimes[i] / maxPedWait, 1.0);
            }
            if (currentPhase >= 0 && currentPhase < 4)
                s[16 + currentPhase] = 1.0;
            s[20] = Math.min((double) phaseElapsed / phaseMaxDur, 1.0);
            return s;
        }

        @Override
        public String toString() {
            return String.format(
                "SensorFrame[counts=%s, peds=%s, dropped=%d]",
                Arrays.toString(vehicleCounts),
                Arrays.toString(pedestrianPresent),
                droppedReadings
            );
        }
    }

    // ── Internal accumulators (per window) ────────────────────────────────────

    private static final class WindowAccumulator {
        final int nLanes, nCrosswalks;

        // Vehicle (loop / camera)
        final AtomicLong[]   vehicleSamples;
        final AtomicLong[]   vehicleCountSum;
        final AtomicLong[]   speedSum;
        final AtomicLong[]   speedSamples;

        // Pedestrian
        final AtomicReference<Boolean>[] pedPresent;
        final AtomicLong[]              pedConfSum;
        final AtomicLong[]              pedConfSamples;

        final AtomicInteger droppedReadings = new AtomicInteger(0);
        final long          startNs;

        @SuppressWarnings("unchecked")
        WindowAccumulator(int nLanes, int nCrosswalks) {
            this.nLanes      = nLanes;
            this.nCrosswalks = nCrosswalks;
            this.startNs     = System.nanoTime();

            vehicleSamples  = new AtomicLong[nLanes];
            vehicleCountSum = new AtomicLong[nLanes];
            speedSum        = new AtomicLong[nLanes];
            speedSamples    = new AtomicLong[nLanes];
            pedPresent      = new AtomicReference[nCrosswalks];
            pedConfSum      = new AtomicLong[nCrosswalks];
            pedConfSamples  = new AtomicLong[nCrosswalks];

            for (int i = 0; i < nLanes; i++) {
                vehicleSamples[i]  = new AtomicLong(0);
                vehicleCountSum[i] = new AtomicLong(0);
                speedSum[i]        = new AtomicLong(0);
                speedSamples[i]    = new AtomicLong(0);
            }
            for (int i = 0; i < nCrosswalks; i++) {
                pedPresent[i]    = new AtomicReference<>(Boolean.FALSE);
                pedConfSum[i]    = new AtomicLong(0);
                pedConfSamples[i]= new AtomicLong(0);
            }
        }

        void addVehicle(int laneId, double count, double speedKmh) {
            if (laneId < 0 || laneId >= nLanes) return;
            vehicleCountSum[laneId].addAndGet((long)(count * 1000));
            vehicleSamples[laneId].incrementAndGet();
            speedSum[laneId].addAndGet((long)(speedKmh * 1000));
            speedSamples[laneId].incrementAndGet();
        }

        void addPedestrian(int crosswalkId, boolean present, double confidence) {
            if (crosswalkId < 0 || crosswalkId >= nCrosswalks) return;
            if (present) pedPresent[crosswalkId].set(Boolean.TRUE);
            pedConfSum[crosswalkId].addAndGet((long)(confidence * 1000));
            pedConfSamples[crosswalkId].incrementAndGet();
        }

        SensorFrame toFrame() {
            double[] counts = new double[nLanes];
            double[] speeds = new double[nLanes];
            for (int i = 0; i < nLanes; i++) {
                long s = vehicleSamples[i].get();
                counts[i] = s > 0 ? vehicleCountSum[i].get() / (1000.0 * s) : 0;
                long ss = speedSamples[i].get();
                speeds[i] = ss > 0 ? speedSum[i].get() / (1000.0 * ss) : 0;
            }
            boolean[] peds    = new boolean[nCrosswalks];
            double[]  pedConf = new double[nCrosswalks];
            for (int i = 0; i < nCrosswalks; i++) {
                peds[i] = pedPresent[i].get();
                long cs = pedConfSamples[i].get();
                pedConf[i] = cs > 0 ? pedConfSum[i].get() / (1000.0 * cs) : 0;
            }
            return new SensorFrame(
                startNs, System.nanoTime(),
                counts, speeds, peds, pedConf,
                droppedReadings.get()
            );
        }
    }

    // ── Pipeline core ─────────────────────────────────────────────────────────

    private final int                     nLanes;
    private final int                     nCrosswalks;
    private final long                    windowMs;
    private final BlockingQueue<SensorReading> queue;
    private final List<Consumer<SensorFrame>>  listeners;
    private final ExecutorService             consumer;
    private final ScheduledExecutorService    emitter;
    private volatile WindowAccumulator        accumulator;
    private final AtomicBoolean               running = new AtomicBoolean(false);

    // Statistics
    private final AtomicLong totalReadings   = new AtomicLong(0);
    private final AtomicLong totalDropped    = new AtomicLong(0);
    private final AtomicLong framesEmitted   = new AtomicLong(0);

    private AsyncSensorPipeline(Builder b) {
        this.nLanes      = b.nLanes;
        this.nCrosswalks = b.nCrosswalks;
        this.windowMs    = b.windowMs;
        this.queue       = new LinkedBlockingQueue<>(b.queueCapacity);
        this.listeners   = Collections.unmodifiableList(new ArrayList<>(b.listeners));
        this.accumulator = new WindowAccumulator(nLanes, nCrosswalks);
        this.consumer    = Executors.newSingleThreadExecutor(
            r -> { Thread t = new Thread(r, "sensor-consumer"); t.setDaemon(true); return t; }
        );
        this.emitter     = Executors.newSingleThreadScheduledExecutor(
            r -> { Thread t = new Thread(r, "frame-emitter"); t.setDaemon(true); return t; }
        );
    }

    /** Start the consumer and emitter threads. */
    public AsyncSensorPipeline start() {
        if (!running.compareAndSet(false, true))
            throw new IllegalStateException("Pipeline already running");

        consumer.submit(this::consumeLoop);
        emitter.scheduleAtFixedRate(
            this::emitFrame, windowMs, windowMs, TimeUnit.MILLISECONDS
        );
        LOG.info(String.format(
            "AsyncSensorPipeline started | window=%dms | lanes=%d | crosswalks=%d",
            windowMs, nLanes, nCrosswalks
        ));
        return this;
    }

    /** Non-blocking submit. Returns false and increments dropped counter if queue full. */
    public boolean submitReading(SensorReading reading) {
        if (!running.get()) return false;
        boolean offered = queue.offer(reading);
        totalReadings.incrementAndGet();
        if (!offered) {
            totalDropped.incrementAndGet();
            accumulator.droppedReadings.incrementAndGet();
            LOG.warning("Queue full — reading dropped (back-pressure)");
        }
        return offered;
    }

    /** Blocking submit with timeout. */
    public boolean submitReading(SensorReading reading, long timeoutMs)
            throws InterruptedException {
        boolean offered = queue.offer(reading, timeoutMs, TimeUnit.MILLISECONDS);
        totalReadings.incrementAndGet();
        if (!offered) totalDropped.incrementAndGet();
        return offered;
    }

    public void stop() {
        running.set(false);
        emitter.shutdown();
        consumer.shutdown();
        LOG.info(String.format(
            "Pipeline stopped | total=%d dropped=%d frames=%d",
            totalReadings.get(), totalDropped.get(), framesEmitted.get()
        ));
    }

    public Map<String, Long> stats() {
        return Map.of(
            "totalReadings", totalReadings.get(),
            "totalDropped",  totalDropped.get(),
            "framesEmitted", framesEmitted.get(),
            "queueSize",     (long) queue.size()
        );
    }

    // ── Internal threads ──────────────────────────────────────────────────────

    private void consumeLoop() {
        while (running.get() || !queue.isEmpty()) {
            try {
                SensorReading r = queue.poll(100, TimeUnit.MILLISECONDS);
                if (r == null) continue;
                dispatch(r);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private void dispatch(SensorReading r) {
        WindowAccumulator acc = accumulator;
        switch (r.type) {
            case LOOP_DETECTOR, CAMERA_VEHICLE ->
                acc.addVehicle(r.channelId, r.value1, r.value2);
            case CAMERA_PEDESTRIAN ->
                acc.addPedestrian(r.channelId, r.value1 > 0.5, r.value2);
        }
    }

    private void emitFrame() {
        WindowAccumulator old = accumulator;
        accumulator = new WindowAccumulator(nLanes, nCrosswalks);  // atomic swap

        SensorFrame frame = old.toFrame();
        framesEmitted.incrementAndGet();
        for (Consumer<SensorFrame> listener : listeners) {
            try { listener.accept(frame); }
            catch (Exception e) {
                LOG.log(Level.WARNING, "Frame listener threw: " + e.getMessage(), e);
            }
        }
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private int    nLanes        = 4;
        private int    nCrosswalks   = 4;
        private long   windowMs      = 5_000;
        private int    queueCapacity = 1024;
        private final List<Consumer<SensorFrame>> listeners = new ArrayList<>();

        public Builder nLanes(int n)        { this.nLanes = n;        return this; }
        public Builder nCrosswalks(int n)   { this.nCrosswalks = n;   return this; }
        public Builder windowMs(long ms)    { this.windowMs = ms;     return this; }
        public Builder queueCapacity(int n) { this.queueCapacity = n; return this; }
        public Builder onFrame(Consumer<SensorFrame> l) { listeners.add(l); return this; }

        public AsyncSensorPipeline build() { return new AsyncSensorPipeline(this); }
    }
}

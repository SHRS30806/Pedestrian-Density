package traffic.model;

import java.util.*;

/**
 * Immutable value objects shared between the sensor pipeline,
 * signal controller, and REST API layers.
 *
 * Using Java records (JDK 16+) for concise, immutable data carriers.
 * All numeric fields are validated on construction.
 */
public final class TrafficModels {

    private TrafficModels() {}   // non-instantiable

    // ── Signal Phase ──────────────────────────────────────────────────────────

    public enum SignalPhase {
        NS_GREEN(0, 10_000, 60_000),
        EW_GREEN(1, 10_000, 60_000),
        PED_CROSSING(2, 15_000, 30_000),
        ALL_RED(3, 3_000, 8_000);

        public final int  index;
        public final long minDurationMs;
        public final long maxDurationMs;

        SignalPhase(int index, long minMs, long maxMs) {
            this.index         = index;
            this.minDurationMs = minMs;
            this.maxDurationMs = maxMs;
        }

        public static SignalPhase fromIndex(int i) {
            return values()[i % values().length];
        }

        public boolean conflictsWith(SignalPhase other) {
            if (this == ALL_RED || other == ALL_RED) return false;
            if (this == PED_CROSSING || other == PED_CROSSING) return this != other;
            return (this == NS_GREEN) != (other == NS_GREEN);
        }

        public boolean isPedestrian() {
            return this == PED_CROSSING;
        }
    }

    // ── Intersection State ────────────────────────────────────────────────────

    /**
     * Snapshot of a single intersection's state at one time step.
     * Produced by the sensor pipeline; consumed by the REST API and agent.
     */
    public record IntersectionState(
        String      intersectionId,
        long        timestampMs,
        SignalPhase currentPhase,
        long        phaseElapsedMs,
        double[]    queueLengths,        // vehicles per lane (0–3)
        double[]    avgWaitTimesS,       // seconds per lane
        boolean[]   pedPresent,          // per crosswalk
        double[]    pedWaitTimesS,       // seconds per crosswalk
        boolean     safetyVetoActive
    ) {
        public IntersectionState {
            Objects.requireNonNull(intersectionId);
            Objects.requireNonNull(currentPhase);
            if (queueLengths.length != 4)  throw new IllegalArgumentException("queueLengths must be length 4");
            if (avgWaitTimesS.length != 4) throw new IllegalArgumentException("avgWaitTimesS must be length 4");
            if (pedPresent.length != 4)    throw new IllegalArgumentException("pedPresent must be length 4");
            if (pedWaitTimesS.length != 4) throw new IllegalArgumentException("pedWaitTimesS must be length 4");
        }

        /** Build the 21-dim state vector for the DRL agent. */
        public double[] toStateVector(
            double maxQueue, double maxWait, double maxPedWait, int phaseMaxSteps
        ) {
            double[] s = new double[21];
            for (int i = 0; i < 4; i++) {
                s[i]      = clamp(queueLengths[i]   / maxQueue);
                s[4 + i]  = clamp(avgWaitTimesS[i]  / maxWait);
                s[8 + i]  = pedPresent[i]   ? 1.0 : 0.0;
                s[12 + i] = clamp(pedWaitTimesS[i]  / maxPedWait);
            }
            s[16 + currentPhase.index] = 1.0;
            s[20] = clamp((double) phaseElapsedMs / (phaseMaxSteps * 1000.0));
            return s;
        }

        public boolean anyPedWaiting() {
            for (boolean p : pedPresent) if (p) return true;
            return false;
        }

        private static double clamp(double v) {
            return Math.max(0.0, Math.min(1.0, v));
        }

        /** JSON serialisation (no external dependency). */
        public String toJson() {
            return String.format(
                """
                {
                  "id": "%s",
                  "timestamp_ms": %d,
                  "phase": "%s",
                  "phase_elapsed_ms": %d,
                  "queue_lengths": %s,
                  "avg_wait_s": %s,
                  "ped_present": %s,
                  "ped_wait_s": %s,
                  "safety_veto": %b
                }""",
                intersectionId, timestampMs, currentPhase.name(),
                phaseElapsedMs,
                Arrays.toString(queueLengths),
                Arrays.toString(avgWaitTimesS),
                Arrays.toString(pedPresent),
                Arrays.toString(pedWaitTimesS),
                safetyVetoActive
            );
        }
    }

    // ── Action Request ────────────────────────────────────────────────────────

    /**
     * Represents a phase action submitted by the DRL agent.
     */
    public record ActionRequest(
        String      intersectionId,
        SignalPhase requestedPhase,
        long        submittedAtMs,
        double      agentConfidence   // max softmax probability [0–1]
    ) {
        public ActionRequest {
            Objects.requireNonNull(intersectionId);
            Objects.requireNonNull(requestedPhase);
            if (agentConfidence < 0 || agentConfidence > 1)
                throw new IllegalArgumentException("confidence must be in [0, 1]");
        }

        public static ActionRequest of(String id, int phaseIndex) {
            return new ActionRequest(id, SignalPhase.fromIndex(phaseIndex),
                                     System.currentTimeMillis(), 1.0);
        }
    }

    // ── Action Response ───────────────────────────────────────────────────────

    public enum ActionStatus {
        APPLIED,           // phase change accepted
        HELD_MIN_DURATION, // held current phase (min green not reached)
        SAFETY_VETO,       // pedestrian safety override
        RATE_LIMITED,      // too many requests
        INVALID_ACTION,    // bad phase index
    }

    public record ActionResponse(
        String       intersectionId,
        SignalPhase  requestedPhase,
        SignalPhase  appliedPhase,
        ActionStatus status,
        long         phaseElapsedMs,
        String       detail
    ) {
        public boolean wasAccepted() {
            return status == ActionStatus.APPLIED;
        }

        public String toJson() {
            return String.format(
                """
                {"id":"%s","requested":"%s","applied":"%s","status":"%s","elapsed_ms":%d,"detail":"%s"}""",
                intersectionId, requestedPhase.name(), appliedPhase.name(),
                status.name(), phaseElapsedMs, detail
            );
        }
    }

    // ── Episode Summary ───────────────────────────────────────────────────────

    public record EpisodeSummary(
        String intersectionId,
        int    totalSteps,
        int    totalThroughput,
        int    pedServed,
        int    pedUnsafeEvents,
        int    unnecessaryPedPhases,
        double avgVehicleWaitS,
        double avgPedWaitS,
        double cumulativeReward,
        Map<String, Integer> phaseCounts
    ) {
        public double vehicleWaitImprovement(double baselineWaitS) {
            return (baselineWaitS - avgVehicleWaitS) / baselineWaitS * 100.0;
        }

        @Override
        public String toString() {
            return String.format(
                "EpisodeSummary[%s | steps=%d throughput=%d pedServed=%d "
                + "unsafe=%d avgWait=%.1fs reward=%.2f]",
                intersectionId, totalSteps, totalThroughput, pedServed,
                pedUnsafeEvents, avgVehicleWaitS, cumulativeReward
            );
        }
    }
}

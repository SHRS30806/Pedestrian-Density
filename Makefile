# Makefile — IntelliSignal v2
# ============================================================
# Usage:
#   make install        Install Python dependencies
#   make test           Run Python test suite
#   make train          Train with default config
#   make eval           Run evaluator on best checkpoint
#   make c              Compile C shared library
#   make c-test         Compile and run C self-test
#   make java           Compile Java sources
#   make java-run       Start Java signal control server
#   make clean          Remove build artifacts
# ============================================================

.PHONY: all install test train eval c c-test java java-run clean lint

PYTHON     := python3
PIP        := pip3
PYTEST     := pytest
CC         := gcc
CFLAGS     := -O3 -march=native -std=c17 -Wall -Wextra
JAVAC      := javac
JAVA       := java
JAVA_SRC   := java/src/main/java
JAVA_OUT   := java/out

# ── Python ────────────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt --break-system-packages

test:
	PYTHONPATH=python $(PYTEST) tests/python/ -v --tb=short -q

lint:
	PYTHONPATH=python $(PYTHON) -m flake8 python/ --max-line-length=100 --ignore=E501,W503
	PYTHONPATH=python $(PYTHON) -m mypy python/ --ignore-missing-imports

train:
	cd python && $(PYTHON) train.py --cfg configs/default.yaml

train-fast:
	cd python && $(PYTHON) train.py \
		--cfg configs/default.yaml \
		--run fast_test

eval:
	cd python && $(PYTHON) -c "
from evaluation.evaluator import Evaluator
from agents.ppo_agent import PPOAgent
from configs.config import TrafficConfig
cfg   = TrafficConfig()
agent = PPOAgent(cfg.ppo)
agent.load('results/run_001/checkpoints/best.pt')
ev    = Evaluator(cfg)
print(ev.full_comparison(agent, n_episodes=10))
"

# ── C extension ───────────────────────────────────────────────────────────────

C_SRC  := c/queue_stats.c
C_LIB  := c/queue_stats.so
C_TEST := c/test_queue

c:
	$(CC) $(CFLAGS) -shared -fPIC -o $(C_LIB) $(C_SRC) -lm
	@echo "Built → $(C_LIB)"

c-win:
	$(CC) $(CFLAGS) -shared -o c/queue_stats.dll $(C_SRC) -lm

c-test:
	$(CC) $(CFLAGS) -DTEST_MAIN -o $(C_TEST) $(C_SRC) -lm
	./$(C_TEST)

# ── Java ──────────────────────────────────────────────────────────────────────

$(JAVA_OUT):
	mkdir -p $(JAVA_OUT)

java: $(JAVA_OUT)
	find $(JAVA_SRC) -name "*.java" | xargs $(JAVAC) \
		--enable-preview --release 21 \
		-d $(JAVA_OUT)
	@echo "Compiled → $(JAVA_OUT)"

java-run: java
	$(JAVA) --enable-preview -cp $(JAVA_OUT) \
		traffic.api.TrafficControlServer 8765

java-test: java
	$(JAVA) --enable-preview -cp $(JAVA_OUT) \
		traffic.sensor.AsyncSensorPipeline

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"       -delete 2>/dev/null || true
	rm -f $(C_LIB) $(C_TEST) c/queue_stats.dll
	rm -rf $(JAVA_OUT)
	rm -rf results/run_001
	@echo "Cleaned."

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

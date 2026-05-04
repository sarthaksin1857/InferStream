#!/bin/bash
# Usage:
#   ./scripts/start_all.sh            — start all services and wait
#   ./scripts/start_all.sh --simulate — start services, fire 16 requests, print results

SIMULATE=false
for arg in "$@"; do
    if [[ "$arg" == "--simulate" ]]; then
        SIMULATE=true
    fi
done

echo "Starting InferStream Distributed System..."

# Start the coordinator in the background
echo "Starting Coordinator on port 50051..."
uv run inferstream-coordinator &
COORD_PID=$!

# Give it a few seconds to boot and bind to the port
sleep 2

# Start the frontend UI in the background
echo "Starting Frontend UI on port 8000..."
uv run inferstream-frontend &
FRONTEND_PID=$!

# Start the worker node in the background
echo "Starting Worker Node..."
uv run python src/inferstream/worker/server.py &
WORKER_PID=$!

# Cleanup function to kill background processes on exit
cleanup() {
    echo ""
    echo "Shutting down all InferStream services..."
    kill $COORD_PID $FRONTEND_PID $WORKER_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run the cleanup function
trap cleanup SIGINT SIGTERM

echo ""
echo "========================================================"
echo "🚀 All services started successfully!"
echo "💻 Frontend UI available at: http://localhost:8000"
if $SIMULATE; then
    echo "🧪 Simulation mode: firing 16 requests..."
fi
echo "🛑 Press Ctrl+C at any time to shut everything down."
echo "========================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE: submit 16 requests, poll until all complete, print results
# ─────────────────────────────────────────────────────────────────────────────
if $SIMULATE; then
    # Give the worker a moment to register with the coordinator and load the model.
    echo "⏳ Waiting 8 seconds for the worker to finish loading the model..."
    sleep 8

    FRONTEND="http://localhost:8000"
    NUM_REQUESTS=16
    POLL_INTERVAL=2  # seconds between status checks

    PROMPTS=(
        "Once upon a time in distributed systems,"
        "What is the meaning of life?"
        "How do you build a large language model?"
        "To be or not to be, that is the question."
        "The quick brown fox jumps over the lazy dog."
        "A long time ago in a galaxy far, far away..."
        "It was the best of times, it was the worst of times."
        "Call me Ishmael. Some years ago, never mind how long,"
        "It is a truth universally acknowledged,"
        "In the beginning God created the heavens and the earth."
        "Two roads diverged in a yellow wood,"
        "I wandered lonely as a cloud"
        "Water water everywhere and all the boards did shrink;"
        "Shall I compare thee to a summer's day?"
        "Fourscore and seven years ago our fathers brought forth"
        "Ask not what your country can do for you,"
    )

    echo ""
    echo "📤 Submitting $NUM_REQUESTS requests to $FRONTEND/api/generate ..."
    echo ""

    # Submit all requests and collect their IDs into an array
    declare -a REQUEST_IDS
    declare -a PROMPTS_USED
    for i in $(seq 0 $((NUM_REQUESTS - 1))); do
        PROMPT="${PROMPTS[$i]}"
        RESPONSE=$(curl -sf -X POST "$FRONTEND/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$PROMPT\", \"length\": \"short\"}" 2>&1)
        if [[ $? -ne 0 ]]; then
            echo "  ❌ Request $((i+1)) failed to submit: $RESPONSE"
            continue
        fi
        RID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['request_id'])" 2>/dev/null)
        if [[ -z "$RID" ]]; then
            echo "  ❌ Request $((i+1)) — could not parse request_id from: $RESPONSE"
            continue
        fi
        REQUEST_IDS+=("$RID")
        PROMPTS_USED+=("$PROMPT")
        echo "  ✅ Submitted request $((i+1))/$NUM_REQUESTS → id=$RID"
    done

    TOTAL=${#REQUEST_IDS[@]}
    echo ""
    echo "⏳ Polling for results ($TOTAL requests in flight)..."
    echo ""

    declare -a RESULTS=()
    declare -a RESULT_PROMPTS=()
    DONE_COUNT=0

    # Keep a parallel pending list
    declare -a PENDING_IDS=("${REQUEST_IDS[@]}")
    declare -a PENDING_PROMPTS=("${PROMPTS_USED[@]}")

    while [[ ${#PENDING_IDS[@]} -gt 0 ]]; do
        sleep $POLL_INTERVAL
        STILL_PENDING=()
        STILL_PROMPTS=()
        for idx in "${!PENDING_IDS[@]}"; do
            RID="${PENDING_IDS[$idx]}"
            PROMPT="${PENDING_PROMPTS[$idx]}"
            STATUS_JSON=$(curl -sf "$FRONTEND/api/result/$RID" 2>/dev/null)
            STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
            if [[ "$STATUS" == "COMPLETED" ]]; then
                TEXT=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('generated_text',''))" 2>/dev/null)
                DONE_COUNT=$((DONE_COUNT + 1))
                RESULTS+=("$TEXT")
                RESULT_PROMPTS+=("$PROMPT")
                echo "  ✔ [$DONE_COUNT/$TOTAL] Completed: \"$PROMPT\""
            else
                STILL_PENDING+=("$RID")
                STILL_PROMPTS+=("$PROMPT")
            fi
        done
        PENDING_IDS=("${STILL_PENDING[@]}")
        PENDING_PROMPTS=("${STILL_PROMPTS[@]}")

        # Print a progress tick if anything is still in flight
        if [[ ${#PENDING_IDS[@]} -gt 0 ]]; then
            echo "  ⏳ ${#PENDING_IDS[@]} still generating... ($DONE_COUNT/$TOTAL done)"
            for RID in "${PENDING_IDS[@]}"; do
                STATUS_JSON=$(curl -sf "$FRONTEND/api/result/$RID" 2>/dev/null)
                STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','ERR'))" 2>/dev/null)
                echo "      $RID → $STATUS"
            done
        fi
    done

    echo ""
    echo "========================================================"
    echo "  🏁 ALL $TOTAL REQUESTS COMPLETED"
    echo "========================================================"
    for idx in "${!RESULTS[@]}"; do
        echo ""
        echo "  Prompt:     ${RESULT_PROMPTS[$idx]}"
        echo "  Completion: ${RESULTS[$idx]}"
        echo "  ────────────────────────────────────────────────────"
    done
    echo ""
    echo "Simulation complete. Services still running — Ctrl+C to stop."
fi

# Wait indefinitely so the script doesn't exit until interrupted
wait

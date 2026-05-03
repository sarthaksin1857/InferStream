#!/bin/bash

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
echo "🛑 Press Ctrl+C at any time to shut everything down."
echo "========================================================"
echo ""

# Wait indefinitely so the script doesn't exit until interrupted
wait

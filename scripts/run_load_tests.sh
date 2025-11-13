#!/bin/bash

# Load Testing Automation Script
# Runs a comprehensive suite of load tests

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         🚀 STARTING COMPREHENSIVE LOAD TEST SUITE 🚀             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Create reports directory
mkdir -p reports/load_tests

# Check if API is running
echo "📋 Checking if API is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running"
else
    echo "❌ API is not running on http://localhost:8000"
    echo "   Start the API first: uvicorn source.api.main:app --reload"
    exit 1
fi

echo ""

# Test 1: Baseline (10 users, 2 minutes)
echo "════════════════════════════════════════════════════════════════════"
echo "Test 1: Baseline Load Test (10 users, 2 minutes)"
echo "════════════════════════════════════════════════════════════════════"
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 10 \
    --spawn-rate 2 \
    --run-time 2m \
    --headless \
    --html=reports/load_tests/baseline_10users.html \
    --csv=reports/load_tests/baseline_10users

echo ""

# Test 2: Medium Load (50 users, 5 minutes)
echo "════════════════════════════════════════════════════════════════════"
echo "Test 2: Medium Load Test (50 users, 5 minutes)"
echo "════════════════════════════════════════════════════════════════════"
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --html=reports/load_tests/medium_50users.html \
    --csv=reports/load_tests/medium_50users

echo ""

# Test 3: High Load (100 users, 5 minutes)
echo "════════════════════════════════════════════════════════════════════"
echo "Test 3: High Load Test (100 users, 5 minutes)"
echo "════════════════════════════════════════════════════════════════════"
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --html=reports/load_tests/high_100users.html \
    --csv=reports/load_tests/high_100users

echo ""

# Test 4: Stress Test (200 users, 10 minutes)
echo "════════════════════════════════════════════════════════════════════"
echo "Test 4: Stress Test (200 users, 10 minutes)"
echo "════════════════════════════════════════════════════════════════════"
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users 200 \
    --spawn-rate 20 \
    --run-time 10m \
    --headless \
    --html=reports/load_tests/stress_200users.html \
    --csv=reports/load_tests/stress_200users

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║             ✅ ALL LOAD TESTS COMPLETE ✅                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Reports generated in: reports/load_tests/"
echo ""
echo "HTML Reports:"
echo "  - reports/load_tests/baseline_10users.html"
echo "  - reports/load_tests/medium_50users.html"
echo "  - reports/load_tests/high_100users.html"
echo "  - reports/load_tests/stress_200users.html"
echo ""
echo "CSV Data:"
echo "  - reports/load_tests/baseline_10users_stats.csv"
echo "  - reports/load_tests/medium_50users_stats.csv"
echo "  - reports/load_tests/high_100users_stats.csv"
echo "  - reports/load_tests/stress_200users_stats.csv"
echo ""

#!/bin/bash
# Automated Load Testing Runner
# Runs comprehensive load tests against the Geo Climate API
#
# Usage:
#   ./tests/load/load-test-runner.sh <environment> <test-type>
#
# Examples:
#   ./tests/load/load-test-runner.sh staging smoke
#   ./tests/load/load-test-runner.sh staging load
#   ./tests/load/load-test-runner.sh staging stress
#   ./tests/load/load-test-runner.sh production soak

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT=${1:-staging}
TEST_TYPE=${2:-load}

# API endpoints by environment
case $ENVIRONMENT in
    dev)
        API_HOST="http://localhost:8000"
        ;;
    staging)
        API_HOST="https://api.staging.geo-climate.example.com"
        ;;
    production)
        API_HOST="https://api.geo-climate.example.com"
        ;;
    *)
        echo -e "${RED}Invalid environment: ${ENVIRONMENT}${NC}"
        echo "Valid options: dev, staging, production"
        exit 1
        ;;
esac

# Test configurations
case $TEST_TYPE in
    smoke)
        # Smoke test: minimal load to verify system works
        USERS=10
        SPAWN_RATE=2
        DURATION="2m"
        ;;
    load)
        # Load test: expected production load
        USERS=100
        SPAWN_RATE=10
        DURATION="10m"
        ;;
    stress)
        # Stress test: beyond normal load
        USERS=500
        SPAWN_RATE=50
        DURATION="15m"
        ;;
    spike)
        # Spike test: sudden increase in load
        USERS=1000
        SPAWN_RATE=100
        DURATION="5m"
        ;;
    soak)
        # Soak test: sustained load over time
        USERS=200
        SPAWN_RATE=20
        DURATION="60m"
        ;;
    *)
        echo -e "${RED}Invalid test type: ${TEST_TYPE}${NC}"
        echo "Valid options: smoke, load, stress, spike, soak"
        exit 1
        ;;
esac

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Geo Climate Load Testing${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Test Type: ${TEST_TYPE}${NC}"
echo -e "${BLUE}Target: ${API_HOST}${NC}"
echo -e "${BLUE}Users: ${USERS}${NC}"
echo -e "${BLUE}Spawn Rate: ${SPAWN_RATE}/s${NC}"
echo -e "${BLUE}Duration: ${DURATION}${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Create results directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="tests/load/results/${ENVIRONMENT}/${TEST_TYPE}/${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo -e "${YELLOW}Results will be saved to: ${RESULTS_DIR}${NC}\n"

# Pre-test health check
echo -e "${BLUE}Running pre-test health check...${NC}"
if curl -sf ${API_HOST}/health > /dev/null; then
    echo -e "${GREEN}✓ API is healthy${NC}\n"
else
    echo -e "${RED}✗ API health check failed${NC}"
    echo -e "${RED}Aborting load test${NC}"
    exit 1
fi

# Run Locust in headless mode
echo -e "${BLUE}Starting load test...${NC}\n"

locust \
    -f tests/load/locustfile.py \
    --headless \
    --users ${USERS} \
    --spawn-rate ${SPAWN_RATE} \
    --run-time ${DURATION} \
    --host ${API_HOST} \
    --html ${RESULTS_DIR}/report.html \
    --csv ${RESULTS_DIR}/stats \
    --logfile ${RESULTS_DIR}/locust.log \
    --loglevel INFO

echo -e "\n${GREEN}Load test completed!${NC}\n"

# Post-test analysis
echo -e "${BLUE}Analyzing results...${NC}\n"

# Parse CSV results
if [ -f "${RESULTS_DIR}/stats_stats.csv" ]; then
    echo -e "${YELLOW}Summary Statistics:${NC}"
    echo "-------------------"

    # Calculate key metrics
    python3 << EOF
import pandas as pd
import sys

try:
    # Read stats
    df = pd.read_csv('${RESULTS_DIR}/stats_stats.csv')

    # Filter out aggregated rows
    df = df[df['Name'] != 'Aggregated']

    if len(df) == 0:
        print("No data found")
        sys.exit(1)

    # Calculate metrics
    print(f"\n📊 Request Statistics:")
    print(f"  Total Requests: {df['Request Count'].sum():,.0f}")
    print(f"  Failed Requests: {df['Failure Count'].sum():,.0f}")
    print(f"  Failure Rate: {(df['Failure Count'].sum() / df['Request Count'].sum() * 100):.2f}%")

    print(f"\n⏱️  Latency (ms):")
    print(f"  Median (P50): {df['Median Response Time'].median():.0f} ms")
    print(f"  95th Percentile (P95): {df['95%'].median():.0f} ms")
    print(f"  99th Percentile (P99): {df['99%'].median():.0f} ms")
    print(f"  Max: {df['Max Response Time'].max():.0f} ms")

    print(f"\n🚀 Throughput:")
    print(f"  Requests/sec: {df['Requests/s'].sum():.2f}")

    # Check SLA compliance
    print(f"\n✅ SLA Compliance:")
    p95 = df['95%'].median()
    error_rate = (df['Failure Count'].sum() / df['Request Count'].sum() * 100)

    if p95 < 1000:
        print(f"  ✓ P95 latency < 1s: PASS ({p95:.0f} ms)")
    else:
        print(f"  ✗ P95 latency < 1s: FAIL ({p95:.0f} ms)")

    if error_rate < 1.0:
        print(f"  ✓ Error rate < 1%: PASS ({error_rate:.2f}%)")
    else:
        print(f"  ✗ Error rate < 1%: FAIL ({error_rate:.2f}%)")

    # Top slowest endpoints
    print(f"\n🐌 Slowest Endpoints:")
    slowest = df.nlargest(5, 'Median Response Time')[['Name', 'Median Response Time', '95%']]
    for idx, row in slowest.iterrows():
        print(f"  {row['Name']}: P50={row['Median Response Time']:.0f}ms, P95={row['95%']:.0f}ms")

    # Save summary
    with open('${RESULTS_DIR}/summary.txt', 'w') as f:
        f.write(f"Load Test Summary\n")
        f.write(f"================\n")
        f.write(f"Environment: ${ENVIRONMENT}\n")
        f.write(f"Test Type: ${TEST_TYPE}\n")
        f.write(f"Total Requests: {df['Request Count'].sum():,.0f}\n")
        f.write(f"Failed Requests: {df['Failure Count'].sum():,.0f}\n")
        f.write(f"P95 Latency: {p95:.0f} ms\n")
        f.write(f"Error Rate: {error_rate:.2f}%\n")
        f.write(f"RPS: {df['Requests/s'].sum():.2f}\n")

except Exception as e:
    print(f"Error analyzing results: {e}")
    sys.exit(1)
EOF
fi

# Post-test health check
echo -e "\n${BLUE}Running post-test health check...${NC}"
if curl -sf ${API_HOST}/health > /dev/null; then
    echo -e "${GREEN}✓ API is still healthy${NC}\n"
else
    echo -e "${RED}✗ API health check failed after load test${NC}\n"
fi

# Generate comparison if previous results exist
PREV_TEST=$(find tests/load/results/${ENVIRONMENT}/${TEST_TYPE} -maxdepth 1 -type d -name "20*" | sort -r | sed -n '2p')
if [ -n "$PREV_TEST" ] && [ -f "${PREV_TEST}/stats_stats.csv" ]; then
    echo -e "${BLUE}Comparing with previous test...${NC}\n"

    python3 << EOF
import pandas as pd

try:
    # Read current and previous results
    current = pd.read_csv('${RESULTS_DIR}/stats_stats.csv')
    previous = pd.read_csv('${PREV_TEST}/stats_stats.csv')

    # Filter aggregated rows
    current = current[current['Name'] == 'Aggregated'].iloc[0]
    previous = previous[previous['Name'] == 'Aggregated'].iloc[0]

    # Calculate changes
    p95_change = ((current['95%'] - previous['95%']) / previous['95%'] * 100)
    error_change = ((current['Failure Count'] / current['Request Count']) -
                    (previous['Failure Count'] / previous['Request Count'])) * 100
    rps_change = ((current['Requests/s'] - previous['Requests/s']) / previous['Requests/s'] * 100)

    print("📈 Comparison with Previous Test:")
    print(f"  P95 Latency: {p95_change:+.1f}%")
    print(f"  Error Rate: {error_change:+.2f}%")
    print(f"  Throughput: {rps_change:+.1f}%")
    print()

except Exception as e:
    print(f"Could not compare with previous test: {e}")
EOF
fi

# Final summary
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Test Complete!${NC}"
echo -e "${BLUE}=====================================${NC}"
echo -e "Results saved to: ${RESULTS_DIR}"
echo -e "HTML Report: ${RESULTS_DIR}/report.html"
echo -e "CSV Stats: ${RESULTS_DIR}/stats_stats.csv"
echo -e "Log File: ${RESULTS_DIR}/locust.log"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review HTML report: open ${RESULTS_DIR}/report.html"
echo "2. Analyze performance metrics in Prometheus"
echo "3. Check application logs for errors"
echo "4. Compare with SLA requirements"
echo ""

# Exit with error if SLA not met (only for staging/production)
if [ "$ENVIRONMENT" != "dev" ]; then
    if [ -f "${RESULTS_DIR}/stats_stats.csv" ]; then
        python3 << EOF
import pandas as pd
import sys

df = pd.read_csv('${RESULTS_DIR}/stats_stats.csv')
df = df[df['Name'] == 'Aggregated'].iloc[0]

p95 = df['95%']
error_rate = (df['Failure Count'] / df['Request Count'] * 100)

if p95 >= 1000 or error_rate >= 1.0:
    print("❌ SLA requirements not met!")
    sys.exit(1)
else:
    print("✅ All SLA requirements met!")
    sys.exit(0)
EOF
    fi
fi

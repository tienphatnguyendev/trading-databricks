#!/bin/bash
STREAM1="tick-stream-btcusdt"
STREAM2="tick-stream-ethusdt"
REGION="ap-southeast-1"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

START_TIME=$(date -u -v-5M +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "5 mins ago" +%Y-%m-%dT%H:%M:%SZ)
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

clear
echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN} AWS Kinesis Health Monitor${NC}"
echo -e "${CYAN} Time: $(date)${NC}"
echo -e "${CYAN}==================================================${NC}"

for STREAM in $STREAM1 $STREAM2; do
  echo ""
  echo -e "${MAGENTA}>>> Stream: ${YELLOW}$STREAM${MAGENTA} <<<${NC}"
  echo "Incoming Bytes (1min Avg):"
  aws cloudwatch get-metric-statistics --namespace AWS/Kinesis --metric-name IncomingBytes --dimensions Name=StreamName,Value=$STREAM --start-time $START_TIME --end-time $END_TIME --period 60 --statistics Average --region $REGION --query 'sort_by(Datapoints, &Timestamp)[-1].[Timestamp, Average]' --output text | awk -v c="\033[0;32m" -v nc="\033[0m" '{printf "  %s | %s%s Bytes/min%s\n", $1, c, $2, nc}'

  echo "Write Provisioned Throughput Exceeded (1min Sum):"
  aws cloudwatch get-metric-statistics --namespace AWS/Kinesis --metric-name WriteProvisionedThroughputExceeded --dimensions Name=StreamName,Value=$STREAM --start-time $START_TIME --end-time $END_TIME --period 60 --statistics Sum --region $REGION --query 'sort_by(Datapoints, &Timestamp)[-1].[Timestamp, Sum]' --output text | awk -v c="\033[0;31m" -v nc="\033[0m" '{printf "  %s | %s%s Errors%s\n", $1, c, $2, nc}'
done

echo ""
echo -e "${CYAN}Press Ctrl+C to exit.${NC}"
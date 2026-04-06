#!/bin/bash
FUNCTION="news-scraper-TradingViewScraper-xtdIdfZ9nANw"
REGION="ap-southeast-1"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

START_TIME=$(date -u -v-15M +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "15 mins ago" +%Y-%m-%dT%H:%M:%SZ)
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

clear
echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN} AWS Lambda Live Monitor: ${YELLOW}$FUNCTION${NC}"
echo -e "${CYAN} Time: $(date)${NC}"
echo -e "${CYAN}==================================================${NC}\n"

echo -e "${YELLOW}--- Invocations (Last 15 mins, 5min periods) ---${NC}"
aws cloudwatch get-metric-statistics --namespace AWS/Lambda --metric-name Invocations --dimensions Name=FunctionName,Value=$FUNCTION --start-time $START_TIME --end-time $END_TIME --period 300 --statistics Sum --region $REGION --query 'sort_by(Datapoints, &Timestamp)[*].[Timestamp, Sum]' --output text | awk -v c="\033[0;32m" -v nc="\033[0m" '{printf "Time: %s | Count: %s%s%s\n", $1, c, $2, nc}'

echo ""
echo -e "${YELLOW}--- Errors (Last 15 mins, 5min periods) ---${NC}"
aws cloudwatch get-metric-statistics --namespace AWS/Lambda --metric-name Errors --dimensions Name=FunctionName,Value=$FUNCTION --start-time $START_TIME --end-time $END_TIME --period 300 --statistics Sum --region $REGION --query 'sort_by(Datapoints, &Timestamp)[*].[Timestamp, Sum]' --output text | awk -v c="\033[0;31m" -v nc="\033[0m" '{printf "Time: %s | Errors: %s%s%s\n", $1, c, $2, nc}'

echo ""
echo -e "${YELLOW}--- Duration Avg (Last 15 mins) ---${NC}"
aws cloudwatch get-metric-statistics --namespace AWS/Lambda --metric-name Duration --dimensions Name=FunctionName,Value=$FUNCTION --start-time $START_TIME --end-time $END_TIME --period 300 --statistics Average --region $REGION --query 'sort_by(Datapoints, &Timestamp)[-1].[Timestamp, Average]' --output text | awk -v c="\033[0;36m" -v nc="\033[0m" '{printf "Time: %s | Avg Duration: %s%.2f ms%s\n", $1, c, $2, nc}'

echo ""
echo -e "${CYAN}Press Ctrl+C to exit.${NC}"
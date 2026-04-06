#!/bin/bash
CLUSTER="tick-streamer-cluster"
SERVICE="tick-streamer"
REGION="ap-southeast-1"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

START_TIME=$(date -u -v-5M +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "5 mins ago" +%Y-%m-%dT%H:%M:%SZ)
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

clear
echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN} AWS ECS Live Monitor: ${YELLOW}$SERVICE${NC}"
echo -e "${CYAN} Time: $(date)${NC}"
echo -e "${CYAN}==================================================${NC}\n"

STATUS=$(aws ecs describe-services --cluster $CLUSTER --services $SERVICE --region $REGION --query 'services[0].[status,runningCount,desiredCount]' --output text)
echo -e "Status / Running / Desired: ${GREEN}$STATUS${NC}\n"

echo -e "${YELLOW}--- CPU Utilization (Last 5 mins) ---${NC}"
aws cloudwatch get-metric-statistics --namespace AWS/ECS --metric-name CPUUtilization --dimensions Name=ClusterName,Value=$CLUSTER Name=ServiceName,Value=$SERVICE --start-time $START_TIME --end-time $END_TIME --period 60 --statistics Average --region $REGION --query 'sort_by(Datapoints, &Timestamp)[-1].[Timestamp, Average]' --output text | awk -v c="\033[0;32m" -v nc="\033[0m" '{printf "Time: %s | CPU: %s%.2f%%%s\n", $1, c, $2, nc}'

echo ""
echo -e "${YELLOW}--- Memory Utilization (Last 5 mins) ---${NC}"
aws cloudwatch get-metric-statistics --namespace AWS/ECS --metric-name MemoryUtilization --dimensions Name=ClusterName,Value=$CLUSTER Name=ServiceName,Value=$SERVICE --start-time $START_TIME --end-time $END_TIME --period 60 --statistics Average --region $REGION --query 'sort_by(Datapoints, &Timestamp)[-1].[Timestamp, Average]' --output text | awk -v c="\033[0;32m" -v nc="\033[0m" '{printf "Time: %s | Memory: %s%.2f%%%s\n", $1, c, $2, nc}'
  
echo ""
echo -e "${CYAN}Press Ctrl+C to exit.${NC}"
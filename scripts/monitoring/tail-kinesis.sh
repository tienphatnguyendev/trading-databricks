#!/bin/bash
# Usage: ./tail-kinesis.sh <stream-name>
# Example: ./tail-kinesis.sh tick-stream-btcusdt

STREAM=${1:-"tick-stream-btcusdt"}
REGION="ap-southeast-1"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}Connecting to Kinesis Stream: ${YELLOW}$STREAM${NC}..."

# Get ALL shards
SHARDS=$(aws kinesis list-shards --stream-name $STREAM --region $REGION --query 'Shards[*].ShardId' --output text)

if [ -z "$SHARDS" ] || [ "$SHARDS" == "None" ]; then
    echo -e "${RED}Error: Could not find shards for stream $STREAM${NC}"
    exit 1
fi

echo -e "${GREEN}Found Shards: ${NC}$SHARDS"
echo -e "${CYAN}Tailing ALL shards in real-time (Press Ctrl+C to stop)...${NC}"
echo -e "${CYAN}----------------------------------------${NC}"

# Function to tail a single shard
tail_shard() {
    local shard=$1
    local iter=$(aws kinesis get-shard-iterator --stream-name $STREAM --shard-id $shard --shard-iterator-type LATEST --region $REGION --query 'ShardIterator' --output text)
    
    while true; do
        # Get records for this shard
        local res=$(aws kinesis get-records --shard-iterator "$iter" --region $REGION 2>/dev/null)
        
        # If we got a response, extract the next iterator and the data
        if [ -n "$res" ]; then
            local next_iter=$(echo "$res" | jq -r '.NextShardIterator')
            
            # Update iterator if valid
            if [ "$next_iter" != "null" ] && [ -n "$next_iter" ]; then
                iter=$next_iter
            fi
            
            local records=$(echo "$res" | jq -r '.Records[].Data')
            
            # Decode and print if there is data
            if [ -n "$records" ] && [ "$records" != "null" ]; then
                for rec in $records; do
                    # Cross-platform base64 decode
                    local decoded=$(echo "$rec" | base64 --decode 2>/dev/null || echo "$rec" | base64 -D 2>/dev/null || echo "$rec" | base64 -d 2>/dev/null)
                    echo -e "${CYAN}[${shard}]${NC} ${GREEN}${decoded}${NC}"
                done
            fi
        fi
        
        # Avoid hitting Kinesis API limits (max 5 reads/sec per shard)
        sleep 2
    done
}

# Array to store background PIDs
PIDS=()

# Trap Ctrl+C (SIGINT) to kill all background jobs cleanly
trap 'echo -e "\n${YELLOW}Stopping tail...${NC}"; for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null; done; exit' SIGINT

# Spawn a background tailing process for EVERY shard
for shard in $SHARDS; do
    tail_shard $shard &
    PIDS+=($!)
done

# Keep the main script alive waiting for background jobs, but allow interrupts
while true; do
    sleep 1
done
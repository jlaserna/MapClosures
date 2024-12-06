#!/bin/bash

# Check if screen or tmux is installed
if command -v screen &> /dev/null; then
    SESSION_TOOL="screen"
elif command -v tmux &> /dev/null; then
    SESSION_TOOL="tmux"
else
    echo "Error: Neither screen nor tmux is installed."
    exit 1
fi

# Session name
SESSION_NAME="map_closure"

# Combinations of Town and Sequence
TOWNS=("Town01" "Town02" "Bridge01" "Bridge02" "Roundabout01")
SEQUENCES=("Aeva" "Avia" "Ouster")

# Temporary script to run the commands
SCRIPT_PATH="/tmp/map_closure_commands.sh"

# Generate the execution script
echo "#!/bin/bash" > $SCRIPT_PATH
for TOWN in "${TOWNS[@]}"; do
    for SEQUENCE in "${SEQUENCES[@]}"; do
        echo "map_closure_pipeline --eval --dataloader helipr /data/hdd/dataset/HeLiPR/${TOWN} --sequence ${SEQUENCE} ./output" >> $SCRIPT_PATH
    done
done

# Make the script executable
chmod +x $SCRIPT_PATH

# Create and run the session in the background
if [ "$SESSION_TOOL" == "screen" ]; then
    screen -dmS $SESSION_NAME bash -c "$SCRIPT_PATH && rm -f $SCRIPT_PATH"
    echo "Session '$SESSION_NAME' created. To attach, use: screen -r $SESSION_NAME"
elif [ "$SESSION_TOOL" == "tmux" ]; then
    tmux new-session -d -s $SESSION_NAME bash -c "$SCRIPT_PATH && rm -f $SCRIPT_PATH"
    echo "Session '$SESSION_NAME' created. To attach, use: tmux attach -t $SESSION_NAME"
fi

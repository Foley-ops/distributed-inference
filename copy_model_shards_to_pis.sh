#!/bin/bash
# Copy model shard files to Pi nodes

echo "Creating shard directories on Pi nodes..."
ssh cc@master-pi "mkdir -p ~/datasets/model_shards"
ssh cc@core-pi "mkdir -p ~/datasets/model_shards"

echo "Copying all shard files to master-pi..."
scp -r ~/datasets/model_shards/* cc@master-pi:~/datasets/model_shards/

echo "Copying all shard files to core-pi..."
scp -r ~/datasets/model_shards/* cc@core-pi:~/datasets/model_shards/

echo "Done! All shard files copied to Pi nodes."

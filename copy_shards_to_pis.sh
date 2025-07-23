#!/bin/bash
# Copy all shard files to Pi nodes

echo "Creating shard directories on Pi nodes..."
ssh cc@master-pi "mkdir -p ~/projects/distributed-inference/model_shards"
ssh cc@core-pi "mkdir -p ~/projects/distributed-inference/model_shards"

echo "Copying shard files to master-pi..."
scp -r model_shards/* cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying shard files to core-pi..."
scp -r model_shards/* cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Done! Shard files copied to all Pi nodes."

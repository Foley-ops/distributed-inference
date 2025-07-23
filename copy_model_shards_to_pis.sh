#!/bin/bash
# Copy model shard files to Pi nodes

echo "Creating shard directories on Pi nodes..."
ssh cc@master-pi "mkdir -p ~/projects/distributed-inference/model_shards"
ssh cc@core-pi "mkdir -p ~/projects/distributed-inference/model_shards"


echo "Copying mobilenetv2 shards to master-pi..."
scp -r model_shards/mobilenetv2 cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying mobilenetv2 shards to core-pi..."
scp -r model_shards/mobilenetv2 cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Copying resnet18 shards to master-pi..."
scp -r model_shards/resnet18 cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying resnet18 shards to core-pi..."
scp -r model_shards/resnet18 cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Copying vgg16 shards to master-pi..."
scp -r model_shards/vgg16 cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying vgg16 shards to core-pi..."
scp -r model_shards/vgg16 cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Copying alexnet shards to master-pi..."
scp -r model_shards/alexnet cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying alexnet shards to core-pi..."
scp -r model_shards/alexnet cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Copying inceptionv3 shards to master-pi..."
scp -r model_shards/inceptionv3 cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying inceptionv3 shards to core-pi..."
scp -r model_shards/inceptionv3 cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Copying squeezenet shards to master-pi..."
scp -r model_shards/squeezenet cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying squeezenet shards to core-pi..."
scp -r model_shards/squeezenet cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Done! All shard files copied to Pi nodes."

#!/bin/bash

#if [ "$#" -ne 2 ]; then
#    echo "Usage: $0 <username> <gpu_number>"
#    exit 1
#fi

#USERNAME=$1
#GPU_NUMBER=$2

docker run -it --rm --gpus all \
    --name nvidia-physicsnemo \
    -v "$(pwd)":/sci-ml \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    nvidia-physicsnemo:25.06 bash
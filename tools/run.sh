xhost local:root

docker run --rm --gpus all --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    -v "$PWD":/workspace \
    -v /dev:/dev:ro \
    --net=host \
    $REGISTRY_NAME/$IMAGE_NAME:latest \
    python main.py $@
if [[ -z "$REGISTRY_NAME" ]]; then
    echo "[*] REGISTRY_NAME is empty."
    echo "[*] Please set REGISTRY_NAME in your environment."
    exit 1
fi

if [[ -z "$IMAGE_NAME" ]]; then
    echo "[*] IMAGE_NAME is empty"
    echo "[*] Please set IMAGE_NAME in your environment."
    exit 1
fi

xhost local:root

docker run --rm --gpus all --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    -v "$PWD":/workspace \
    -v /dev:/dev:ro \
    --network=host \
    $REGISTRY_NAME/$IMAGE_NAME:latest \
    python main.py $@

sudo chown -R $USER:$USER ./
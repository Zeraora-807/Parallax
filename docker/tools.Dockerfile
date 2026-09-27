FROM debian:bookworm-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils coreutils checksec file ca-certificates tar unzip tshark \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /inputs /work

USER 65534:65534
WORKDIR /workspace

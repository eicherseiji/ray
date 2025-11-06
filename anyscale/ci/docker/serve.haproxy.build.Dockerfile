# syntax=docker/dockerfile:1.3-labs

# Base this on the existing servebuild image to avoid duplicating dependencies like wrk
FROM cr.ray.io/rayproject/servebuild-py3.10

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

SHELL ["/bin/bash", "-ice"]

RUN <<EOF
#!/bin/bash

set -euo pipefail

# Install HAProxy dependencies
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    curl \
    libc6-dev \
    liblua5.3-dev \
    libpcre3-dev \
    libssl-dev \
    socat \
    wget \
    zlib1g-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Create haproxy user and group
sudo groupadd -r haproxy
sudo useradd -r -g haproxy haproxy

# Download and compile HAProxy from official source
HAPROXY_VERSION="2.8.12"
HAPROXY_BUILD_DIR="$(mktemp -d)"
wget -O "${HAPROXY_BUILD_DIR}/haproxy.tar.gz" "https://www.haproxy.org/download/2.8/src/haproxy-${HAPROXY_VERSION}.tar.gz"
tar -xzf "${HAPROXY_BUILD_DIR}/haproxy.tar.gz" -C "${HAPROXY_BUILD_DIR}" --strip-components=1
make -C "${HAPROXY_BUILD_DIR}" TARGET=linux-glibc USE_OPENSSL=1 USE_ZLIB=1 USE_PCRE=1 USE_LUA=1 USE_PROMEX=1
sudo make -C "${HAPROXY_BUILD_DIR}" install
rm -rf "${HAPROXY_BUILD_DIR}"

# Create HAProxy directories
sudo mkdir -p /etc/haproxy /run/haproxy /var/log/haproxy
sudo chown -R haproxy:haproxy /run/haproxy

# Allow the ray user to manage HAProxy files without password
echo "ray ALL=(ALL) NOPASSWD: /bin/cp * /etc/haproxy/*, /bin/touch /etc/haproxy/*, /usr/local/sbin/haproxy*" | sudo tee /etc/sudoers.d/haproxy-ray

EOF

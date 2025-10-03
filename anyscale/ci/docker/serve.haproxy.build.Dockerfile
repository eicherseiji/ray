# syntax=docker/dockerfile:1.3-labs

# Base this on the existing servebuild image to avoid duplicating dependencies like wrk
FROM cr.ray.io/rayproject/servebuild

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
    wget \
    curl \
    socat \
    libssl-dev \
    libpcre3-dev \
    zlib1g-dev \
    && sudo rm -rf /var/lib/apt/lists/*

# Create haproxy user and group
sudo groupadd -r haproxy
sudo useradd -r -g haproxy haproxy

# Download and compile HAProxy from official source
wget -O /tmp/haproxy-2.4.25.tar.gz https://www.haproxy.org/download/2.4/src/haproxy-2.4.25.tar.gz
tar -xzf /tmp/haproxy-2.4.25.tar.gz -C /tmp
make -C /tmp/haproxy-2.4.25 TARGET=linux-glibc USE_OPENSSL=1 USE_ZLIB=1 USE_PCRE=1
sudo make -C /tmp/haproxy-2.4.25 install
rm -rf /tmp/haproxy-2.4.25*

# Create HAProxy directories
sudo mkdir -p /etc/haproxy /run/haproxy /var/log/haproxy
sudo chown -R haproxy:haproxy /run/haproxy

# Allow the ray user to manage HAProxy files without password
echo "ray ALL=(ALL) NOPASSWD: /bin/cp * /etc/haproxy/*, /bin/touch /etc/haproxy/*, /usr/local/sbin/haproxy*" | sudo tee /etc/sudoers.d/haproxy-ray

EOF

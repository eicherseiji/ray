# bray

bray (pronounced like pray, but with a b) is a CLI utility that offloads local Ray
builds to a remote Anyscale job, dramatically improving build and test times. It also
creates a workspace with the build artifacts for quick e2e testing.

## features
- Remote compilation on Linux-based build servers
- Integrated remote Bazel cache for faster incremental builds

## prerequisites
Before using bray, ensure the following are set up on your machine:
- Python 3.8+
- Python packages: anyscale, click
- Logged into Anyscale Staging, on your bash shell:
  > export ANYSCALE_HOST=https://console.anyscale-staging.com
  > anyscale login

Note: You typically need to re-run anyscale login weekly, as credentials are
short-lived. All other setup steps are one-time only.

## installation
If you already have a Ray Turbo repository that includes bray, install it with:
  > ./anyscale/core/bray/install_bray.sh

## usage
To view help and available options:
  > bray --help

Basic example usage:
  > bray --build-name <ANY_NAME> --ray-dir <RAY_CHECKOUT_DIRECTORY>

- --build-name: A unique name for your remote build session
- --ray-dir: Path to the local Ray or RayTurbo checkout to be built remotely

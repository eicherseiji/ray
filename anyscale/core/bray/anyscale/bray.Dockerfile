FROM 830883877497.dkr.ecr.us-west-2.amazonaws.com/anyscale/runtime:__REMOTE_BUILD_COMMIT_BASE__-py312-cu123-min-as

RUN echo "__REMOTE_RANDOM_ECHO__"
RUN mkdir -p /tmp/patches /tmp/newray && \
    aws s3 cp s3://core-bazel-cache/__REMOTE_BUILD_NAME__/ray-opt.tgz /tmp/ray-opt.tgz && \
    tar -xvzf /tmp/ray-opt.tgz -C /tmp/patches && \
    tar -xvzf /opt/anyscale/ray-opt.tgz -C /tmp/newray && \
    cp -rf /tmp/patches/python/ray /tmp/newray/ && \
    cd /tmp/newray && \
    tar -czf ray-opt.tgz ray ray-3.0.0.dev0.dist-info && \
    sudo mv ray-opt.tgz /opt/anyscale/ray-opt.tgz

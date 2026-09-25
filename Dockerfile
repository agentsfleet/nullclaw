# syntax=docker/dockerfile:1

# ── Stage 1: Build ────────────────────────────────────────────
# Build on the target architecture so Alpine supplies matching libcurl headers
# and libraries for the native HTTP transport.
FROM alpine:3.23 AS builder

ARG ZIG_VERSION=0.16.0

RUN apk add --no-cache \
    bash build-base ca-certificates curl git musl-dev \
    openssl-dev openssl-libs-static pkgconf python3 zlib-dev zlib-static

# A small static libcurl keeps the self-execing sandbox child independent of
# host shared libraries. Pin the upstream source and disable unused protocols
# and GCC-LTO optional libraries while retaining HTTPS and proxy support.
RUN set -eu; \
    curl -fsSLo /tmp/curl.tar.xz https://curl.se/download/curl-8.22.0.tar.xz; \
    echo 'f7ef3ae8a22e521f289803fe93543eb64c329b58aa73a9e224dfd915a2a5f4f7  /tmp/curl.tar.xz' | sha256sum -c -; \
    tar -xf /tmp/curl.tar.xz -C /tmp; \
    cd /tmp/curl-8.22.0; \
    CFLAGS='-Os -fno-lto -ffunction-sections -fdata-sections' ./configure \
      --disable-shared --enable-static --with-openssl --without-libpsl \
      --without-brotli --without-zstd --without-libidn2 --without-nghttp2 \
      --without-libssh2 --disable-ldap --disable-ldaps --disable-ftp \
      --disable-rtsp --disable-telnet --disable-tftp --disable-smb \
      --disable-smtp --disable-pop3 --disable-imap --disable-gopher \
      --disable-mqtt --disable-dict --disable-file --disable-manual \
      --prefix=/opt/curl-min >/tmp/curl-configure.log; \
    make -j4 -C lib >/tmp/curl-make.log; \
    mkdir -p /opt/curl-min/include /opt/curl-min/lib; \
    cp lib/.libs/libcurl.a /opt/curl-min/lib/libcurl.a; \
    cp -R include/curl /opt/curl-min/include/curl

WORKDIR /app
COPY .github/scripts/install-zig.sh .github/scripts/install-zig.sh
COPY build.zig build.zig.zon ./
COPY src/ src/
COPY vendor/sqlite3/ vendor/sqlite3/

RUN set -eu; \
    mkdir -p /tmp/zig-path; \
    GITHUB_PATH=/tmp/zig-path/path RUNNER_TEMP=/opt bash .github/scripts/install-zig.sh "${ZIG_VERSION}"; \
    ln -sf "$(cat /tmp/zig-path/path)/zig" /usr/local/bin/zig; \
    test "$(zig version)" = "0.16.0"

ARG VERSION=dev
RUN --mount=type=cache,target=/root/.cache/zig \
    --mount=type=cache,target=/app/.zig-cache \
    zig build -Doptimize=ReleaseSmall -Dstatic=true -Dversion="${VERSION}"

# ── Stage 2: Config Prep ─────────────────────────────────────
FROM busybox:1.38 AS config

# Keep config.json at the volume root so existing compose volumes remain readable.
RUN mkdir -p /nullclaw-data/workspace

RUN cat > /nullclaw-data/config.json << 'EOF'
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "openrouter/anthropic/claude-sonnet-4"
      }
    }
  },
  "models": {
    "providers": {
      "openrouter": {}
    }
  },
  "gateway": {
    "port": 3000,
    "host": "::",
    "allow_public_bind": true
  }
}
EOF

# Default runtime runs as non-root (uid/gid 65534).
# Keep writable ownership for HOME/workspace in safe mode.
RUN chown -R 65534:65534 /nullclaw-data

# ── Stage 3: Runtime Base (shared) ────────────────────────────
FROM alpine:3.23 AS release-base

LABEL org.opencontainers.image.source=https://github.com/nullclaw/nullclaw

RUN apk add --no-cache ca-certificates curl git tzdata

COPY --from=builder /app/zig-out/bin/nullclaw /usr/local/bin/nullclaw
COPY --from=config /nullclaw-data /nullclaw-data

ENV NULLCLAW_WORKSPACE=/nullclaw-data/workspace
ENV NULLCLAW_HOME=/nullclaw-data
ENV HOME=/nullclaw-data
ENV SHELL=/bin/sh
ENV NULLCLAW_GATEWAY_PORT=3000

WORKDIR /nullclaw-data
EXPOSE 3000
ENTRYPOINT ["nullclaw"]
CMD ["gateway", "--port", "3000", "--host", "::"]

# Optional autonomous mode (explicit opt-in):
#   make build DOCKER_TARGET=release-root IMAGE=nullclaw:root
FROM release-base AS release-root
USER 0:0

# Safe default image (used when no --target is provided)
FROM release-base AS release
USER 65534:65534

#!/usr/bin/env bash
# Reuse the enabled Windows loopback proxy only for this setup/training process.
# No proxy, network mode, listener or firewall configuration is changed.
if [[ "${PPSUC_HOST_PROXY_PORT:-}" =~ ^[0-9]+$ ]]; then
  PPSUC_HOST_GATEWAY="$(ip -4 route show default | awk 'NR==1 {print $3}')"
  PPSUC_PROCESS_PROXY="http://$PPSUC_HOST_GATEWAY:$PPSUC_HOST_PROXY_PORT"
  if curl --proxy "$PPSUC_PROCESS_PROXY" -fsSI --connect-timeout 3 --max-time 8 https://github.com >/dev/null; then
    export HTTP_PROXY="$PPSUC_PROCESS_PROXY" HTTPS_PROXY="$PPSUC_PROCESS_PROXY"
    export http_proxy="$PPSUC_PROCESS_PROXY" https_proxy="$PPSUC_PROCESS_PROXY"
    export NO_PROXY='localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn'
    export no_proxy="$NO_PROXY"
    echo 'Using the existing Windows proxy for this process; PyPI mirror stays direct.'
  else
    echo 'Existing Windows proxy is unavailable from WSL; using direct downloads.'
  fi
fi

#!/bin/sh
# 브라우저 웹캠(getUserMedia)은 https/localhost 에서만 동작 → 자체 서명 인증서로 https 제공
set -e
CERT_DIR=/certs
mkdir -p "$CERT_DIR"
if [ ! -f "$CERT_DIR/cert.pem" ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=asl-demo" \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -addext "subjectAltName=IP:${HOST_IP:-127.0.0.1},DNS:localhost" 2>/dev/null
fi
exec uvicorn aslrob.web.app:app --host 0.0.0.0 --port 8080 \
  --ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem"

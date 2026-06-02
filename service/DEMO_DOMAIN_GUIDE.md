# Demo Domain Guide

This project includes a one-command script for sharing your local Streamlit app to other devices over a public URL.

## A) Fast demo link (random domain, no account)

```bash
cd /Users/who/Desktop/code_it/project01/final_files/서비스_깃
./scripts/share_demo.sh
```

The terminal prints a URL like:

```text
https://xxxxx.trycloudflare.com
```

Share that link to open the app from another device.

## B) Fixed custom domain (your own domain)

If you want a stable domain like `demo.yourdomain.com`, use a named Cloudflare Tunnel.

1) Sign in and create tunnel credentials:

```bash
cloudflared tunnel login
cloudflared tunnel create search-rfp-demo
```

2) Route DNS to that tunnel:

```bash
cloudflared tunnel route dns search-rfp-demo demo.yourdomain.com
```

3) Create config file (`~/.cloudflared/config.yml`):

```yaml
tunnel: search-rfp-demo
credentials-file: /Users/<you>/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: demo.yourdomain.com
    service: http://localhost:8501
  - service: http_status:404
```

4) Run Streamlit:

```bash
python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

5) Run tunnel:

```bash
cloudflared tunnel run search-rfp-demo
```

Now open `https://demo.yourdomain.com` on any device.

## Troubleshooting

- If the app keeps loading forever, test:
  - `--server.enableCORS=false`
  - `--server.enableWebsocketCompression=false`
- If Quick Tunnel fails and mentions config file conflict, temporarily rename:
  - `~/.cloudflared/config.yml` or `~/.cloudflared/config.yaml`

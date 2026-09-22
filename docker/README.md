# docker/

Docker Compose setup — 3 services on the client machine.

Populated by Tasks 9.1 (backend Dockerfile) and 9.2 (docker-compose.yml).

## Planned layout

```
docker/
├── postgres/           # init SQL, volume config
├── redis/              # redis.conf
└── backend/            # (Dockerfile lives at ../backend/Dockerfile per convention)
```

`docker-compose.yml` lives at repo root so `docker compose up` from the repo root just works.

See `../docs/TRD.md` §10 for the topology.

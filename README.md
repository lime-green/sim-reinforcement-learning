# Manual setup
- Clone and setup https://github.com/lime-green/wow-sims-wotlk locally
- In wow-sims directory: 
```
git checkout dk-rl
go run sim/agent/* 1234
```
- In this directory: `make learn`

# Docker setup

Build and start the sim server:
```bash
make docker-start
```

Train the agent:
```bash
make docker-learn
```

# Health Endpoints

## Overview

The health endpoints provide service status information for monitoring and orchestration tools like Kubernetes or load balancers.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Generic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |

---

## GET /health

**Generic Health Check**

Returns the overall health status of the API service.

### Flow Diagram

```mermaid
flowchart LR
    A[Client Request] --> B["/health endpoint"]
    B --> C{Service Running?}
    C -->|Yes| D["Return {status: healthy}"]
    C -->|No| E[No Response / Timeout]
```

### Response

```json
{
  "status": "healthy"
}
```

---

## GET /health/live

**Liveness Probe**

Indicates whether the service process is running. Used by orchestrators to determine if the container needs to be restarted.

### Flow Diagram

```mermaid
flowchart LR
    A[Kubernetes/Orchestrator] --> B["/health/live"]
    B --> C{Process Alive?}
    C -->|Yes| D["Return {status: alive}"]
    C -->|No| E[Restart Container]
```

### Response

```json
{
  "status": "alive"
}
```

---

## GET /health/ready

**Readiness Probe**

Indicates whether the service is ready to accept traffic. Can be extended to check database/storage connectivity.

### Flow Diagram

```mermaid
flowchart LR
    A[Load Balancer] --> B["/health/ready"]
    B --> C{Ready to Serve?}
    C -->|Yes| D["Return {status: ready}"]
    C -->|No| E[Remove from Pool]
    
    subgraph "Future Checks"
        F[DB Connection]
        G[Storage Access]
        H[Model Loaded]
    end
```

### Response

```json
{
  "status": "ready"
}
```

---

## Usage

These endpoints are typically used in:
- **Kubernetes**: `livenessProbe` and `readinessProbe` configuration
- **Docker Compose**: `healthcheck` configuration
- **Load Balancers**: Backend health monitoring
- **Monitoring Systems**: Uptime tracking

# Blog 24: Deploying AI Applications

## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 60-90 minutes
**Hands-on time:** 2-4 hours (depending on cloud provider setup)
**Total investment:** ~4-5 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Containerize AI applications** with Docker for consistent deployment
2. **Deploy to major cloud platforms** (AWS, GCP, Azure)
3. **Scale with Kubernetes** for high-availability services
4. **Implement monitoring and observability** for ML systems
5. **Build CI/CD pipelines** for continuous model deployment
6. **Design production-ready APIs** with proper error handling and rate limiting

> **How to read this blog:** If you are new to deployment and infrastructure, start with the Docker section and the Production Application Structure. Skip the cloud-specific sections (AWS CDK, GCP Cloud Run, Azure Container Apps) until you need them for a real project. If you already deploy web services, focus on the AI-specific monitoring (LLM metrics, token tracking, cost estimation) and the GPU configuration sections. The Kubernetes section assumes familiarity with basic K8s concepts.

---

## Prerequisites

Before starting this blog, you should:

1. **Completed Blog 14-19** (AI APIs, chatbot, RAG, function calling, LangChain) -- you need a working AI application to deploy
2. **Basic Docker knowledge** -- ability to write a Dockerfile and run `docker build`/`docker run`
3. **Command-line comfort** -- you will work with `kubectl`, `gcloud`, `aws`, or `az` CLIs
4. **A cloud account** (optional but recommended) -- free tiers on AWS, GCP, or Azure are sufficient for exercises

If you have never used Docker, work through the [Docker Getting Started guide](https://docs.docker.com/get-started/) first (about 1 hour).

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **ML model training pipelines** -- this blog covers deploying inference services, not training infrastructure (GPU clusters, distributed training, experiment tracking). MLflow, Weights & Biases, and training orchestration are separate disciplines.
- **MLOps maturity models** -- we cover CI/CD and monitoring basics, but not full MLOps frameworks (feature stores, model registries with lineage, A/B testing infrastructure, shadow deployments).
- **Cost optimization in depth** -- we show cost tables and monitoring, but reserved instances, spot/preemptible instances, and detailed cloud cost engineering are beyond scope.
- **Security hardening** -- we cover basics (non-root users, secrets management, CORS), but not WAFs, DDoS protection, SOC 2 compliance, or penetration testing.
- **Networking deep dives** -- VPCs, subnets, service meshes (Istio), and network policies are mentioned but not explained in detail.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Deploying AI applications requires careful consideration of infrastructure, scaling, monitoring, and cost management. The choices made here directly impact reliability, performance, and operational costs.

**Deployment Options Comparison:**

| Approach | Setup Time | Scalability | Cost | Control |
|----------|------------|-------------|------|---------|
| Serverless (Lambda) | Hours | Auto | Pay-per-use | Limited |
| Managed Services | Days | Auto | Premium | Medium |
| Container (ECS/GKE) | Days | Manual/Auto | Medium | High |
| Kubernetes | Weeks | Full control | Varies | Full |
| Self-hosted VMs | Days | Manual | Low | Full |

> **Note on costs:** The cost estimates below are rough approximations based on publicly listed cloud pricing as of early 2025. Actual costs vary significantly based on region, reserved capacity, negotiated discounts, and workload patterns. Always run your own cost estimates using your cloud provider's pricing calculator before making infrastructure decisions.

**Cost Considerations:**

| Workload Type | Recommended Approach | Typical Cost Range |
|---------------|---------------------|-------------------|
| Sporadic (<1K/day) | Serverless | $5-50/month |
| Moderate (10K/day) | Container services | $100-500/month |
| High (100K+/day) | Kubernetes | $500-2000/month |
| ML Inference (GPU) | GPU instances | $500-5000/month |

**Key Decisions:**
1. **GPU Requirements**: Self-host vs cloud GPU pricing -- see the GPU Configuration section below
2. **Latency Requirements**: Regional deployment strategy
3. **Data Residency**: Compliance with regulations
4. **Scaling Pattern**: Predictable vs bursty traffic

---

## Why Deploying AI Applications Is Different

Before diving into configs, understand what makes AI deployment unique compared to deploying a standard web application:

### The Three AI Deployment Challenges

**1. Non-Deterministic Latency.** A traditional API endpoint takes roughly the same time per request. An LLM call to GPT-4o can take 500ms or 15 seconds depending on output length, provider load, and prompt complexity. This breaks standard auto-scaling assumptions — CPU utilization stays low while requests pile up waiting for external API responses.

**Why this matters for deployment:** You cannot rely solely on CPU-based auto-scaling. You need request-count-based scaling (HPA with custom metrics) or concurrency-based scaling (Cloud Run's `containerConcurrency`). The Kubernetes HPA in this blog uses both CPU and `http_requests_per_second` for this reason.

**2. High Memory, Low CPU.** If you run local models (not just calling APIs), models consume 2-16GB of VRAM/RAM but use minimal CPU during inference. This means:
- Resource `requests` and `limits` must be tuned for memory, not CPU
- Pods can appear "idle" to CPU-based autoscalers while being fully loaded
- Model loading at startup takes 30-120 seconds, requiring generous `initialDelaySeconds` on health probes

**3. External Dependency Fragility.** Your service depends on LLM providers (OpenAI, Anthropic) that have real outages. Unlike a database that you control, you cannot restart a downed provider. You need:
- Circuit breakers to fail fast instead of queueing requests
- Retry with exponential backoff (but NOT for all errors — retrying a 400 wastes money)
- Cost tracking because a retry storm can 10x your bill in minutes

### Key Design Decisions Explained

Every configuration choice in this blog follows from these challenges. Here is the reasoning behind non-obvious decisions:

| Configuration | Value | Why This Value |
|--------------|-------|---------------|
| `maxUnavailable: 0` in RollingUpdate | Zero pods removed before new ones are ready | AI services have slow startup (model loading). If you remove pods before replacements are ready, you get cascading failures as remaining pods get overloaded |
| `podAntiAffinity` | Spread pods across nodes | If one node fails, you lose at most 1/N capacity instead of everything |
| HPA `scaleDown.stabilizationWindowSeconds: 300` | Wait 5 minutes before scaling down | LLM traffic is bursty. Scaling down too fast means you have to cold-start pods again when the next burst arrives. The 60s scale-up vs 300s scale-down asymmetry is intentional |
| Histogram buckets `[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]` | Wider than typical web buckets | LLM calls routinely take 2-10 seconds. Standard web buckets (100ms, 250ms, 500ms) would put all LLM calls in the overflow bucket, losing resolution |
| `containerConcurrency: 80` in Cloud Run | 80 concurrent requests per container | LLM calls are IO-bound (waiting for provider response), so a single container can handle many concurrent requests. For CPU-bound local inference, set this to 1-4 |
| `cpu-throttling: "false"` in Cloud Run | Always allocate full CPU | With throttling, CPU is only allocated during request processing. AI services often do background work (model warmup, cache management) that needs CPU between requests |

---

## Containerization with Docker

### Building Production Docker Images

```dockerfile
# Dockerfile for FastAPI AI Service
# Multi-stage build for smaller final image

# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
# Note: The top-level 'version' key is deprecated in modern Docker Compose (v2+).
# Docker Compose now infers the schema version automatically.

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/aiapp
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app  # For development hot-reload
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aiapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Vector database for embeddings
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  grafana_data:
```

### Production Application Structure

```python
"""
Production FastAPI Application Structure
"""
# app/main.py

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)
LLM_REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)
LLM_TOKENS_USED = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting application...")

    # Initialize connections
    from app.services import init_services
    await init_services()

    yield

    # Shutdown
    logger.info("Shutting down...")
    from app.services import cleanup_services
    await cleanup_services()


app = FastAPI(
    title="AI Application API",
    description="Production AI service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — SECURITY: Configure allowed origins explicitly for production.
# Using ["*"] permits any domain to call your API, which is acceptable
# for public APIs but dangerous for authenticated services.
# For production with authentication, list your specific frontend domains.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"  # Default: local dev only
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# Path normalization to prevent high-cardinality metric explosion.
# Without this, paths like /api/v1/chat/abc123 and /api/v1/chat/xyz789
# create unbounded label values, eventually crashing Prometheus.
KNOWN_PREFIXES = ["/api/v1/chat", "/api/v1/embeddings", "/api/v1/images"]

def normalize_path(path: str) -> str:
    """Collapse dynamic path segments into static labels."""
    for prefix in KNOWN_PREFIXES:
        if path.startswith(prefix):
            # /api/v1/chat/abc123 → /api/v1/chat/{id}
            suffix = path[len(prefix):]
            if suffix and suffix != "/":
                return f"{prefix}/{{id}}"
            return prefix
    # System endpoints pass through as-is
    if path in ("/health", "/health/ready", "/metrics"):
        return path
    return "/other"  # Catch-all for unknown paths


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Record metrics with normalized path to prevent cardinality explosion
    duration = time.time() - start_time
    normalized = normalize_path(request.url.path)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=normalized,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=normalized
    ).observe(duration)

    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check - verify all dependencies."""
    from app.services import check_dependencies
    deps = await check_dependencies()

    if all(deps.values()):
        return {"status": "ready", "dependencies": deps}

    raise HTTPException(status_code=503, detail={"status": "not ready", "dependencies": deps})


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


# Include routers
from app.routers import chat, embeddings, images
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4
    )
```

---

## Cloud Deployment

### AWS Deployment

```python
"""
AWS Deployment Configuration
"""
# infrastructure/aws/cdk_stack.py

from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_elasticloadbalancingv2 as elbv2,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
    Duration,
    RemovalPolicy
)
from constructs import Construct


class AIServiceStack(Stack):
    """AWS CDK Stack for AI Service Deployment."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC
        vpc = ec2.Vpc(
            self, "AIServiceVPC",
            max_azs=2,
            nat_gateways=1
        )

        # ECS Cluster
        cluster = ecs.Cluster(
            self, "AIServiceCluster",
            vpc=vpc,
            container_insights=True
        )

        # ECR Repository
        repository = ecr.Repository(
            self, "AIServiceRepo",
            repository_name="ai-service",
            removal_policy=RemovalPolicy.RETAIN
        )

        # Secrets
        api_key_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "OpenAIKey",
            secret_name="openai-api-key"
        )

        # Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self, "AIServiceTaskDef",
            memory_limit_mib=2048,
            cpu=1024
        )

        # Container
        container = task_definition.add_container(
            "AIServiceContainer",
            image=ecs.ContainerImage.from_ecr_repository(repository, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="ai-service",
                log_retention=logs.RetentionDays.ONE_MONTH
            ),
            environment={
                "LOG_LEVEL": "INFO",
                "ENVIRONMENT": "production"
            },
            secrets={
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(api_key_secret)
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3
            )
        )

        container.add_port_mappings(
            ecs.PortMapping(container_port=8000)
        )

        # Fargate Service with ALB
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "AIService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            public_load_balancer=True,
            listener_port=443,
            redirect_http=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True
            )
        )

        # Auto Scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10
        )

        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60)
        )

        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=1000,
            target_group=service.target_group
        )

        # Health check configuration
        service.target_group.configure_health_check(
            path="/health",
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
            interval=Duration.seconds(30)
        )
```

### GCP Deployment

```yaml
# infrastructure/gcp/cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-service
  labels:
    cloud.googleapis.com/location: us-central1
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/PROJECT_ID/ai-service:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
          env:
            - name: LOG_LEVEL
              value: "INFO"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-api-key
                  key: latest
          livenessProbe:
            httpGet:
              path: /health
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health/ready
            initialDelaySeconds: 5
            periodSeconds: 10
```

```bash
#!/bin/bash
# deploy-gcp.sh

# Set variables
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="ai-service"

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest \
    --set-env-vars LOG_LEVEL=INFO
```

### Azure Deployment

```yaml
# infrastructure/azure/azure-container-apps.yaml
apiVersion: apps/v1
kind: ContainerApp
metadata:
  name: ai-service
spec:
  resourceGroupName: ai-service-rg
  location: eastus
  properties:
    managedEnvironmentId: /subscriptions/.../managedEnvironments/ai-env
    configuration:
      secrets:
        - name: openai-api-key
          keyVaultUrl: https://your-keyvault.vault.azure.net/secrets/openai-api-key
      ingress:
        external: true
        targetPort: 8000
        traffic:
          - latestRevision: true
            weight: 100
      registries:
        - server: youracr.azurecr.io
          identity: system
    template:
      containers:
        - name: ai-service
          image: youracr.azurecr.io/ai-service:latest
          resources:
            cpu: 2.0
            memory: 4Gi
          env:
            - name: LOG_LEVEL
              value: INFO
            - name: OPENAI_API_KEY
              secretRef: openai-api-key
          probes:
            - type: Liveness
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 10
              periodSeconds: 30
            - type: Readiness
              httpGet:
                path: /health/ready
                port: 8000
              initialDelaySeconds: 5
              periodSeconds: 10
      scale:
        minReplicas: 1
        maxReplicas: 10
        rules:
          - name: http-scaling
            http:
              metadata:
                concurrentRequests: "100"
```

---

## GPU Configuration for AI Workloads

Many AI applications -- especially those running local model inference (not just calling APIs) -- require GPU resources. This section covers the key decisions.

### When You Need GPUs

| Workload | GPU Needed? | Notes |
|----------|-------------|-------|
| Calling OpenAI/Anthropic APIs | No | The provider runs inference on their GPUs |
| Running Hugging Face models locally | Usually yes | Small models (distilbert) can run on CPU; larger models need GPU |
| Image generation (Stable Diffusion) | Yes | CPU inference is impractically slow (minutes per image) |
| Embedding generation at scale | Depends | CPU works for small batches; GPU needed for high throughput |
| Fine-tuned model serving | Usually yes | Depends on model size |

### Kubernetes GPU Configuration

```yaml
# kubernetes/gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-inference-service
  namespace: ai-services
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-inference
  template:
    metadata:
      labels:
        app: ai-inference
    spec:
      containers:
        - name: inference
          image: your-registry/ai-inference:v1.0.0
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
              nvidia.com/gpu: "1"  # Request 1 GPU
            limits:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"  # Limit to 1 GPU
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
      # Node selector ensures scheduling on GPU nodes
      nodeSelector:
        accelerator: nvidia-tesla-t4
      # Tolerate the GPU node taint
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### Docker with GPU Support

```dockerfile
# Dockerfile.gpu — for local model inference
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Run with GPU access
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Run Docker container with GPU access (requires nvidia-container-toolkit)
docker run --gpus all -p 8000:8000 ai-inference:latest

# Verify GPU is accessible inside the container
docker run --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

### Cloud GPU Pricing Comparison

> **Pricing changes frequently.** The table below reflects approximate on-demand pricing as of early 2025. Always check your provider's current pricing page. Spot/preemptible instances can be 60-90% cheaper but may be interrupted.

| GPU Type | AWS (p3/g5) | GCP | Azure | Use Case |
|----------|------------|-----|-------|----------|
| T4 (16GB) | ~$0.53/hr | ~$0.35/hr | ~$0.53/hr | Inference, small fine-tuning |
| A10G (24GB) | ~$1.01/hr | ~$0.90/hr | N/A | Medium inference, fine-tuning |
| A100 (40GB) | ~$3.91/hr | ~$2.93/hr | ~$3.40/hr | Large model inference, training |
| A100 (80GB) | ~$4.90/hr | ~$3.67/hr | ~$4.23/hr | LLM fine-tuning, large batches |

**Decision framework for GPU vs CPU:**
1. If you only call external APIs (OpenAI, Anthropic) -- no GPU needed
2. If latency per inference > 1 second on CPU and you need < 200ms -- use GPU
3. If running models with > 1B parameters -- GPU strongly recommended
4. Always benchmark your specific workload: `time python inference_benchmark.py`

---

## Kubernetes Deployment

### Production Kubernetes Configuration

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-services
  labels:
    name: ai-services

---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-service
  namespace: ai-services
  labels:
    app: ai-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-service
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: ai-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: ai-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
        - name: ai-service
          image: your-registry/ai-service:v1.0.0
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          env:
            - name: LOG_LEVEL
              value: "INFO"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: openai-api-key
            - name: REDIS_URL
              valueFrom:
                configMapKeyRef:
                  name: ai-service-config
                  key: redis-url
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: ai-service
                topologyKey: kubernetes.io/hostname

---
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-service
  namespace: ai-services
spec:
  selector:
    app: ai-service
  ports:
    - port: 80
      targetPort: 8000
      name: http
  type: ClusterIP

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-service
  namespace: ai-services
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: ai-service-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ai-service
                port:
                  number: 80

---
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-service
  namespace: ai-services
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### Helm Chart Structure

```yaml
# helm/ai-service/Chart.yaml
apiVersion: v2
name: ai-service
description: AI Service Helm Chart
type: application
version: 1.0.0
appVersion: "1.0.0"

---
# helm/ai-service/values.yaml
replicaCount: 3

image:
  repository: your-registry/ai-service
  tag: "latest"
  pullPolicy: Always

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ai-service-tls
      hosts:
        - api.yourdomain.com

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

env:
  LOG_LEVEL: INFO

secrets:
  - name: OPENAI_API_KEY
    secretName: api-keys
    secretKey: openai-api-key

configMaps:
  - name: REDIS_URL
    configMapName: ai-service-config
    configMapKey: redis-url

probes:
  liveness:
    path: /health
    initialDelaySeconds: 10
    periodSeconds: 30
  readiness:
    path: /health/ready
    initialDelaySeconds: 5
    periodSeconds: 10
```

---

## Monitoring and Observability

### Comprehensive Monitoring Setup

```python
"""
Monitoring and Observability Configuration
"""
import logging
import time
from functools import wraps
from typing import Dict, Any, Optional
import json

from prometheus_client import Counter, Histogram, Gauge, Info
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ============ Logging ============

class StructuredLogger:
    """JSON structured logging for production."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(self.JsonFormatter())
        self.logger.addHandler(handler)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }

            if hasattr(record, 'extra_data'):
                log_data.update(record.extra_data)

            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_data)

    def log(self, level: str, message: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            "", 0, message, [], None
        )
        record.extra_data = kwargs
        self.logger.handle(record)


# ============ Prometheus Metrics ============

class MetricsCollector:
    """Prometheus metrics for AI service."""

    def __init__(self, service_name: str = "ai_service"):
        self.service_name = service_name

        # HTTP metrics
        self.http_requests = Counter(
            f"{service_name}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )

        self.http_latency = Histogram(
            f"{service_name}_http_latency_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        # LLM metrics
        self.llm_requests = Counter(
            f"{service_name}_llm_requests_total",
            "Total LLM API requests",
            ["provider", "model", "status"]
        )

        self.llm_latency = Histogram(
            f"{service_name}_llm_latency_seconds",
            "LLM API latency",
            ["provider", "model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.llm_tokens = Counter(
            f"{service_name}_llm_tokens_total",
            "Total LLM tokens used",
            ["provider", "model", "type"]
        )

        self.llm_cost = Counter(
            f"{service_name}_llm_cost_dollars",
            "Total LLM API cost in dollars",
            ["provider", "model"]
        )

        # Active requests
        self.active_requests = Gauge(
            f"{service_name}_active_requests",
            "Currently active requests"
        )

        # Service info
        self.service_info = Info(
            f"{service_name}_info",
            "Service information"
        )

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        self.http_requests.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        self.http_latency.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ):
        self.llm_requests.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()

        self.llm_latency.labels(
            provider=provider,
            model=model
        ).observe(duration)

        self.llm_tokens.labels(
            provider=provider,
            model=model,
            type="input"
        ).inc(input_tokens)

        self.llm_tokens.labels(
            provider=provider,
            model=model,
            type="output"
        ).inc(output_tokens)

        self.llm_cost.labels(
            provider=provider,
            model=model
        ).inc(cost)


# ============ Distributed Tracing ============

def setup_tracing(service_name: str, otlp_endpoint: str):
    """Setup OpenTelemetry distributed tracing."""
    provider = TracerProvider()

    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=otlp_endpoint)
    )
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    # Auto-instrument frameworks
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor().instrument()

    return trace.get_tracer(service_name)


# ============ Retry with Exponential Backoff ============

class LLMRetryPolicy:
    """
    Retry policy for LLM API calls with exponential backoff.

    Key design decisions:
    - Only retry on transient errors (429, 500, 502, 503, 504, timeouts).
      NEVER retry 400 (bad request) or 401 (auth) — those waste money.
    - Exponential backoff with jitter prevents thundering herd when a
      provider recovers from an outage and all clients retry simultaneously.
    - Cost cap: track retry cost to prevent a retry storm from 10x-ing your bill.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        max_retry_cost_dollars: float = 1.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retry_cost_dollars = max_retry_cost_dollars
        self._retry_cost_accumulated = 0.0

    # Status codes that are safe to retry (transient server/rate-limit errors)
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Decide whether to retry based on error type and attempt count."""
        if attempt >= self.max_retries:
            return False

        # Check cost cap — stop retrying if we've spent too much
        if self._retry_cost_accumulated >= self.max_retry_cost_dollars:
            return False

        # Timeout errors — always retry
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True

        # HTTP errors — only retry on transient status codes
        status_code = getattr(error, 'status_code', None)
        if status_code is not None:
            return status_code in self.RETRYABLE_STATUS_CODES

        # Unknown errors — do not retry (fail loud)
        return False

    def get_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        import random
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add 0-50% jitter to prevent thundering herd
        jitter = delay * random.uniform(0, 0.5)
        return delay + jitter

    def record_retry_cost(self, cost: float):
        self._retry_cost_accumulated += cost


# ============ Decorator for LLM Calls ============

def track_llm_call(metrics: MetricsCollector, provider: str, model: str):
    """Decorator to track LLM API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            result: Optional[Any] = None

            try:
                result = await func(*args, **kwargs)
                return result

            except (ConnectionError, TimeoutError) as e:
                status = "error"
                raise

            except ValueError as e:
                status = "error"
                raise

            except Exception as e:
                status = "error"
                raise

            finally:
                duration = time.time() - start_time

                # Safely extract token counts — result may be None if
                # the call raised an exception.
                input_tokens = 0
                output_tokens = 0
                if result is not None:
                    usage = getattr(result, 'usage', None)
                    if usage is not None:
                        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                        output_tokens = getattr(usage, 'completion_tokens', 0) or 0

                # Calculate cost (simplified)
                cost = calculate_cost(provider, model, input_tokens, output_tokens)

                metrics.record_llm_request(
                    provider=provider,
                    model=model,
                    status=status,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost
                )

        return wrapper
    return decorator


def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate API cost based on provider and model.

    NOTE: These prices are approximate as of early 2025. Always verify
    current pricing at your provider's pricing page.
    """
    # Pricing per 1K tokens (simplified)
    pricing = {
        ("openai", "gpt-4o"): (0.005, 0.015),
        ("openai", "gpt-4o-mini"): (0.00015, 0.0006),
        ("anthropic", "claude-3-5-sonnet"): (0.003, 0.015),
    }

    input_rate, output_rate = pricing.get((provider, model), (0.001, 0.002))

    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "AI Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_service_http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Request Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ai_service_http_latency_seconds_bucket[5m]))",
            "legendFormat": "p95 {{endpoint}}"
          }
        ]
      },
      {
        "title": "LLM API Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ai_service_llm_latency_seconds_bucket[5m]))",
            "legendFormat": "p95 {{model}}"
          }
        ]
      },
      {
        "title": "Token Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_service_llm_tokens_total[1h])",
            "legendFormat": "{{model}} {{type}}"
          }
        ]
      },
      {
        "title": "Estimated Cost (Hourly)",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(ai_service_llm_cost_dollars[1h])",
            "legendFormat": "Cost"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_service_http_requests_total{status=~\"5..\"}[5m]) / rate(ai_service_http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

---

## Measuring Deployment Quality

Deployment is not a one-time event -- you need to continuously measure whether your deployment is actually working. Here are the key metrics and how to evaluate them.

### Service-Level Objectives (SLOs)

Define measurable targets before deploying:

| Metric | What to Measure | Typical SLO for AI Services | How to Measure |
|--------|----------------|---------------------------|----------------|
| Availability | Uptime percentage | 99.9% (8.7 hours downtime/year) | Health check success rate |
| Latency (p50) | Median response time | < 500ms for API-only, < 2s for inference | Prometheus histogram quantiles |
| Latency (p99) | Tail latency | < 2s for API-only, < 10s for inference | Prometheus histogram quantiles |
| Error rate | 5xx responses / total | < 0.1% | `rate(http_requests_total{status=~"5.."}[5m])` |
| LLM success rate | Successful LLM calls / total | > 99% (provider outages happen) | Custom counter |

### Deployment Verification Checklist

After every deployment, verify:

```bash
# 1. Health check passes
curl -f https://your-api.com/health

# 2. Readiness check passes (all dependencies connected)
curl -f https://your-api.com/health/ready

# 3. Smoke test — send a real request
curl -X POST https://your-api.com/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, deployment test"}'

# 4. Check error rate in first 5 minutes (should be near zero)
# In Grafana or via PromQL:
# rate(http_requests_total{status=~"5.."}[5m])

# 5. Check latency hasn't regressed
# histogram_quantile(0.95, rate(http_latency_seconds_bucket[5m]))
```

### What to Alert On

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High error rate | > 1% 5xx for 5 minutes | Critical | Page on-call, check logs |
| High latency | p95 > 5s for 10 minutes | Warning | Check LLM provider status |
| Pod restarts | > 3 restarts in 10 minutes | Critical | Check OOM kills, health probes |
| LLM cost spike | Hourly cost > 2x average | Warning | Check for request loops, abuse |
| Disk full | > 90% disk usage | Critical | Rotate logs, expand volume |

---

## Cold Start, Rollback, and Load Testing

These three topics are critical for production reliability but are often overlooked in deployment tutorials.

### Cold Start Analysis

Cold start is the time between a new container starting and it being ready to serve traffic. For AI services, this is often the deployment bottleneck:

| Component | Typical Cold Start | Mitigation |
|-----------|-------------------|------------|
| Container pull + OS boot | 5-30 seconds | Use slim base images, pre-pull images to nodes |
| Python interpreter + dependencies | 2-10 seconds | Minimize dependencies, use compiled wheels |
| Database/Redis connections | 1-3 seconds | Connection pooling, lazy initialization |
| ML model loading into RAM | 10-120 seconds | Model caching on persistent volumes, model warmup endpoints |
| ML model loading into GPU VRAM | 30-180 seconds | Keep GPU pods warm (min replicas > 0), use model servers (Triton, vLLM) |

**The health check trap:** If your model takes 60 seconds to load but `initialDelaySeconds` is 10 seconds, Kubernetes will kill and restart your pod in a loop (CrashLoopBackOff). Set `initialDelaySeconds` generously:

```yaml
# For services that load models at startup
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 120  # Give model time to load
  periodSeconds: 30
  failureThreshold: 3
readinessProbe:
  httpGet:
    path: /health/ready   # Only returns 200 after model is loaded
    port: http
  initialDelaySeconds: 10  # Start checking early
  periodSeconds: 5         # Check frequently
  failureThreshold: 24     # 10s + (24 × 5s) = 130s total patience
```

**Cloud Run cold start:** Set `minScale: "1"` (as shown in this blog) to keep at least one instance warm. For GPU workloads, cold starts can exceed 3 minutes — consider always-on instances instead of serverless.

### Rollback Procedures

Rollbacks must be fast, tested, and require no debugging under pressure. Here are three strategies:

**Strategy 1: Kubernetes rollback (fastest, for K8s deployments)**

```bash
#!/bin/bash
# rollback.sh — revert to the previous deployment
# Usage: ./rollback.sh [namespace] [deployment-name]

NAMESPACE="${1:-ai-services}"
DEPLOYMENT="${2:-ai-service}"

echo "=== Current rollout status ==="
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=10s || true

echo ""
echo "=== Rollout history ==="
kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE

echo ""
echo "=== Rolling back to previous revision ==="
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

echo ""
echo "=== Waiting for rollback to complete ==="
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

echo ""
echo "=== Post-rollback verification ==="
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT
echo ""
echo "Verify health: curl -f https://your-api.com/health"
echo "Verify readiness: curl -f https://your-api.com/health/ready"
```

**Strategy 2: Container image pin (safest, for any platform)**

```bash
# Always tag images with git SHA, not just "latest"
# In CI/CD, record the last-known-good image:
echo "$REGISTRY/$IMAGE_NAME:$GITHUB_SHA" > .last-good-image

# To rollback, deploy the previous image:
ROLLBACK_IMAGE=$(cat .last-good-image)
kubectl set image deployment/ai-service ai-service=$ROLLBACK_IMAGE -n ai-services
```

**Strategy 3: Traffic shifting (safest for critical services)**

```yaml
# Azure Container Apps / GCP Cloud Run — shift traffic gradually
# Start: 100% old, 0% new
# After smoke tests pass: 10% new, 90% old
# After 30 minutes with no errors: 50/50
# After 1 hour: 100% new
# If errors at any stage: instant rollback to 100% old
traffic:
  - revisionName: ai-service-v2    # new
    weight: 10
  - revisionName: ai-service-v1    # old (rollback target)
    weight: 90
```

### Load Testing Before Deployment

Never deploy without knowing your capacity. Here is a minimal load test using `locust`:

```python
# load_test.py — run with: locust -f load_test.py --host https://your-api.com
from locust import HttpUser, task, between


class AIServiceUser(HttpUser):
    """Simulates realistic AI service usage patterns."""
    wait_time = between(1, 5)  # 1-5 seconds between requests (realistic user pace)

    @task(3)  # Weight: 3x more chat requests than embeddings
    def chat_request(self):
        self.client.post(
            "/api/v1/chat",
            json={"message": "Summarize the key points of this document", "max_tokens": 200},
            headers={"Authorization": "Bearer test-key"},
            timeout=30,  # LLM calls can be slow
        )

    @task(1)
    def embedding_request(self):
        self.client.post(
            "/api/v1/embeddings",
            json={"text": "Sample text for embedding generation"},
            headers={"Authorization": "Bearer test-key"},
            timeout=10,
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

**Interpreting load test results:**

| Metric | Healthy | Warning | Action Required |
|--------|---------|---------|-----------------|
| p50 latency | < 1s (API) / < 3s (inference) | 1-3s / 3-8s | Check LLM provider latency, consider caching |
| p99 latency | < 5s / < 15s | 5-15s / 15-30s | Add timeout enforcement, check for head-of-line blocking |
| Error rate | < 0.1% | 0.1-1% | Check logs for specific error types |
| Throughput ceiling | Scales linearly with pods | Sub-linear scaling | Check for shared bottleneck (DB connections, rate limits) |
| Memory per pod | Stable | Growing over time | Memory leak — check for accumulating caches or connections |

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Service

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Lint
        run: |
          pip install ruff
          ruff check .

  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to staging
        run: |
          kubectl set image deployment/ai-service \
            ai-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n ai-services

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ai-service -n ai-services --timeout=300s

      - name: Run smoke tests
        run: |
          ./scripts/smoke-tests.sh https://staging-api.yourdomain.com

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to production
        run: |
          kubectl set image deployment/ai-service \
            ai-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n ai-services

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ai-service -n ai-services --timeout=300s

      - name: Notify deployment
        uses: slackapi/slack-github-action@v1.24.0
        with:
          payload: |
            {
              "text": "AI Service deployed to production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*AI Service deployed to production*\nCommit: ${{ github.sha }}\nBy: ${{ github.actor }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Interview Preparation

### Conceptual Questions

**1. "How do you handle scaling for AI services differently than traditional web services?"**

The core difference is that AI services have non-deterministic, IO-bound latency from LLM calls. A standard web API might take 10-50ms per request (CPU-bound), so CPU utilization is a good scaling signal. An AI service calling GPT-4o might take 500ms-15s per request while using almost zero CPU (just waiting for the provider response).

This means:
- **CPU-based HPA fails**: Pods appear idle at 5% CPU while 100 requests are queued waiting for LLM responses. The autoscaler won't add pods because CPU is low.
- **Solution**: Scale on request count or concurrency, not CPU alone. In Kubernetes, use a custom metric like `http_requests_per_second` via Prometheus Adapter. In Cloud Run, use `containerConcurrency` (e.g., 80 for API-calling services, 1-4 for local inference).
- **Queue-based processing for long tasks**: For requests that take >10s (document processing, image generation), use a job queue (Redis + Celery, SQS + Lambda). Return a job ID immediately, let the client poll for results. This prevents HTTP timeout issues and allows independent scaling of the worker pool.
- **Caching layer**: Identical prompts produce identical results (with temperature=0). A Redis cache keyed on `hash(model + prompt + params)` can eliminate 20-40% of LLM calls for many workloads, directly reducing cost and latency.

**2. "What monitoring is essential for AI applications beyond standard web metrics?"**

Standard metrics (HTTP latency, error rate, CPU/memory) are necessary but not sufficient. AI services need three additional monitoring dimensions:

- **Token economics**: Track `input_tokens`, `output_tokens`, and `cost_dollars` per model per endpoint. A bug that causes prompt inflation (e.g., appending entire documents instead of summaries) can 100x your costs in hours. Alert on `hourly_cost > 2x rolling_average`.
- **Provider health**: Track per-provider error rates and latency separately. OpenAI might have a regional outage while Anthropic is fine. The `track_llm_call` decorator in this blog labels metrics by `provider` and `model` so you can build per-provider dashboards.
- **Quality drift**: LLM outputs can degrade silently (model updates, prompt sensitivity). Log a sample of inputs/outputs and run periodic quality checks — even a simple "did the response contain the expected JSON structure?" catch goes a long way.

The Grafana dashboard in this blog covers the first two. Quality drift monitoring requires a separate evaluation pipeline (covered in Blogs 18-19).

**3. "Walk me through how you would achieve zero-downtime deployment for an AI service."**

Zero-downtime deployment has four layers, each solving a different failure mode:

1. **Rolling updates with `maxUnavailable: 0`**: Kubernetes creates new pods before terminating old ones. The `maxSurge: 1` setting means at most one extra pod exists during the transition. For AI services, this is critical because model loading during startup can take 30-120 seconds — you cannot afford to remove pods before replacements are ready.

2. **Health checks (liveness + readiness)**: The liveness probe (`/health`) tells K8s "this container is alive." The readiness probe (`/health/ready`) tells K8s "this container can serve traffic." For AI services, separate these: a container might be alive but not ready because the model is still loading. Only route traffic to pods that pass the readiness check.

3. **Graceful shutdown**: When K8s sends SIGTERM, your app should stop accepting new requests but finish in-flight ones. For LLM calls that take 10-30 seconds, set `terminationGracePeriodSeconds: 60` (default is 30). The `lifespan` handler in the FastAPI code handles this via `cleanup_services()`.

4. **Deployment circuit breaker (ECS) or rollback (K8s)**: If the new version starts crashing, automatically revert. ECS has `DeploymentCircuitBreaker(rollback=True)`. In K8s, use `kubectl rollout undo` (scripted in the rollback section of this blog).

**Follow-up trap**: "What about blue-green deployments?" Answer: Blue-green requires running two full environments simultaneously, which doubles GPU costs for AI workloads. For most AI services, rolling updates with proper health checks are sufficient. Blue-green is justified when rollback speed is critical (financial services) or when you need to run A/B tests on different model versions.

**4. "Design a deployment architecture for an AI chatbot handling 10K requests/day with a 2-second SLA and $500/month budget."**

First, back-of-envelope math:
- 10K requests/day = ~7 requests/minute average, but with bursty peaks of 50-100 RPM
- At 2s SLA, each request blocks a "slot" for 2 seconds
- Need: 100 RPM × (2s / 60s) = ~3.3 concurrent slots at peak → 2 pods minimum

Architecture:
- **Platform**: Cloud Run (simplest for this scale, auto-scales to zero during off-hours)
- **Compute**: 2 vCPU, 4GB RAM per instance, `minScale: 1`, `maxScale: 5`
- **Cost estimate**: 1 instance warm 24/7 ≈ $50/month compute + ~$300/month LLM API costs (10K requests × 1K tokens avg × $0.03/1K for GPT-4o-mini) = ~$350/month total, within budget
- **Monitoring**: Prometheus metrics + Cloud Run built-in monitoring
- **Caching**: Redis on Cloud Memorystore ($30/month) for repeated queries, expected 25% cache hit rate → saves ~$75/month in LLM costs

If the SLA were 200ms instead of 2s, you'd need local model inference (GPU) which would blow the budget — at that point, either increase budget to $1500/month for a T4 instance or use a faster model (GPT-4o-mini has ~300ms p50 which might work).

**5. "What are the security considerations specific to deploying AI services?"**

AI services have unique security concerns beyond standard web security:

- **Prompt injection via API**: Users can craft inputs that manipulate the LLM into revealing system prompts, ignoring instructions, or generating harmful content. Defense: input validation (length limits, character filtering), output filtering (check for system prompt leakage), and system prompt hardening.
- **API key exposure**: AI API keys (OpenAI, Anthropic) are high-value targets — they grant direct billing access. Never put keys in environment variables in Docker images, Dockerfiles, or git repos. Use cloud-native secrets managers (AWS Secrets Manager, GCP Secret Manager, K8s Secrets with encryption at rest).
- **Cost-based denial of service**: An attacker can send large prompts with `max_tokens=4096` to run up your bill. Defense: rate limiting per API key (shown in the Ingress annotation: `rate-limit: "100"`), per-user token budgets, and cost alerting.
- **CORS misconfiguration**: `allow_origins=["*"]` on an authenticated service lets any website make authenticated requests on behalf of your users. The code in this blog uses environment-configured origins for this reason.
- **Data exfiltration via LLM**: If your AI service has access to a knowledge base (RAG), prompt injection could extract sensitive documents. Defense: access control on the retrieval layer, not just the API layer.

### Coding Challenges

**Challenge 1: Deployment Architecture Designer**

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import math


@dataclass
class DeploymentRecommendation:
    platform: str
    compute_config: Dict
    scaling_config: Dict
    estimated_monthly_cost: float
    cost_breakdown: Dict[str, float]
    warnings: List[str]
    gpu_config: Optional[Dict]


def design_deployment_architecture(
    expected_rps: float,
    latency_sla_ms: int,
    budget_monthly: float,
    gpu_required: bool = False,
    avg_tokens_per_request: int = 500,
    llm_model: str = "gpt-4o-mini",
) -> DeploymentRecommendation:
    """
    Design deployment architecture based on requirements.

    This function demonstrates the systematic decision-making process
    a platform engineer uses when designing AI service infrastructure.
    """
    warnings = []

    # --- Step 1: Estimate LLM API costs ---
    LLM_PRICING = {  # (input $/1K tokens, output $/1K tokens)
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "claude-3-5-sonnet": (0.003, 0.015),
    }
    input_rate, output_rate = LLM_PRICING.get(llm_model, (0.001, 0.002))
    # Assume 60/40 input/output token split
    input_tokens = avg_tokens_per_request * 0.6
    output_tokens = avg_tokens_per_request * 0.4
    cost_per_request = (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
    requests_per_month = expected_rps * 86400 * 30
    llm_monthly_cost = cost_per_request * requests_per_month

    # --- Step 2: Determine compute needs ---
    # Each request "holds" a connection for latency_sla_ms
    concurrent_connections = math.ceil(expected_rps * (latency_sla_ms / 1000))
    # Each pod handles ~50 concurrent IO-bound connections (API-calling)
    # or ~2 concurrent connections (local GPU inference)
    connections_per_pod = 2 if gpu_required else 50
    min_pods = max(2, math.ceil(concurrent_connections / connections_per_pod))
    # Add 50% headroom for traffic bursts
    recommended_pods = math.ceil(min_pods * 1.5)

    # --- Step 3: Choose platform ---
    if gpu_required:
        if budget_monthly < 500:
            warnings.append(
                "GPU workloads typically cost $500+/month. "
                "Consider using API-based inference instead."
            )
        platform = "kubernetes"  # GPU scheduling needs K8s
        compute_cost = recommended_pods * 720 * 0.53  # T4 hourly rate × hours/month
        gpu_config = {
            "gpu_type": "nvidia-tesla-t4",
            "gpu_per_pod": 1,
            "vram_gb": 16,
            "node_selector": {"accelerator": "nvidia-tesla-t4"},
        }
    elif expected_rps < 1:
        platform = "cloud-run"
        compute_cost = 50  # Minimal with scale-to-zero
        gpu_config = None
    elif expected_rps < 50:
        platform = "cloud-run"
        compute_cost = recommended_pods * 50  # ~$50/pod/month on Cloud Run
        gpu_config = None
    else:
        platform = "kubernetes"
        compute_cost = recommended_pods * 70  # ~$70/pod/month on GKE/EKS
        gpu_config = None

    # --- Step 4: Check budget ---
    total_cost = llm_monthly_cost + compute_cost
    if total_cost > budget_monthly:
        overage_pct = ((total_cost - budget_monthly) / budget_monthly) * 100
        warnings.append(
            f"Estimated cost ${total_cost:.0f}/month exceeds budget "
            f"${budget_monthly:.0f}/month by {overage_pct:.0f}%. "
            f"Consider: caching (saves 20-40% LLM costs), "
            f"cheaper model, or reducing traffic."
        )

    # --- Step 5: Assemble recommendation ---
    return DeploymentRecommendation(
        platform=platform,
        compute_config={
            "pods": recommended_pods,
            "cpu_per_pod": "2" if not gpu_required else "4",
            "memory_per_pod": "4Gi" if not gpu_required else "16Gi",
            "concurrent_connections": concurrent_connections,
        },
        scaling_config={
            "min_replicas": max(1, min_pods),
            "max_replicas": recommended_pods * 3,
            "scale_metric": "concurrency" if platform == "cloud-run" else "http_requests_per_second",
            "scale_up_window_seconds": 60,
            "scale_down_window_seconds": 300,
        },
        estimated_monthly_cost=total_cost,
        cost_breakdown={
            "llm_api": llm_monthly_cost,
            "compute": compute_cost,
            "monitoring": 20,  # Prometheus/Grafana
        },
        warnings=warnings,
        gpu_config=gpu_config,
    )


# --- Example usage ---
if __name__ == "__main__":
    # Scenario: moderate chatbot
    rec = design_deployment_architecture(
        expected_rps=7 / 60,  # 10K requests/day ≈ 7/minute ≈ 0.12 RPS
        latency_sla_ms=2000,
        budget_monthly=500,
        gpu_required=False,
        llm_model="gpt-4o-mini",
    )
    print(f"Platform: {rec.platform}")
    print(f"Pods: {rec.compute_config['pods']}")
    print(f"Scaling: {rec.scaling_config['min_replicas']}-{rec.scaling_config['max_replicas']} replicas")
    print(f"Monthly cost: ${rec.estimated_monthly_cost:.0f}")
    print(f"  LLM API: ${rec.cost_breakdown['llm_api']:.0f}")
    print(f"  Compute: ${rec.cost_breakdown['compute']:.0f}")
    for w in rec.warnings:
        print(f"  ⚠ {w}")
```

**Challenge 2: Deployment Health Monitor**

```python
import time
import asyncio
import httpx
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HealthReport:
    timestamp: str
    overall_status: str  # "healthy", "degraded", "down"
    checks: Dict[str, Dict]
    recommendations: List[str]


@dataclass
class DeploymentMonitor:
    """
    Monitors a deployed AI service and produces actionable health reports.

    This goes beyond simple health checks by verifying:
    1. HTTP connectivity (is the service reachable?)
    2. Functional correctness (does the AI endpoint actually work?)
    3. Latency compliance (is it within SLA?)
    4. Dependency health (are all backends connected?)
    """
    base_url: str
    latency_sla_ms: int = 2000
    timeout_seconds: int = 30
    auth_token: str = ""
    _latency_history: List[float] = field(default_factory=list)

    async def run_full_check(self) -> HealthReport:
        """Run all health checks and produce a report."""
        checks = {}
        recommendations = []

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Check 1: Basic health
            checks["health"] = await self._check_endpoint(
                client, "GET", "/health", expected_status=200
            )

            # Check 2: Readiness (dependency health)
            checks["readiness"] = await self._check_endpoint(
                client, "GET", "/health/ready", expected_status=200
            )

            # Check 3: Functional smoke test
            checks["functional"] = await self._check_endpoint(
                client, "POST", "/api/v1/chat",
                json={"message": "Respond with exactly: OK", "max_tokens": 10},
                expected_status=200,
            )

            # Check 4: Latency compliance
            if checks["functional"].get("latency_ms"):
                latency = checks["functional"]["latency_ms"]
                self._latency_history.append(latency)
                checks["latency_sla"] = {
                    "status": "pass" if latency <= self.latency_sla_ms else "fail",
                    "latency_ms": latency,
                    "sla_ms": self.latency_sla_ms,
                    "recent_p50": self._percentile(50),
                    "recent_p99": self._percentile(99),
                }

        # Generate recommendations
        if checks["health"].get("status") != "pass":
            recommendations.append("Service is DOWN. Check pod logs: kubectl logs -l app=ai-service -n ai-services")
        if checks["readiness"].get("status") != "pass":
            recommendations.append("Dependencies unhealthy. Check: database, redis, vector store connectivity")
        if checks.get("latency_sla", {}).get("status") == "fail":
            recommendations.append(
                f"Latency {checks['latency_sla']['latency_ms']:.0f}ms exceeds SLA {self.latency_sla_ms}ms. "
                "Check: LLM provider status, pod resource limits, network latency"
            )
        if checks["functional"].get("status") != "pass":
            recommendations.append("Functional check failed. The AI endpoint is not returning valid responses.")

        # Determine overall status
        statuses = [c.get("status", "unknown") for c in checks.values()]
        if all(s == "pass" for s in statuses):
            overall = "healthy"
        elif checks["health"].get("status") == "pass":
            overall = "degraded"
        else:
            overall = "down"

        return HealthReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_status=overall,
            checks=checks,
            recommendations=recommendations,
        )

    async def _check_endpoint(
        self, client: httpx.AsyncClient, method: str, path: str,
        expected_status: int = 200, **kwargs
    ) -> Dict:
        """Check a single endpoint and return structured results."""
        url = f"{self.base_url}{path}"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            start = time.time()
            response = await client.request(method, url, headers=headers, **kwargs)
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "pass" if response.status_code == expected_status else "fail",
                "http_status": response.status_code,
                "latency_ms": latency_ms,
                "body_preview": str(response.json())[:200] if response.status_code == 200 else None,
            }
        except httpx.TimeoutException:
            return {"status": "fail", "error": f"Timeout after {self.timeout_seconds}s"}
        except httpx.ConnectError as e:
            return {"status": "fail", "error": f"Connection failed: {e}"}

    def _percentile(self, pct: int) -> float:
        """Calculate percentile from recent latency history."""
        if not self._latency_history:
            return 0.0
        sorted_latencies = sorted(self._latency_history[-100:])  # Last 100 measurements
        idx = int(len(sorted_latencies) * pct / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


# --- Example usage ---
async def post_deployment_verification():
    """Run after every deployment to verify the service is healthy."""
    monitor = DeploymentMonitor(
        base_url="https://your-api.com",
        latency_sla_ms=2000,
        auth_token="your-test-token",
    )

    print("Running post-deployment verification...")
    report = await monitor.run_full_check()

    print(f"Overall status: {report.overall_status}")
    for name, check in report.checks.items():
        status = check.get('status', 'unknown')
        print(f"  {name}: {status}")
        if check.get('latency_ms'):
            print(f"    latency: {check['latency_ms']:.0f}ms")

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    # Exit with non-zero code if unhealthy (for CI/CD integration)
    if report.overall_status == "down":
        print("\nDEPLOYMENT FAILED — triggering rollback")
        return 1
    elif report.overall_status == "degraded":
        print("\nDEPLOYMENT DEGRADED — manual review recommended")
        return 0  # Don't auto-rollback on degraded
    else:
        print("\nDEPLOYMENT VERIFIED — all checks passed")
        return 0
```

---

### Career Mapping

| Role | What They Use from This Blog | Interview Focus |
|------|------------------------------|-----------------|
| **Platform / Infrastructure Engineer** | K8s configs, Helm charts, CI/CD pipelines, GPU scheduling, HPA tuning | "Design a K8s deployment for a service with 60s cold starts" |
| **MLOps Engineer** | Model serving, GPU config, monitoring, CI/CD for ML, rollback procedures | "How do you deploy a new model version without downtime?" |
| **Backend / API Engineer** | FastAPI structure, Docker, monitoring middleware, retry logic, CORS | "How would you add rate limiting and cost tracking to this API?" |
| **SRE / DevOps** | SLOs, alerting, load testing, rollback procedures, cold start mitigation | "Your p99 latency jumped from 2s to 15s after a deploy — walk me through debugging" |
| **AI Engineering Manager** | Cost tables, platform comparison, budget estimation, deployment trade-offs | "Given a $500/month budget, what deployment architecture would you recommend?" |

---

## Exercises

### Exercise 1: Complete Docker Setup
Create a complete Docker deployment including:
- Multi-stage build
- Health checks
- Non-root user
- Optimized image size

### Exercise 2: Kubernetes Deployment
Deploy to Kubernetes with:
- Horizontal Pod Autoscaler
- Resource limits
- Secrets management
- Ingress configuration

### Exercise 3: Monitoring Dashboard
Build a monitoring solution with:
- Prometheus metrics
- Grafana dashboards
- Alerting rules
- Log aggregation

### Exercise 4: CI/CD Pipeline
Create a complete pipeline with:
- Automated testing
- Container building
- Staged deployments
- Rollback capability

---

## Summary

### Key Takeaways

1. **Containerization is fundamental**: Docker ensures consistency across environments
2. **Choose infrastructure wisely**: Match to workload patterns and budget
3. **Monitoring is essential**: You cannot fix what you cannot see
4. **Automate deployments**: CI/CD reduces errors and speeds delivery
5. **Plan for scale**: Design for growth from the start
6. **GPU decisions matter**: Only pay for GPUs when your workload demands them

### Deployment Checklist

- [ ] Docker image optimized and secure
- [ ] CORS configured to specific origins (not wildcard)
- [ ] Health checks configured
- [ ] Secrets properly managed (no keys in code or environment files)
- [ ] Monitoring and alerting in place
- [ ] Auto-scaling configured
- [ ] CI/CD pipeline working
- [ ] Rollback procedure tested
- [ ] GPU resources configured (if needed)
- [ ] Documentation updated

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|-----------------|------------|------------------|
| **Docker** | Multi-stage builds, optimized layers, security best practices | Working Dockerfile with health checks | Can write a basic Dockerfile |
| **Cloud deployment** | Deployed to at least one cloud provider with auto-scaling | Understands cloud deployment configs | Only runs locally |
| **Kubernetes** | Can write deployments, services, HPAs, and debug pod issues | Can deploy with provided YAML | No K8s experience |
| **Monitoring** | Prometheus + Grafana + alerting + LLM-specific metrics | Basic health checks and logging | No monitoring |
| **CI/CD** | Complete pipeline with staging and production stages | Partial automation (build + test) | Manual deployment only |

### What This Blog Does Well
- Provides complete, copy-pasteable deployment configurations for three major cloud providers
- Covers the full deployment stack: Docker, cloud, Kubernetes, monitoring, CI/CD
- Includes AI-specific monitoring (token tracking, LLM cost estimation)
- Shows production patterns (non-root users, health checks, secrets management)

### Where This Blog Falls Short
- All code is configuration-heavy and untested -- you cannot run these examples without real cloud accounts and infrastructure
- The monitoring decorator (`track_llm_call`) is simplified and does not handle streaming responses, retries, or fallback models
- No coverage of canary deployments, feature flags, or A/B testing infrastructure
- The cost estimates are approximations that will be outdated -- always check current provider pricing
- No discussion of compliance (HIPAA, GDPR, SOC 2) which matters for many production AI deployments
- GPU section covers basics but does not address multi-GPU inference, model sharding, or quantization for deployment

---

### Deployment Sanity Checks

**1. Would you trust someone who learned only this blog to touch a production AI system?**

**YES, with caveats.** This blog covers the full deployment stack: containerization, cloud deployment (3 providers), Kubernetes with production-grade configs, monitoring with AI-specific metrics, CI/CD with staging gates, retry logic with cost caps, rollback procedures, and load testing. The reader understands *why* AI deployments differ from standard web deployments (non-deterministic latency, cold start, provider fragility). The coding challenges produce working deployment tools, not toy demos. **Caveat**: the reader still needs hands-on practice with a real cloud account — configs alone don't build operational intuition. They should also study the security hardening topics listed in "What This Blog Does NOT Cover" before handling sensitive data.

**2. Can you explain at least one real failure case using only what's taught here?**

**YES.** Example: "After deploying a new version, p99 latency jumped from 2s to 15s and the error rate spiked to 5%." Using this blog's knowledge:
- **Detection**: The Grafana dashboard shows the latency spike via `histogram_quantile(0.95, rate(ai_service_llm_latency_seconds_bucket[5m]))`. The alerting rule fires because p95 > 5s for 10 minutes.
- **Diagnosis**: Check if the new version changed the LLM model (cost/latency change), if the new pods are failing readiness checks (cold start too long for `initialDelaySeconds`), or if the HPA hasn't scaled up yet (check `kubectl get hpa`).
- **Mitigation**: Run `./rollback.sh ai-services ai-service` to revert to the previous deployment. The rollback script verifies health post-revert.
- **Root cause investigation**: After rollback stabilizes the service, check the new version's logs for model loading errors, compare the Docker image size (did dependencies bloat?), and run the load test against the new version in staging.

**3. Would this blog survive senior-engineer interview follow-up questions?**

**YES.** The interview section provides mechanism-level answers to 5 questions, including: why CPU-based autoscaling fails for AI services (IO-bound LLM calls), how to achieve zero-downtime deployment with model loading delays (4-layer approach), security considerations unique to AI services (prompt injection, cost-based DoS), and a worked architecture design with cost math. The coding challenges produce a deployment architecture designer and a deployment health monitor — both demonstrate systematic engineering thinking, not just config knowledge.

---

## What's Next?

In **Blog 25: Career in Generative AI**, we'll explore how to build your career in this field. You'll learn:
- In-demand skills and roles
- Building a portfolio
- Interview preparation
- Continuous learning strategies
- Industry trends and opportunities

The final step -- turning your knowledge into a career!

---

*Deployment is where your code meets reality. Build robust systems, monitor relentlessly, and always have a rollback plan.*

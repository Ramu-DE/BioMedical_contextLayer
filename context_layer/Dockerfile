# Context layer on Lambda MicroVM. Mirrors module-5-langgraph-agent.
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
      "neo4j>=5.28,<6" \
      "qdrant-client>=1.19,<2" \
      "boto3>=1.35,<2" \
      "rdflib>=7.1,<8" \
      "pyshacl>=0.26,<0.30" \
      "PyYAML>=6.0.2,<7" \
      "pydantic>=2.9,<3" \
      "langgraph>=1.2.11,<2" \
      "langchain-aws>=1.7.3,<2"

COPY config.py ./
COPY src/ ./src/
COPY ontology/ ./ontology/

ENV PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    APP_PORT=8080 \
    HOOKS_PORT=9000

EXPOSE 8080 9000

# Credentials arrive from the MicroVM execution role and environment; no secrets
# are baked into the image. .env is deliberately NOT copied.
# PYTHONPATH above is load-bearing: `python -m` resolves the module before
# server.py can extend sys.path.
CMD ["python", "-m", "context_layer.agent.server"]

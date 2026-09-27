FROM python:3.12-slim

RUN pip install --no-cache-dir \
    httpx==0.28.1 \
    jsonschema==4.25.1 \
    pytest==8.4.2 \
    ruff==0.13.3

COPY docker/validate_candidate.py /opt/parallax/validate_candidate.py
COPY tests /opt/parallax/tests
RUN mkdir -p /candidate /work
USER 65534:65534
ENTRYPOINT ["python", "/opt/parallax/validate_candidate.py"]

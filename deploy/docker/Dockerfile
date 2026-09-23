# ASTRA control-plane image (exporter / validate / plan).
# nvidia-smi is injected at runtime by the NVIDIA Container Toolkit
# (NVIDIA_DRIVER_CAPABILITIES=utility); the image itself carries no driver.

FROM python:3.11-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps -w /wheels .

FROM python:3.11-slim
LABEL org.opencontainers.image.title="astra-control" \
      org.opencontainers.image.description="ASTRA external GPU node control plane"
RUN useradd --system --uid 10001 --no-create-home astra
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY config/astra.example.toml /etc/astra/astra.toml

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility \
    PYTHONUNBUFFERED=1
USER astra
EXPOSE 9835
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9835/healthz', timeout=4)"
ENTRYPOINT ["astra"]
CMD ["exporter"]

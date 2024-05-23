ARG PY_VERSION=3.11

FROM python:$PY_VERSION as base
RUN apt-get update && apt-get install -y libvips
# create non-root user (primarily for devcontainer)
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode

WORKDIR /app
COPY . /app/
RUN chown -R vscode /app

FROM base AS hatch
RUN pip3 install hatch
ENV HATCH_ENV=default
ENTRYPOINT ["hatch", "run"]

FROM base AS dev
RUN pip3 install hatch ipykernel
USER vscode
RUN find requirements -name 'requirement*.txt' | while read requirement; do \
        pip3 install -r "$requirement"; \
    done
RUN pip3 install -r requirements.txt
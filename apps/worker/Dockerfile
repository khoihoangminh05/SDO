FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment for build execution
SHELL ["conda", "run", "-n", "worker-env", "/bin/bash", "-c"]

COPY . .

EXPOSE 8000

# Run uvicorn inside activated environment with no capture output for proper logging
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker-env", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

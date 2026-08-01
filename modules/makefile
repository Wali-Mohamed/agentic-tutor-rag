# Makefile

.PHONY: setup ingest run up down

setup:
	@echo "Installing dependencies via uv..."
	uv sync

ingest:
	@echo "Running dlt ingestion pipeline into DuckDB..."
	uv run python modules/ingest_pipeline.py

run:
	@echo "Running Streamlit app locally..."
	uv run streamlit run app.py

up:
	@echo "Starting the app via Docker Compose..."
	docker compose up --build -d

down:
	@echo "Stopping Docker containers..."
	docker compose down
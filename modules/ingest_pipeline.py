import json
import dlt

@dlt.resource(write_disposition="replace")
def faq_source():
    # Simple relative path: Go up one folder (..), then into data/
    with open("data/knowledge-base.json", "r") as f:
        yield json.load(f)

if __name__ == "__main__":
    print("🚀 Starting dlt ingestion pipeline...")
    
    pipeline = dlt.pipeline(
        pipeline_name="tutor_pipeline",
        # Tell dlt we want duckdb, and pass the exact relative path
        destination=dlt.destinations.duckdb("data/tutor_pipeline.duckdb"),
        dataset_name="wali_kb"
    )
    
    load_info = pipeline.run(faq_source())
    
    print(load_info)
    print("✅ Ingestion complete! Data successfully loaded into data/tutor_pipeline.duckdb")
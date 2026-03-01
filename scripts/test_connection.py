"""
Run this FIRST to verify VectorAI DB is alive.
Usage: python scripts/test_connection.py
"""
from cortex import CortexClient, DistanceMetric

def main():
    print("Connecting to VectorAI DB at localhost:50051...")
    try:
        with CortexClient("localhost:50051") as client:
            version, uptime = client.health_check()
            print(f"✅ Connected! Version: {version}, Uptime: {uptime}")

            # Quick smoke test
            test_name = "_smoke_test"
            client.create_collection(test_name, dimension=4, distance_metric=DistanceMetric.COSINE)
            client.upsert(test_name, id=0, vector=[1.0, 0.0, 0.0, 0.0], payload={"test": True})
            results = client.search(test_name, query=[1.0, 0.0, 0.0, 0.0], top_k=1)
            assert len(results) == 1
            assert results[0].id == 0
            print(f"✅ CRUD works! Search returned score: {results[0].score}")
            client.delete_collection(test_name)
            print("✅ Cleanup done. VectorAI DB is ready!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Docker running? Run: docker ps")
        print("  2. Is the container up? Run: docker compose up -d")
        print("  3. Port conflict? Check: lsof -i :50051")
        raise

if __name__ == "__main__":
    main()

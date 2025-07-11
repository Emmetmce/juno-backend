import requests
import json

# Test your API endpoints
base_url = "https://juno-backend-7lj5.onrender.com"

def test_debug_endpoint():
    """Test the debug endpoint to see what's in the database"""
    try:
        response = requests.get(f"{base_url}/debug/chunks")
        print("=== DEBUG CHUNKS RESPONSE ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error testing debug endpoint: {e}")
        return None

def test_ping():
    """Test if API is alive"""
    try:
        response = requests.get(f"{base_url}/ping")
        print("=== PING RESPONSE ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error testing ping: {e}")
        return None

def test_simple_query():
    """Test a simple query"""
    try:
        response = requests.post(
            f"{base_url}/query",
            headers={"Content-Type": "application/json"},
            json={"query": "podcast"}
        )
        print("=== SIMPLE QUERY RESPONSE ===")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error testing simple query: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Testing Juno API...")
    
    # Test 1: Check if API is alive
    ping_result = test_ping()
    
    # Test 2: Check database content
    debug_result = test_debug_endpoint()
    
    # Test 3: Simple query
    query_result = test_simple_query()
    
    print("\n📊 SUMMARY:")
    print(f"API Status: {'✅ OK' if ping_result else '❌ FAILED'}")
    print(f"Database Access: {'✅ OK' if debug_result else '❌ FAILED'}")
    print(f"Query Function: {'✅ OK' if query_result else '❌ FAILED'}")
    
    if debug_result:
        total_chunks = debug_result.get('total_chunks', 0)
        print(f"Total chunks in database: {total_chunks}")
        if total_chunks == 0:
            print("⚠️  Database is empty - you need to run embedder.py")
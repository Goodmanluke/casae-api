#!/usr/bin/env python3
"""
Test script for RentCast AVM accuracy improvements
Run this to verify the enhanced AVM functionality is working correctly.
"""

import asyncio
import os
import sys
import httpx
from datetime import datetime

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_avm_improvements():
    """Test the enhanced AVM functionality"""
    
    # Test configuration
    BASE_URL = "http://localhost:8000"
    TEST_ADDRESS = "123 Main St, Austin, TX 78701"  # Change to a real address for testing
    
    print("🏠 Testing RentCast AVM Accuracy Improvements")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Test Address: {TEST_ADDRESS}")
    print(f"Timestamp: {datetime.now()}")
    print()
    
    async with httpx.AsyncClient(timeout=30) as client:
        
        # Test 1: Debug AVM endpoint with enhanced parameters
        print("📊 Test 1: Debug AVM with Enhanced Parameters")
        try:
            response = await client.get(f"{BASE_URL}/debug/avm", params={
                "address": TEST_ADDRESS,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "squareFootage": 2100,
                "propertyType": "Single Family"
            })
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Request params sent: {data.get('request_params', {})}")
                if 'response' in data and isinstance(data['response'], dict):
                    price = data['response'].get('price')
                    comps = data['response'].get('comparables', [])
                    print(f"AVM Estimate: ${price:,.0f}" if price else "No price found")
                    print(f"Comparables Count: {len(comps)}")
                else:
                    print("Raw response:", data.get('response', 'No response data'))
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
        
        print()
        
        # Test 2: Enhanced rent estimate  
        print("🏘️ Test 2: Enhanced Rent Estimate")
        try:
            response = await client.get(f"{BASE_URL}/rent/estimate", params={
                "address": TEST_ADDRESS,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "squareFootage": 2100,
                "propertyType": "Single Family"
            })
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                rent = data.get('monthly_rent')
                print(f"Monthly Rent Estimate: ${rent:,.0f}/month" if rent else "No rent estimate")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
            
        print()
        
        # Test 3: CMA Baseline with AVM enabled
        print("🏡 Test 3: CMA Baseline with Enhanced AVM")
        try:
            response = await client.post(f"{BASE_URL}/cma/baseline", json={
                "subject": {
                    "address": TEST_ADDRESS,
                    "property_type": "Single Family",
                    "beds": 3,
                    "baths": 2.5,
                    "sqft": 2100,
                    "year_built": 2010
                }
            })
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                estimate = data.get('estimate')
                comps_count = len(data.get('comps', []))
                print(f"CMA Estimate: ${estimate:,.0f}" if estimate else "No estimate")
                print(f"Comparables Used: {comps_count}")
                print(f"Explanation: {data.get('explanation', 'No explanation')[:100]}...")
            else:
                print(f"Error: {response.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")
            
        print()
        
        # Test 4: Property details lookup
        print("🔍 Test 4: Property Details Lookup")
        try:
            response = await client.get(f"{BASE_URL}/debug/rentcast", params={
                "address": TEST_ADDRESS
            })
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                details = data.get('details', {})
                if details:
                    print(f"Property Details Found:")
                    print(f"  Beds: {details.get('beds')}")
                    print(f"  Baths: {details.get('baths')}")
                    print(f"  Sqft: {details.get('sqft')}")
                    print(f"  Year Built: {details.get('year_built')}")
                    print(f"  Coordinates: {details.get('lat')}, {details.get('lng')}")
                else:
                    print("No property details found")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    print()
    print("✅ Testing Complete!")
    print()
    print("💡 Tips for Better Accuracy:")
    print("- Ensure RENTCAST_API_KEY is set in your environment")
    print("- Set RENTCAST_USE_AVM=1 to enable AVM in CMA baseline")
    print("- Use real addresses for testing")
    print("- Check server logs for detailed AVM request/response info")
    print("- Adjust RentCastConfig parameters in main.py if needed")

if __name__ == "__main__":
    # Check if server is likely running
    print("🚀 Starting AVM Improvement Tests...")
    print()
    
    # Check environment
    if os.getenv("RENTCAST_API_KEY"):
        print("✅ RENTCAST_API_KEY found in environment")
    else:
        print("⚠️  RENTCAST_API_KEY not found - some tests may fail")
    
    if os.getenv("RENTCAST_USE_AVM") == "1":
        print("✅ RENTCAST_USE_AVM enabled")
    else:
        print("ℹ️  RENTCAST_USE_AVM not enabled (CMA baseline won't use AVM)")
    
    print()
    
    # Run tests
    asyncio.run(test_avm_improvements())

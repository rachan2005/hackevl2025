#!/usr/bin/env python3
"""
Test script to verify the integration between Behavioral Analyzer and ADK Voice Agent
"""

import asyncio
import aiohttp
import time
import sys
from pathlib import Path

async def test_behavioral_analyzer_api():
    """Test the Behavioral Analyzer API endpoints"""
    print("🧪 Testing Behavioral Analyzer API...")
    
    base_url = "http://localhost:8083"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            print("  Testing health endpoint...")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Health check passed: {data.get('service', 'Unknown')}")
                else:
                    print(f"  ❌ Health check failed: {response.status}")
                    return False
            
            # Test behavioral data endpoint
            print("  Testing behavioral data endpoint...")
            async with session.get(f"{base_url}/api/behavioral-data") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        print("  ✅ Behavioral data endpoint working")
                        print(f"  📊 Current data: {data.get('data', {})}")
                    else:
                        print(f"  ⚠️ Behavioral data endpoint returned error: {data.get('error')}")
                else:
                    print(f"  ❌ Behavioral data endpoint failed: {response.status}")
                    return False
            
            # Test individual endpoints
            endpoints = [
                ("/api/emotion", "emotion"),
                ("/api/attention", "attention"),
                ("/api/fatigue", "fatigue"),
                ("/api/sentiment", "sentiment"),
                ("/api/person-tracking", "person tracking")
            ]
            
            for endpoint, name in endpoints:
                print(f"  Testing {name} endpoint...")
                async with session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            print(f"  ✅ {name} endpoint working")
                        else:
                            print(f"  ⚠️ {name} endpoint returned error: {data.get('error')}")
                    else:
                        print(f"  ❌ {name} endpoint failed: {response.status}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Error testing Behavioral Analyzer API: {e}")
        return False

async def test_adk_voice_agent_api():
    """Test the ADK Voice Agent API endpoints"""
    print("🧪 Testing ADK Voice Agent API...")
    
    base_url = "http://localhost:8082"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            print("  Testing health endpoint...")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Health check passed: {data.get('service', 'Unknown')}")
                else:
                    print(f"  ❌ Health check failed: {response.status}")
                    return False
            
            return True
            
    except Exception as e:
        print(f"  ❌ Error testing ADK Voice Agent API: {e}")
        return False

async def test_real_extractor():
    """Test the real extractor connection"""
    print("🧪 Testing Real Extractor...")
    
    try:
        # Import the real extractor
        sys.path.append(str(Path(__file__).parent / "server"))
        from core.behavioral_extractors.real_extractor import RealBehavioralExtractor
        
        # Create extractor instance
        extractor = RealBehavioralExtractor("http://localhost:8083")
        
        # Initialize
        print("  Initializing real extractor...")
        success = await extractor.initialize()
        if success:
            print("  ✅ Real extractor initialized successfully")
            
            # Test feature extraction
            print("  Testing feature extraction...")
            features = await extractor.extract_features_for_timestamp(time.time())
            print(f"  ✅ Extracted {len(features)} features")
            
            # Cleanup
            await extractor.cleanup()
            print("  ✅ Real extractor cleaned up")
            return True
        else:
            print("  ❌ Real extractor initialization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing real extractor: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("🎯 ADK Voice Agent + Behavioral Analyzer Integration Test")
    print("=" * 60)
    
    # Test Behavioral Analyzer API
    behavioral_ok = await test_behavioral_analyzer_api()
    print()
    
    # Test ADK Voice Agent API
    adk_ok = await test_adk_voice_agent_api()
    print()
    
    # Test Real Extractor
    extractor_ok = await test_real_extractor()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary:")
    print(f"  Behavioral Analyzer API: {'✅ PASS' if behavioral_ok else '❌ FAIL'}")
    print(f"  ADK Voice Agent API: {'✅ PASS' if adk_ok else '❌ FAIL'}")
    print(f"  Real Extractor: {'✅ PASS' if extractor_ok else '❌ FAIL'}")
    
    if behavioral_ok and adk_ok and extractor_ok:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


import requests
import sys
import time
import json
from datetime import datetime

class SolanaTradingBotTester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.wallet_address = "9ZNTfG4NyQgxy2SWjSiQoUyBPEvXT2xo7fKc5hPYYJ7b"  # Example Solana wallet

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, params=params, timeout=10)
            
            print(f"URL: {url}")
            print(f"Status Code: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.text}")
                except:
                    pass
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_api_root(self):
        """Test API root endpoint"""
        return self.run_test(
            "API Root",
            "GET",
            "",
            200
        )

    def test_dashboard_stats(self):
        """Test dashboard stats endpoint"""
        return self.run_test(
            "Dashboard Stats",
            "GET",
            "dashboard/stats",
            200
        )

    def test_pools_data(self):
        """Test pools data endpoint"""
        return self.run_test(
            "Pools Data",
            "GET",
            "pools/data",
            200
        )

    def test_wallet_connect(self):
        """Test wallet connection endpoint"""
        return self.run_test(
            "Wallet Connect",
            "POST",
            "wallet/connect",
            200,
            data={"wallet_address": self.wallet_address}
        )

    def test_wallet_balance(self):
        """Test wallet balance endpoint"""
        return self.run_test(
            "Wallet Balance",
            "GET",
            f"wallet/{self.wallet_address}/balance",
            200
        )

    def test_ai_analysis(self, pool_address):
        """Test AI analysis endpoint"""
        return self.run_test(
            "AI Analysis",
            "GET",
            f"ai/analysis/{pool_address}",
            200
        )

    def test_mev_opportunities(self):
        """Test MEV opportunities endpoint"""
        return self.run_test(
            "MEV Opportunities",
            "GET",
            "mev/opportunities",
            200
        )

    def test_swap_quote(self):
        """Test swap quote endpoint"""
        return self.run_test(
            "Swap Quote",
            "POST",
            "swap/quote",
            200,
            params={
                "input_mint": "So11111111111111111111111111111111111111112",  # SOL
                "output_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "amount": 0.1,
                "slippage_bps": 50
            }
        )

def main():
    # Get backend URL from environment variable
    backend_url = "https://875d5089-b758-4823-8c4c-497fda84d97a.preview.emergentagent.com"
    
    print(f"🚀 Testing Solana DeFi Trading Bot API at {backend_url}")
    tester = SolanaTradingBotTester(backend_url)
    
    # Test API root
    api_root_success, _ = tester.test_api_root()
    if not api_root_success:
        print("❌ API root test failed, stopping tests")
        return 1
    
    # Test dashboard stats
    dashboard_success, _ = tester.test_dashboard_stats()
    
    # Test pools data
    pools_success, pools_data = tester.test_pools_data()
    
    # If pools data is available, test AI analysis for the first pool
    pool_address = None
    if pools_success and pools_data:
        try:
            pool_address = pools_data[0]["pool_address"]
            print(f"Found pool address: {pool_address}")
        except (KeyError, IndexError) as e:
            print(f"Could not extract pool address from response: {e}")
    
    if pool_address:
        tester.test_ai_analysis(pool_address)
    
    # Test wallet connect
    tester.test_wallet_connect()
    
    # Test wallet balance
    tester.test_wallet_balance()
    
    # Test MEV opportunities
    tester.test_mev_opportunities()
    
    # Test swap quote
    tester.test_swap_quote()
    
    # Print results
    print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())

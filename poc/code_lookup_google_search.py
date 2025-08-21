import requests
from datetime import datetime
import json
import re
from typing import List, Dict, Optional, Tuple
import time
from bs4 import BeautifulSoup
import asyncio
from google import genai
from google.genai import types
import os
import hashlib
from collections import defaultdict
import pandas as pd

class BuildingCodeFetcher:
    def __init__(self, cse_api_key: str, cse_id: str, use_vertex_ai: bool = True):
        """
        Initialize the Building Code Fetcher
        
        Args:
            cse_api_key: Google Custom Search Engine API key
            cse_id: Custom Search Engine ID
            use_vertex_ai: Whether to use Vertex AI for LLM processing
        """
        self.cse_api_key = cse_api_key
        self.cse_id = cse_id
        self.use_vertex_ai = use_vertex_ai
        self.raw_search_results = []  # Store raw search results
        
        if self.use_vertex_ai:
            # Initialize Vertex AI client
            self.client = genai.Client(
                vertexai=True,
                project="aistimate",
                location="global",
            )
            
            self.model_name = "gemini-2.0-flash-exp"
            
            self.generate_content_config = types.GenerateContentConfig(
                temperature=0,  # Keep at 0 for consistency
                top_p=0.95,  # Slightly reduced for more consistency
                seed=7,  # Fixed seed for reproducibility
                max_output_tokens=8192,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
            )
    
    def extract_location_info(self, address: str) -> Dict[str, str]:
        """Extract ZIP code, state, and city from address"""
        info = {}
        
        # Extract ZIP code
        zip_pattern = r'\b(\d{5}(?:-\d{4})?)\b'
        zip_match = re.search(zip_pattern, address)
        if zip_match:
            info['zip_code'] = zip_match.group(1)
        
        # Extract state (2-letter abbreviation)
        state_pattern = r'\b([A-Z]{2})\b'
        state_match = re.search(state_pattern, address)
        if state_match:
            info['state'] = state_match.group(1)
        
        # Extract city (word before state)
        city_pattern = r'([A-Za-z\s]+),\s*[A-Z]{2}'
        city_match = re.search(city_pattern, address)
        if city_match:
            info['city'] = city_match.group(1).strip().split(',')[-1].strip()
        
        return info
    
    def search_building_codes(self, address: str, loss_date: str, num_results: int = 10) -> List[Dict]:
        """
        Search for building codes using Google CSE
        
        Args:
            address: Property address
            loss_date: Date of loss (YYYY-MM-DD format)
            num_results: Number of search results to return
        
        Returns:
            List of search results
        """
        location_info = self.extract_location_info(address)
        
        # Build search query - for both IBC and IRC
        query_parts = [
            'IBC and IRC codes',
            loss_date,
            location_info.get('zip_code', ''),
            'requirements adoption'
        ]
        query = ' '.join(filter(None, query_parts))
        
        print(f"\n{'='*50}")
        print(f"Search Query: {query}")
        print(f"Location Info: {location_info}")
        print(f"{'='*50}\n")
        
        # Make API request
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'key': self.cse_api_key,
            'cx': self.cse_id,
            'q': query,
            'num': num_results,
            'sort': 'date:r:20200101:20251231'  # Sort by date for consistency
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Store raw results
            self.raw_search_results = data
            
            if 'items' in data:
                # Print raw search results for debugging
                print("📋 RAW GOOGLE SEARCH RESULTS:")
                print("-" * 50)
                for idx, item in enumerate(data['items'], 1):
                    print(f"{idx}. {item['title']}")
                    print(f"   URL: {item['link']}")
                    print(f"   Snippet: {item['snippet'][:100]}...")
                print("-" * 50 + "\n")
                
                return data['items']
            else:
                print("No search results found")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error during search: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text[:500]}")
            return []
    
    def fetch_page_content(self, url: str, max_chars: int = 10000) -> str:
        """
        Fetch and extract text content from a URL
        
        Args:
            url: URL to fetch
            max_chars: Maximum characters to extract
        
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            return text[:max_chars]
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""
    
    async def extract_building_codes_with_llm(self, content: str, address: str, loss_date: str, url: str) -> str:
        """
        Use Vertex AI Gemini to extract relevant building code information
        
        Args:
            content: Page content
            address: Property address
            loss_date: Date of loss
            url: Source URL for reference
        
        Returns:
            Extracted building code information
        """
        if not self.use_vertex_ai:
            return "Vertex AI not configured. Cannot extract with LLM."
        
        # Clear prompt for IBC and IRC codes
        prompt = f"""You are a building code expert. Extract ONLY IBC and IRC code information from this content.

IMPORTANT INSTRUCTIONS:
- Extract both IBC (International Building Code) and IRC (International Residential Code) information
- Focus on codes that apply to {address} on {loss_date}
- Include specific section numbers, chapters, and requirements
- If no building codes are found, return "No building codes found"

Extract the following information:
1. IBC version/year (e.g., IBC 2018, IBC 2021)
2. IRC version/year (e.g., IRC 2018, IRC 2021)
3. Specific section numbers for both codes
4. Chapter references
5. Table references
6. Adoption/effective dates in this location
7. Local amendments if mentioned

Format as clear bullet points with specific references.

Source URL: {url}
Content to analyze:
{content[:5000]}"""
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generate_content_config,
            )
            
            # Extract text from response
            if response and response.text:
                return response.text
            else:
                return "No response generated from LLM"
                
        except Exception as e:
            print(f"Error processing with Vertex AI: {e}")
            return f"Error processing with Vertex AI: {e}"
    
    async def process_search_results_async(self, address: str, loss_date: str, max_links: int = 5, 
                                          run_id: int = 1) -> Dict:
        """
        Main async function to search, fetch, and process building codes
        
        Args:
            address: Property address
            loss_date: Date of loss
            max_links: Maximum number of links to process
            run_id: Identifier for this run
        
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"BUILDING CODE SEARCH - RUN #{run_id}")
        print(f"Address: {address}")
        print(f"Date of Loss: {loss_date}")
        print(f"{'='*60}\n")
        
        run_start_time = time.time()
        
        # Step 1: Search for building codes
        search_results = self.search_building_codes(address, loss_date)
        
        if not search_results:
            print("No search results found.")
            return {
                'run_id': run_id,
                'address': address,
                'loss_date': loss_date,
                'raw_search_results': self.raw_search_results,
                'extracted_codes': [],
                'run_time': time.time() - run_start_time,
                'success': False
            }
        
        print(f"Found {len(search_results)} search results\n")
        
        # Step 2: Process each result and collect extracted codes
        all_extracted_codes = []
        
        for i, result in enumerate(search_results[:max_links], 1):
            print(f"\n{'='*60}")
            print(f"PROCESSING RESULT #{i}")
            print(f"Title: {result['title']}")
            print(f"URL: {result['link']}")
            print(f"{'='*60}")
            
            # Fetch page content
            print("📥 Fetching page content...")
            content = self.fetch_page_content(result['link'])
            
            if content:
                print(f"✓ Fetched {len(content)} characters")
                
                # Extract building codes with LLM
                if self.use_vertex_ai:
                    print("🔍 Extracting building codes with Vertex AI...")
                    extracted = await self.extract_building_codes_with_llm(
                        content, address, loss_date, result['link']
                    )
                    
                    print("\n📋 EXTRACTED BUILDING CODES:")
                    print("-" * 40)
                    print(extracted)
                    print("-" * 40)
                    
                    # Store the extracted codes
                    all_extracted_codes.append({
                        'source_title': result['title'],
                        'source_url': result['link'],
                        'source_snippet': result['snippet'],
                        'extracted_codes': extracted,
                        'extraction_time': datetime.now().isoformat()
                    })
                else:
                    print("❌ Vertex AI not configured")
                    all_extracted_codes.append({
                        'source_title': result['title'],
                        'source_url': result['link'],
                        'source_snippet': result['snippet'],
                        'extracted_codes': "Vertex AI not configured",
                        'extraction_time': datetime.now().isoformat()
                    })
            else:
                print("❌ Failed to fetch content")
                all_extracted_codes.append({
                    'source_title': result['title'],
                    'source_url': result['link'],
                    'source_snippet': result['snippet'],
                    'extracted_codes': "Failed to fetch page content",
                    'extraction_time': datetime.now().isoformat()
                })
            
            # Rate limiting
            await asyncio.sleep(1)
        
        run_time = time.time() - run_start_time
        
        # Return simplified results
        result_data = {
            'run_id': run_id,
            'address': address,
            'loss_date': loss_date,
            'timestamp': datetime.now().isoformat(),
            'raw_search_results': self.raw_search_results,
            'extracted_codes': all_extracted_codes,
            'run_time': run_time,
            'success': True
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total URLs processed: {len(all_extracted_codes)}")
        print(f"Run time: {run_time:.2f} seconds")
        print(f"Results saved with {len(all_extracted_codes)} extractions")
        
        return result_data
    
    def process_single_url(self, url: str, address: str, loss_date: str):
        """
        Process a single URL synchronously
        """
        return asyncio.run(self.process_single_url_async(url, address, loss_date))
    
    async def process_single_url_async(self, url: str, address: str, loss_date: str) -> Dict:
        """
        Process a single URL to extract building codes
        
        Args:
            url: URL to process
            address: Property address
            loss_date: Date of loss
        
        Returns:
            Dictionary with extracted codes
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING SINGLE URL")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        # Fetch page content
        print("📥 Fetching page content...")
        content = self.fetch_page_content(url)
        
        if content:
            print(f"✓ Fetched {len(content)} characters")
            
            # Extract building codes with LLM
            if self.use_vertex_ai:
                print("🔍 Extracting building codes with Vertex AI...")
                extracted = await self.extract_building_codes_with_llm(
                    content, address, loss_date, url
                )
                
                print("\n📋 EXTRACTED BUILDING CODES:")
                print("-" * 40)
                print(extracted)
                print("-" * 40)
                
                return {
                    'url': url,
                    'extracted_codes': extracted,
                    'success': True,
                    'extraction_time': datetime.now().isoformat()
                }
            else:
                return {
                    'url': url,
                    'extracted_codes': "Vertex AI not configured",
                    'success': False,
                    'extraction_time': datetime.now().isoformat()
                }
        else:
            return {
                'url': url,
                'extracted_codes': "Failed to fetch page content",
                'success': False,
                'extraction_time': datetime.now().isoformat()
            }


class SimplifiedTester:
    """
    Simplified tester that focuses on extraction results
    """
    
    def __init__(self, fetcher: BuildingCodeFetcher):
        self.fetcher = fetcher
        self.all_results = []
    
    async def run_test(self, address: str, loss_date: str, max_links: int = 3):
        """
        Run a single test and return results
        
        Args:
            address: Property address
            loss_date: Date of loss
            max_links: Maximum number of links to process
        
        Returns:
            Extracted codes from all sources
        """
        print(f"\n{'='*80}")
        print("BUILDING CODE EXTRACTION")
        print(f"{'='*80}\n")
        
        result = await self.fetcher.process_search_results_async(
            address=address,
            loss_date=loss_date,
            max_links=max_links,
            run_id=1
        )
        
        self.all_results = result
        
        # Save results
        self.save_results(address, loss_date)
        
        return result
    
    def save_results(self, address: str, loss_date: str):
        """
        Save extraction results to files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        json_filename = f"building_codes_extraction_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete results saved to: {json_filename}")
        
        # Save simplified text output
        text_filename = f"building_codes_summary_{timestamp}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(f"Building Code Extraction Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Address: {address}\n")
            f.write(f"Date of Loss: {loss_date}\n")
            f.write(f"Extraction Time: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
            
            if 'extracted_codes' in self.all_results:
                for i, item in enumerate(self.all_results['extracted_codes'], 1):
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Source #{i}: {item['source_title']}\n")
                    f.write(f"URL: {item['source_url']}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"\nExtracted Codes:\n")
                    f.write(f"{item['extracted_codes']}\n")
                    f.write(f"\n")
        
        print(f"📝 Summary saved to: {text_filename}")
        
        # Create a consolidated codes file
        codes_filename = f"consolidated_codes_{timestamp}.txt"
        with open(codes_filename, 'w', encoding='utf-8') as f:
            f.write(f"Consolidated Building Codes\n")
            f.write(f"Address: {address}\n")
            f.write(f"Date of Loss: {loss_date}\n")
            f.write(f"{'='*60}\n\n")
            
            all_codes = []
            if 'extracted_codes' in self.all_results:
                for item in self.all_results['extracted_codes']:
                    if item['extracted_codes'] and "No building codes found" not in item['extracted_codes']:
                        all_codes.append(item['extracted_codes'])
            
            if all_codes:
                f.write("\n\n".join(all_codes))
            else:
                f.write("No building codes were extracted from the sources.")
        
        print(f"📄 Consolidated codes saved to: {codes_filename}")


# Main execution
# async def main():
#     # Configuration
#     CSE_API_KEY = "AIzaSyD3xOVScQF_Glljhtnc5TEmBzANf4PHepc"
#     CSE_ID = "93ed7b757d0114273"
    
#     # Initialize fetcher with Vertex AI
#     fetcher = BuildingCodeFetcher(
#         cse_api_key=CSE_API_KEY,
#         cse_id=CSE_ID,
#         use_vertex_ai=True
#     )
    
#     # Initialize simplified tester
#     tester = SimplifiedTester(fetcher)
#     # Test case
#     test_address = "8705 COUNTY ROAD 206A, Alvarado, TX 76009"
#     test_date = "May 30 2024"
    
#     # Run the test and get results
#     results = await tester.run_test(
#         address=test_address,
#         loss_date=test_date,
#         max_links=5  # Process top 3 search results
#     )
    
#     # Return the extracted codes
#     return results


# # Simple function to get codes for a specific address
# async def get_building_codes(address: str, loss_date: str, max_links: int = 3):
#     """
#     Simple function to get building codes for an address
    
#     Args:
#         address: Property address
#         loss_date: Date of loss
#         max_links: Number of search results to process
    
#     Returns:
#         Dictionary with extracted codes
#     """
#     fetcher = BuildingCodeFetcher(
#         cse_api_key="AIzaSyD3xOVScQF_Glljhtnc5TEmBzANf4PHepc",
#         cse_id="93ed7b757d0114273",
#         use_vertex_ai=True
#     )
    
#     results = await fetcher.process_search_results_async(
#         address=address,
#         loss_date=loss_date,
#         max_links=max_links,
#         run_id=1
#     )
    
#     # Print consolidated results
#     print("\n" + "="*60)
#     print("CONSOLIDATED BUILDING CODES")
#     print("="*60)
    
#     for item in results['extracted_codes']:
#         if "No building codes found" not in item['extracted_codes']:
#             print(f"\nFrom: {item['source_title']}")
#             print(item['extracted_codes'])
#             print("-"*40)
    
#     return results


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
    
    # Or use the simple function:
    # asyncio.run(get_building_codes(
    #     address="8705 COUNTY ROAD 206A, Alvarado, TX 76009",
    #     loss_date="May 30 2024",
    #     max_links=3
    # ))
import re
import os
import sys
import psutil
import asyncio
from typing import List
from urllib.parse import urlparse
from app.utils.helper import helper
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter, LLMContentFilter

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "openai")
os.makedirs(__output__, exist_ok=True)

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def strip_line_numbers(text: str) -> str:
    """Remove leading line numbers, and delete lines that are just numbers."""
    cleaned_lines = []
    for line in text.splitlines():
        # If the line is only digits or whitespace, skip it entirely
        if re.fullmatch(r'\s*\d+\s*', line):
            continue
        # Otherwise, strip leading numbers and optional space
        cleaned_line = re.sub(r'^\s*\d+\s+', '', line)
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)

def sanitize_filename(url: str) -> str:
    """Convert URL to a safe filename"""
    parsed = urlparse(url)
    # Combine domain and path
    filename = f"{parsed.netloc}_{parsed.path}"
    # Replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    # Ensure it's not empty
    if not filename:
        filename = "unnamed"
    # Add .txt extension
    return f"{filename}.txt"

def save_content_to_file(url: str, content: str, output_dir: str) -> bool:
    """Save crawled content to a text file, after stripping line numbers"""
    try:
        # Remove line numbers before saving
        cleaned_content = strip_line_numbers(content)

        filename = sanitize_filename(url)
        filepath = os.path.join(output_dir, filename)
        
        # Handle potential duplicate filenames
        counter = 1
        original_filepath = filepath
        while os.path.exists(filepath):
            name, ext = os.path.splitext(original_filepath)
            filepath = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✓ Saved: {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to save {url}: {e}")
        return False

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check + File Output ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    md_generator = DefaultMarkdownGenerator(
        options={"ignore_links": True,}
    )
    
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        markdown_generator=md_generator,
        excluded_tags=['form', 'header', 'footer', 'nav', 'sidebar', 'aside'],
        exclude_internal_links=True,
        exclude_external_links=True,
        exclude_all_images=True,
        target_elements=['main'],
        prettiify=True,
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        saved_count = 0
        
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results and save to files
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"✗ Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    success_count += 1
                    
                    # Save the markdown content to file
                    content = result.markdown or result.cleaned_html or "No content extracted"
                    if save_content_to_file(url, content, __output__):
                        saved_count += 1
                else:
                    print(f"✗ Failed to crawl {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")
        print(f"  - Files saved: {saved_count}")
        print(f"  - Output directory: {__output__}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")

async def main():
    urls = helper.get_urls_from_sitemap("https://platform.openai.com/sitemap.xml")
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        print(f"Output will be saved to: {__output__}")
        await crawl_parallel(urls)
    else:
        print("No URLs found to crawl")


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
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
        target_elements=['main', 'article'],
        prettiify=True,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://docs.djangoproject.com/en/5.2/", config=crawl_config)
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from playwright.async_api import async_playwright

PROXIES = [
    '88.198.212.91:3128',
    '185.93.89.145:24666',
    '43.131.9.114:1777',
    '209.38.83.56:1088',
    '222.59.173.105:44228',
    '74.119.147.209:4145',
    '192.95.33.162:1129',
    '192.95.33.162:1129',
    '192.95.33.162:1129',
]

TEST_URL = 'https://www.airbnb.com/'

async def check_proxy(playwright, proxy_str):
    proxy_parts = proxy_str.split(':')
    proxy_config = {
        'server': f'http://{proxy_parts[0]}:{proxy_parts[1]}'
    }
    try:
        browser = await playwright.chromium.launch(headless=True, proxy=proxy_config)
        context = await browser.new_context()
        page = await context.new_page()
        response = await page.goto(TEST_URL, timeout=20000)
        status = response.status if response else None
        if status and 200 <= status < 400:
            print(f"[OK]    {proxy_str} (Status: {status})")
            result = True
        else:
            print(f"[FAIL]  {proxy_str} (Status: {status})")
            result = False
        await page.close()
        await context.close()
        await browser.close()
        return result
    except Exception as e:
        print(f"[ERROR] {proxy_str} - {e}")
        return False

async def main():
    working = 0
    async with async_playwright() as playwright:
        results = []
        for proxy in PROXIES:
            ok = await check_proxy(playwright, proxy)
            results.append((proxy, ok))
            if ok:
                working += 1
    print(f"\nSummary: {working}/{len(PROXIES)} proxies working.")
    for proxy, ok in results:
        if not ok:
            print(f"  Not working: {proxy}")

if __name__ == "__main__":
    asyncio.run(main()) 
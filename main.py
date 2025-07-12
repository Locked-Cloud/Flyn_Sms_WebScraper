import asyncio
import pandas as pd
import re
import time
import os
import glob
from playwright.async_api import async_playwright, Page, BrowserContext
from typing import Optional, List, Dict
import logging
from datetime import datetime
import json
import argparse
import functools
from tqdm import tqdm
import random
import string

# Configure logging
import sys

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = 'sms_service.log'

# Will be overridden by CLI if provided
if '--log-level' in sys.argv:
    idx = sys.argv.index('--log-level')
    if idx + 1 < len(sys.argv):
        LOG_LEVEL = sys.argv[idx + 1].upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Airbnb country code mapping (partial, add more as needed)
AIRBNB_COUNTRY_CODE_MAP = {
    'Ghana': '233GH',
    'Egypt': '20EG',
    'United States': '1US',
    'Nigeria': '234NG',
    'United Kingdom': '44GB',
    # Add more as needed
}

BROWSER_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def random_name():
    first = ''.join(random.choices(string.ascii_letters, k=6)).capitalize()
    last = ''.join(random.choices(string.ascii_letters, k=8)).capitalize()
    return first, last

def random_email():
    user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])
    return f"{user}@{domain}"

class PlaywrightNotInitialized(Exception):
    pass

def load_config() -> dict:
    """Load configuration from config.json or environment variables."""
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    # Override with environment variables if present
    config['SMS_USERNAME'] = os.getenv('SMS_USERNAME', config.get('SMS_USERNAME', ''))
    config['SMS_PASSWORD'] = os.getenv('SMS_PASSWORD', config.get('SMS_PASSWORD', ''))
    return config

async def async_retry(func, retries=3, delay=2, exceptions=(Exception,)):
    for attempt in range(retries):
        try:
            return await func()
        except exceptions as e:
            if attempt < retries - 1:
                logger.warning(f"Retrying {func.__name__} due to error: {e} (attempt {attempt+1}/{retries})")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed after {retries} attempts: {e}")
                raise

class SMSServiceHandler:
    sms_context: Optional[BrowserContext]
    sms_page: Optional[Page]
    def __init__(self, sms_username: str, sms_password: str):
        self.sms_username = sms_username
        self.sms_password = sms_password
        self.browser = None
        self.sms_context = None
        self.sms_page = None
        
    async def initialize(self):
        """Initialize Playwright browser and contexts"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Set to True for production
            slow_mo=1000    # Slow down operations
        )
        # Choose a random user agent for this session
        user_agent = random.choice(USER_AGENTS)
        # Merge headers, override User-Agent
        headers = BROWSER_HEADERS.copy()
        headers['User-Agent'] = user_agent
        # Create context for SMS service with stealth headers
        self.sms_context = await self.browser.new_context(
            user_agent=user_agent,
            extra_http_headers=headers,
            locale='en-US'
        )
        logger.info(f"Browser initialized with User-Agent: {user_agent}")
    
    def find_csv_file(self) -> str:
        """Find CSV file in current directory"""
        csv_patterns = [
            "*SMS*.csv",
            "*Number*.csv",
            "*Phone*.csv",
            "*.csv"
        ]
        
        for pattern in csv_patterns:
            files = glob.glob(pattern)
            if files:
                csv_file = files[0]
                logger.info(f"Found CSV file: {csv_file}")
                return csv_file
        
        all_files = os.listdir('.')
        logger.error(f"No CSV file found. Files in directory: {all_files}")
        raise FileNotFoundError("No CSV file found in current directory")
    
    def read_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Read CSV file and return DataFrame"""
        try:
            if file_path is None:
                file_path = self.find_csv_file()
            df = pd.read_csv(str(file_path))
            logger.info(f"Successfully read {len(df)} phone numbers from {file_path}")
            logger.info(f"CSV columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def solve_math_captcha(self, captcha_text: str) -> str:
        """Solve simple math captcha intelligently"""
        try:
            logger.info(f"Captcha text: {captcha_text}")
            
            patterns = [
                r'What is (\d+)\s*([+\-*/×÷])\s*(\d+)\s*=\s*\?',
                r'(\d+)\s*([+\-*/×÷])\s*(\d+)\s*=\s*\?',
                r'(\d+)\s*([+\-*/×÷])\s*(\d+)',
                r'What is (\d+)\s*([+\-*/×÷])\s*(\d+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, captcha_text, re.IGNORECASE)
                if match:
                    num1, operator, num2 = match.groups()
                    num1, num2 = int(num1), int(num2)
                    
                    if operator in ['+']:
                        result = num1 + num2
                    elif operator in ['-']:
                        result = num1 - num2
                    elif operator in ['*', '×']:
                        result = num1 * num2
                    elif operator in ['/', '÷']:
                        result = num1 // num2 if num2 != 0 else 0
                    else:
                        result = num1 + num2
                    
                    logger.info(f"Solved captcha: {num1} {operator} {num2} = {result}")
                    return str(result)
            
            numbers = re.findall(r'\d+', captcha_text)
            if len(numbers) >= 2:
                num1, num2 = int(numbers[0]), int(numbers[1])
                result = num1 + num2
                logger.info(f"Fallback solution: {num1} + {num2} = {result}")
                return str(result)
            
            logger.warning("Could not parse captcha, returning default")
            return "10"
            
        except Exception as e:
            logger.error(f"Error solving captcha: {e}")
            return "10"
    
    async def login_to_sms_service(self) -> bool:
        """Login to SMS service"""
        async def _login():
            try:
                if self.sms_context is None:
                    raise PlaywrightNotInitialized("Playwright context is not initialized.")
                self.sms_page = await self.sms_context.new_page()
                if not self.sms_page:
                    logger.error("Failed to create new page for SMS service.")
                    return False
                await self.sms_page.goto('http://91.232.105.47/ints/login')
                await self.sms_page.wait_for_load_state('networkidle')
                await self.sms_page.wait_for_selector('input[name="username"]', timeout=15000)
                await self.sms_page.fill('input[name="username"]', self.sms_username)
                await self.sms_page.fill('input[name="password"]', self.sms_password)
                captcha_solved = False
                try:
                    captcha_div = await self.sms_page.query_selector('div.wrap-input100')
                    captcha_text = await captcha_div.text_content() if captcha_div else None
                    if captcha_text and (('What is' in captcha_text) or ('=' in captcha_text)):
                        captcha_answer = self.solve_math_captcha(captcha_text or "")
                        await self.sms_page.fill('input[name="capt"]', captcha_answer)
                        logger.info(f"Filled captcha answer: {captcha_answer}")
                        captcha_solved = True
                except Exception as e:
                    logger.warning(f"Captcha method 1 failed: {e}")
                if not captcha_solved:
                    try:
                        page_text = await self.sms_page.text_content('body') if self.sms_page else None
                        if page_text and (('What is' in page_text) or ('=' in page_text)):
                            captcha_answer = self.solve_math_captcha(page_text or "")
                            captcha_input = await self.sms_page.query_selector('input[name="capt"]') if self.sms_page else None
                            if captcha_input:
                                await captcha_input.fill(captcha_answer)
                                logger.info(f"Filled captcha answer (method 2): {captcha_answer}")
                                captcha_solved = True
                    except Exception as e:
                        logger.warning(f"Captcha method 2 failed: {e}")
                if not captcha_solved:
                    logger.warning("Could not solve captcha automatically, using fallback")
                    try:
                        captcha_input = await self.sms_page.query_selector('input[name="capt"]') if self.sms_page else None
                        if captcha_input:
                            await captcha_input.fill('9')
                            logger.info("Used fallback captcha answer: 9")
                    except Exception as e:
                        logger.warning(f"Fallback captcha failed: {e}")
                login_button = await self.sms_page.query_selector('button[type="submit"]') if self.sms_page else None
                if not login_button and self.sms_page:
                    login_button = await self.sms_page.query_selector('input[type="submit"]')
                if not login_button and self.sms_page:
                    login_button = await self.sms_page.query_selector('button')
                if login_button:
                    await login_button.click()
                    await self.sms_page.wait_for_load_state('networkidle')
                    current_url = self.sms_page.url if self.sms_page else ''
                    if 'login' not in current_url.lower() or 'dashboard' in current_url.lower():
                        logger.info("Login successful, navigating to SMS data page for extraction")
                        # After login, go to the SMSCDRStats page to extract messages
                        await self.sms_page.goto('http://91.232.105.47/ints/client/SMSCDRStats')
                        await self.sms_page.wait_for_load_state('networkidle')
                        logger.info("Successfully navigated to SMSCDRStats for SMS extraction")
                        return True
                    else:
                        logger.error("Login may have failed - still on login page")
                        return False
                else:
                    logger.error("Could not find login button")
                    return False
            except Exception as e:
                logger.error(f"Failed to login to SMS service: {e}")
                return False
        try:
            result = await async_retry(_login, retries=3, delay=3)
            return bool(result)
        except Exception:
            return False
    
    async def extract_sms_data_from_table(self) -> List[Dict]:
        try:
            if self.sms_page is None:
                raise PlaywrightNotInitialized("SMS page is not initialized.")
            await self.sms_page.wait_for_selector('#dt', timeout=10000)  # type: ignore[reportGeneralTypeIssues]
            rows = await self.sms_page.query_selector_all('#dt tbody tr')  # type: ignore[reportGeneralTypeIssues]
            sms_data = []
            for row in rows:
                try:
                    style = await row.get_attribute('style')
                    if style and 'display: none' in style:
                        logger.debug("Skipping hidden row")
                        continue
                    cells = await row.query_selector_all('td')
                    if len(cells) >= 7:
                        date_cell = await cells[0].text_content() or ''
                        range_cell = await cells[1].text_content() or ''
                        number_cell = await cells[2].text_content() or ''
                        cli_cell = await cells[3].text_content() or ''
                        sms_cell = await cells[4].text_content() or ''
                        currency_cell = await cells[5].text_content() or ''
                        payout_cell = await cells[6].text_content() or ''
                        if date_cell.strip() == "0,0,0,2" or "0,0,0," in date_cell.strip():
                            logger.debug("Skipping total row")
                            continue
                        if not date_cell.strip() or not number_cell.strip():
                            logger.debug("Skipping empty row")
                            continue
                        sms_record = {
                            'date': date_cell.strip(),
                            'range': range_cell.strip(),
                            'number': number_cell.strip(),
                            'cli': cli_cell.strip(),
                            'sms': sms_cell.strip(),
                            'currency': currency_cell.strip(),
                            'payout': payout_cell.strip()
                        }
                        sms_data.append(sms_record)
                except Exception as e:
                    logger.warning(f"Error extracting row data: {e}")
                    continue
            logger.info(f"Extracted {len(sms_data)} SMS records from table")
            return sms_data
        except PlaywrightNotInitialized as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(f"Error extracting SMS data from table: {e}")
            return []

    async def display_existing_messages(self):
        try:
            logger.info("=" * 60)
            logger.info("CHECKING EXISTING MESSAGES IN SMS SERVICE")
            logger.info("=" * 60)
            if self.sms_page is None:
                raise PlaywrightNotInitialized("SMS page is not initialized.")
            await self.sms_page.wait_for_timeout(3000)  # type: ignore[reportGeneralTypeIssues]
            table_exists = await self.sms_page.query_selector('#dt')  # type: ignore[reportGeneralTypeIssues]
            if not table_exists:
                logger.warning("DataTable not found on page")
                await self.sms_page.screenshot(path='sms_page_after_login.png')  # type: ignore[reportGeneralTypeIssues]
                logger.info("Screenshot saved as 'sms_page_after_login.png'")
                return
            try:
                table_info = await self.sms_page.query_selector('#dt_info')  # type: ignore[reportGeneralTypeIssues]
                if table_info:
                    info_text = await table_info.text_content()
                    logger.info(f"Table info: {info_text}")
            except:
                pass
            existing_messages = await self.extract_sms_data_from_table()
            if existing_messages:
                logger.info(f"Found {len(existing_messages)} existing messages:")
                logger.info("-" * 60)
                for i, message in enumerate(existing_messages, 1):
                    logger.info(f"Message {i}:")
                    logger.info(f"  Date: {message['date']}")
                    logger.info(f"  Range: {message['range']}")
                    logger.info(f"  Number: {message['number']}")
                    logger.info(f"  CLI: {message['cli']}")
                    logger.info(f"  SMS: {message['sms']}")
                    logger.info(f"  Currency: {message['currency']}")
                    logger.info(f"  Payout: {message['payout']}")
                    logger.info("-" * 40)
                verification_codes = []
                for message in existing_messages:
                    if message['cli'].upper() == 'AIRBNB':
                        sms_text = message['sms']
                        code_patterns = [
                            r'verification code is (\d{6})',
                            r'code is (\d{6})',
                            r'code: (\d{6})',
                            r'(\d{6})'
                        ]
                        for pattern in code_patterns:
                            matches = re.findall(pattern, sms_text)
                            if matches:
                                verification_codes.append({
                                    'number': message['number'],
                                    'code': matches[0],
                                    'date': message['date'],
                                    'full_sms': sms_text
                                })
                                break
                if verification_codes:
                    logger.info("=" * 60)
                    logger.info("EXTRACTED VERIFICATION CODES:")
                    logger.info("=" * 60)
                    for i, code_info in enumerate(verification_codes, 1):
                        logger.info(f"Code {i}:")
                        logger.info(f"  Number: {code_info['number']}")
                        logger.info(f"  Code: {code_info['code']}")
                        logger.info(f"  Date: {code_info['date']}")
                        logger.info("-" * 40)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    codes_filename = f'verification_codes_{timestamp}.json'
                    with open(codes_filename, 'w', encoding='utf-8') as f:
                        json.dump(verification_codes, f, ensure_ascii=False, indent=2)
                    logger.info(f"Verification codes saved to {codes_filename}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'sms_messages_{timestamp}.json'
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(existing_messages, f, ensure_ascii=False, indent=2)
                logger.info(f"All messages saved to {filename}")
            else:
                logger.info("No existing messages found in the table")
            logger.info("=" * 60)
            logger.info("FINISHED CHECKING EXISTING MESSAGES")
            logger.info("=" * 60)
        except PlaywrightNotInitialized as e:
            logger.error(str(e))
            return
        except Exception as e:
            logger.error(f"Error displaying existing messages: {e}")
            try:
                if self.sms_page and hasattr(self.sms_page, 'screenshot'):
                    await self.sms_page.screenshot(path='error_existing_messages.png')  # type: ignore[reportGeneralTypeIssues]
                    logger.info("Error screenshot saved as 'error_existing_messages.png'")
            except:
                pass
    
    async def test_specific_phone_numbers(self, phone_numbers: List[str]):
        """Test specific phone numbers for verification codes"""
        try:
            logger.info("=" * 60)
            logger.info("TESTING SPECIFIC PHONE NUMBERS")
            logger.info("=" * 60)
            
            for phone_number in phone_numbers:
                logger.info(f"Testing phone number: {phone_number}")
                code = await self.get_verification_code_for_number(phone_number, timeout=30)
                
                if code:
                    logger.info(f"✓ Found verification code for {phone_number}: {code}")
                else:
                    logger.info(f"✗ No verification code found for {phone_number}")
                
                logger.info("-" * 40)
                
        except Exception as e:
            logger.error(f"Error testing specific phone numbers: {e}")
    
    async def get_verification_code_for_number(self, phone_number: str, timeout: int = 120) -> Optional[str]:
        try:
            if self.sms_page is None:
                raise PlaywrightNotInitialized("SMS page is not initialized.")
            start_time = time.time()
            logger.info(f"Looking for verification code for number: {phone_number}")
            clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '')
            logger.info(f"Cleaned phone number: {clean_phone}")
            initial_messages = await self.extract_sms_data_from_table()
            initial_count = len(initial_messages)
            logger.info(f"Initial message count: {initial_count}")
            logger.info("Checking existing messages for verification code...")
            for message in initial_messages:
                message_number = message['number'].replace('+', '').replace('-', '').replace(' ', '')
                if (clean_phone == message_number or 
                    clean_phone in message_number or 
                    message_number in clean_phone or
                    clean_phone.endswith(message_number[-8:]) or 
                    message_number.endswith(clean_phone[-8:])):
                    if message['cli'].upper() == 'AIRBNB':
                        logger.info(f"Found existing Airbnb SMS for {phone_number}")
                        sms_text = message['sms']
                        code_patterns = [
                            r'verification code is (\d{6})',
                            r'code is (\d{6})',
                            r'code: (\d{6})',
                            r'(\d{6})'
                        ]
                        for pattern in code_patterns:
                            matches = re.findall(pattern, sms_text)
                            if matches:
                                code = matches[0]
                                logger.info(f"✓ Found existing verification code: {code}")
                                return code
            logger.info("No existing verification code found. Monitoring for new messages...")
            while time.time() - start_time < timeout:
                if self.sms_page is None:
                    raise PlaywrightNotInitialized("SMS page is not initialized.")
                logger.info("Refreshing SMS page to check for new messages...")
                await async_retry(lambda: self.sms_page.reload(), retries=3, delay=2)  # type: ignore[reportAttributeAccessIssue]
                await self.sms_page.wait_for_load_state('networkidle')  # type: ignore[reportGeneralTypeIssues]
                try:
                    await self.sms_page.wait_for_selector('#dt tbody', timeout=5000)  # type: ignore[reportGeneralTypeIssues]
                except:
                    logger.warning("DataTable not found, waiting...")
                    await asyncio.sleep(5)
                    continue
                sms_data = await self.extract_sms_data_from_table()
                current_count = len(sms_data)
                if current_count > initial_count:
                    logger.info(f"New messages detected! Count: {current_count} (was {initial_count})")
                    new_messages = sms_data[initial_count:]
                    for message in new_messages:
                        message_number = message['number'].replace('+', '').replace('-', '').replace(' ', '')
                        if (clean_phone == message_number or 
                            clean_phone in message_number or 
                            message_number in clean_phone or
                            clean_phone.endswith(message_number[-8:]) or 
                            message_number.endswith(clean_phone[-8:])):
                            if message['cli'].upper() == 'AIRBNB':
                                logger.info(f"Found new Airbnb SMS for {phone_number}")
                                sms_text = message['sms']
                                code_patterns = [
                                    r'verification code is (\d{6})',
                                    r'code is (\d{6})',
                                    r'code: (\d{6})',
                                    r'(\d{6})'
                                ]
                                for pattern in code_patterns:
                                    matches = re.findall(pattern, sms_text)
                                    if matches:
                                        code = matches[0]
                                        logger.info(f"✓ Found new verification code: {code}")
                                        return code
                    initial_count = current_count
                elapsed_time = int(time.time() - start_time)
                logger.info(f"No verification code found yet, waiting... ({elapsed_time}s/{timeout}s elapsed)")
                await asyncio.sleep(10)
            logger.warning(f"No verification code received for {phone_number} within {timeout} seconds")
            return None
        except PlaywrightNotInitialized as e:
            logger.error(str(e))
            return None
        except Exception as e:
            logger.error(f"Error getting verification code: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def monitor_new_messages(self, duration_minutes: int = 5):
        try:
            if self.sms_page is None:
                raise PlaywrightNotInitialized("SMS page is not initialized.")
            logger.info("=" * 60)
            logger.info(f"MONITORING FOR NEW MESSAGES ({duration_minutes} minutes)")
            logger.info("=" * 60)
            initial_messages = await self.extract_sms_data_from_table()
            initial_count = len(initial_messages)
            logger.info(f"Initial message count: {initial_count}")
            start_time = time.time()
            duration_seconds = duration_minutes * 60
            check_interval = 30
            with tqdm(total=duration_seconds, desc="Monitoring", unit="sec") as pbar:
                while time.time() - start_time < duration_seconds:
                    elapsed = int(time.time() - start_time)
                    remaining = duration_seconds - elapsed
                    logger.info(f"Checking for new messages... ({elapsed}s elapsed, {remaining}s remaining)")
                    if self.sms_page is None:
                        raise PlaywrightNotInitialized("SMS page is not initialized.")
                    await async_retry(lambda: self.sms_page.reload(), retries=3, delay=2)  # type: ignore[reportAttributeAccessIssue]
                    await self.sms_page.wait_for_load_state('networkidle')  # type: ignore[reportGeneralTypeIssues]
                    try:
                        await self.sms_page.wait_for_selector('#dt tbody', timeout=10000)  # type: ignore[reportGeneralTypeIssues]
                    except:
                        logger.warning("DataTable not found, waiting...")
                        await asyncio.sleep(check_interval)
                        pbar.update(check_interval)
                        continue
                    current_messages = await self.extract_sms_data_from_table()
                    current_count = len(current_messages)
                    if current_count > initial_count:
                        new_message_count = current_count - initial_count
                        logger.info(f"🔔 NEW MESSAGES DETECTED! {new_message_count} new message(s)")
                        new_messages = current_messages[-new_message_count:]
                        logger.info("-" * 60)
                        logger.info("NEW MESSAGES:")
                        for i, message in enumerate(new_messages, 1):
                            logger.info(f"New Message {i}:")
                            logger.info(f"  Date: {message['date']}")
                            logger.info(f"  Range: {message['range']}")
                            logger.info(f"  Number: {message['number']}")
                            logger.info(f"  CLI: {message['cli']}")
                            logger.info(f"  SMS: {message['sms']}")
                            logger.info(f"  Currency: {message['currency']}")
                            logger.info(f"  Payout: {message['payout']}")
                            logger.info("-" * 40)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f'new_sms_messages_{timestamp}.json'
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(new_messages, f, ensure_ascii=False, indent=2)
                        logger.info(f"New messages saved to {filename}")
                        initial_count = current_count
                    else:
                        logger.info(f"No new messages. Current count: {current_count}")
                    await asyncio.sleep(check_interval)
                    pbar.update(check_interval)
                pbar.n = duration_seconds
                pbar.refresh()
            logger.info("=" * 60)
            logger.info("FINISHED MONITORING FOR NEW MESSAGES")
            logger.info("=" * 60)
        except PlaywrightNotInitialized as e:
            logger.error(str(e))
            return
        except Exception as e:
            logger.error(f"Error monitoring new messages: {e}")
            import traceback
            traceback.print_exc()
    
    async def analyze_phone_numbers(self, df: pd.DataFrame):
        """Analyze phone numbers from CSV against received SMS messages"""
        try:
            logger.info("=" * 60)
            logger.info("ANALYZING PHONE NUMBERS FROM CSV")
            logger.info("=" * 60)
            current_messages = await self.extract_sms_data_from_table()
            logger.info(f"CSV contains {len(df)} phone numbers")
            logger.info(f"SMS service has {len(current_messages)} messages")
            phone_to_messages = {}
            for index, row in tqdm(list(enumerate(df.itertuples(index=False))), desc="Analyzing numbers", unit="number"):
                phone_number = str(getattr(row, 'Number')).strip()
                clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '')
                logger.info(f"Analyzing phone number {index + 1}: {phone_number}")
                matching_messages = []
                for message in current_messages:
                    message_number = message['number'].replace('+', '').replace('-', '').replace(' ', '')
                    if (clean_phone == message_number or 
                        clean_phone in message_number or 
                        message_number in clean_phone or
                        clean_phone.endswith(message_number[-8:]) or 
                        message_number.endswith(clean_phone[-8:])):
                        matching_messages.append(message)
                phone_to_messages[phone_number] = matching_messages
                if matching_messages:
                    logger.info(f"  ✓ Found {len(matching_messages)} message(s) for {phone_number}")
                    for msg in matching_messages:
                        logger.info(f"    - CLI: {msg['cli']}, SMS: {msg['sms'][:50]}...")
                else:
                    logger.info(f"  ✗ No messages found for {phone_number}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = f'phone_analysis_{timestamp}.json'
            analysis_data = {
                'timestamp': timestamp,
                'total_phone_numbers': len(df),
                'total_messages': len(current_messages),
                'phone_to_messages': phone_to_messages
            }
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Analysis results saved to {analysis_filename}")
            numbers_with_messages = sum(1 for messages in phone_to_messages.values() if messages)
            numbers_without_messages = len(df) - numbers_with_messages
            logger.info(f"Summary:")
            logger.info(f"  - Phone numbers with messages: {numbers_with_messages}")
            logger.info(f"  - Phone numbers without messages: {numbers_without_messages}")
            logger.info("=" * 60)
            logger.info("FINISHED ANALYZING PHONE NUMBERS")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Error analyzing phone numbers: {e}")
            import traceback
            traceback.print_exc()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.sms_context:
                await self.sms_context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="SMS Service Automation Tool")
    parser.add_argument('--username', type=str, help='SMS service username')
    parser.add_argument('--password', type=str, help='SMS service password')
    parser.add_argument('--csv', type=str, help='Path to CSV file with phone numbers')
    parser.add_argument('--action', type=str, choices=['analyze', 'monitor', 'display', 'test'], default='analyze', help='Action to perform')
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes for monitoring new messages')
    parser.add_argument('--test-numbers', type=str, nargs='*', help='Phone numbers to test (for test action)')
    parser.add_argument('--log-level', type=str, default=LOG_LEVEL, help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    return parser.parse_args()

async def signup_airbnb_with_numbers(browser, phone_numbers: list, handler=None):
    for phone_number in phone_numbers:
        page = await browser.new_page()
        await page.goto('https://www.airbnb.com/signup_login')
        await page.wait_for_load_state('networkidle')
        # Click "Continue with phone" tab if needed
        try:
            await page.click('button[data-testid="signup-login-phone-tab"]')
        except Exception:
            pass  # If already on phone tab, ignore
        # Fill the phone number
        try:
            await page.fill('input[name="phoneNumber"]', phone_number)
            await page.click('button[type="submit"]')
            logger.info(f'Submitted signup for phone number: {phone_number}')
        except Exception as e:
            logger.error(f'Error submitting signup for {phone_number}: {e}')
        await asyncio.sleep(5)  # Wait for the code to arrive
        # Refresh SMS data and get verification code
        if handler is not None and handler.sms_page is not None:
            await handler.sms_page.reload()  # type: ignore[reportAttributeAccessIssue]
            await handler.sms_page.wait_for_load_state('networkidle')  # type: ignore[reportGeneralTypeIssues]
            messages = await handler.extract_sms_data_from_table()
            clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '')
            found_code = None
            for message in messages:
                message_number = message['number'].replace('+', '').replace('-', '').replace(' ', '')
                if (clean_phone == message_number or 
                    clean_phone in message_number or 
                    message_number in clean_phone or
                    clean_phone.endswith(message_number[-8:]) or 
                    message_number.endswith(clean_phone[-8:])):
                    if message['cli'].upper() == 'AIRBNB':
                        sms_text = message['sms']
                        import re
                        code_patterns = [
                            r'verification code is (\d{6})',
                            r'code is (\d{6})',
                            r'code: (\d{6})',
                            r'(\d{6})'
                        ]
                        for pattern in code_patterns:
                            matches = re.findall(pattern, sms_text)
                            if matches:
                                found_code = matches[0]
                                break
                if found_code:
                    break
            if found_code:
                logger.info(f'Verification code for {phone_number}: {found_code}')
                print(f'Verification code for {phone_number}: {found_code}')
            else:
                logger.warning(f'No verification code found for {phone_number} after refresh.')
        await page.close()

async def get_fresh_verification_code_for_number(handler, sms_page, phone_number, timeout=120):
    await sms_page.reload()
    await sms_page.wait_for_load_state('networkidle')
    initial_messages = await handler.extract_sms_data_from_table()
    initial_count = len(initial_messages)
    clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '')
    start_time = time.time()
    signup_start_dt = datetime.now()
    while time.time() - start_time < timeout:
        await sms_page.reload()
        await sms_page.wait_for_load_state('networkidle')
        try:
            await sms_page.wait_for_selector('#dt tbody', timeout=10000)
        except:
            await asyncio.sleep(5)
            continue
        sms_data = await handler.extract_sms_data_from_table()
        current_count = len(sms_data)
        if current_count > initial_count:
            new_messages = sms_data[initial_count:]
            for message in new_messages:
                message_number = message['number'].replace('+', '').replace('-', '').replace(' ', '')
                message_time_str = message.get('date', '').strip()
                try:
                    message_time = datetime.strptime(message_time_str, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    message_time = signup_start_dt
                if (message_time >= signup_start_dt and
                    (clean_phone == message_number or 
                     clean_phone in message_number or 
                     message_number in clean_phone or
                     clean_phone.endswith(message_number[-8:]) or 
                     message_number.endswith(clean_phone[-8:]))):
                    if message['cli'].upper() == 'AIRBNB':
                        sms_text = message['sms']
                        code_patterns = [
                            r'verification code is (\d{6})',
                            r'code is (\d{6})',
                            r'code: (\d{6})',
                            r'(\d{6})'
                        ]
                        for pattern in code_patterns:
                            matches = re.findall(pattern, sms_text)
                            if matches:
                                return matches[0]
            initial_count = current_count
        await asyncio.sleep(5)
    logger.warning(f'No new verification code found for {phone_number} after {timeout} seconds (checked by number and time)')
    return None

# --- Stealth/anti-detection improvements ---
# For each signup, use a new incognito context, rotate user agent, set headers, and clear cookies/storage by context isolation.
# (If Playwright Stealth plugin for Python becomes available, it should be used here for even better anti-detection.)

# Proxy list for rotation (format: host:port)
PROXIES = [
    '88.198.212.91:3128',
    '185.93.89.145:24666',
    '43.131.9.114:1777',
    '209.38.83.56:1088',
    '222.59.173.105:44228',
    '74.119.147.209:4145',
    '192.95.33.162:1129',
]

async def create_full_airbnb_account_with_sms(playwright, handler, phone_number, country_name):
    country_code_value = AIRBNB_COUNTRY_CODE_MAP.get(country_name, '20EG')
    first_name, last_name = random_name()
    email = random_email()
    birthday = f"19{random.randint(80, 99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

    # Remove country code prefix from phone_number for input
    country_prefix = ''.join([c for c in country_code_value if c.isdigit()])
    local_number = phone_number
    if phone_number.startswith(country_prefix):
        local_number = phone_number[len(country_prefix):]
    local_number = local_number.lstrip('0').strip()

    # --- Stealth: new browser, rotate user agent, set headers, clear cookies/storage, rotate proxy ---
    user_agent = random.choice(USER_AGENTS)
    headers = BROWSER_HEADERS.copy()
    headers['User-Agent'] = user_agent
    proxy_str = random.choice(PROXIES)
    proxy_parts = proxy_str.split(':')
    proxy_config = {
        'server': f'http://{proxy_parts[0]}:{proxy_parts[1]}'
    }
    # No username/password for these proxies
    logger.info(f"[STEALTH] Using User-Agent: {user_agent} | Proxy: {proxy_config['server']}")
    browser = await playwright.chromium.launch(
        headless=False,
        slow_mo=1000,
        proxy=proxy_config
    )
    context = await browser.new_context(
        user_agent=user_agent,
        extra_http_headers=headers,
        locale='en-US'
    )
    # (If Playwright Stealth plugin for Python is available, apply it here)
    airbnb_page = await context.new_page()
    try:
        await airbnb_page.goto('https://www.airbnb.com/signup_login')
        await airbnb_page.wait_for_load_state('networkidle')
    except Exception as e:
        logger.error(f'Proxy failed or blocked for {proxy_config["server"]}: {e}')
        await airbnb_page.close()
        await context.close()
        await browser.close()
        return  # Skip to next number/proxy
    try:
        await airbnb_page.click('button[data-testid="signup-login-phone-tab"]')
    except Exception:
        pass
    try:
        await airbnb_page.select_option('select[data-testid="login-signup-countrycode"]', value=country_code_value)
    except Exception as e:
        logger.warning(f"Could not select country code: {e}")
    try:
        await airbnb_page.fill('input[name="phoneInputphone-login"]', local_number)
        await airbnb_page.click('button[type="submit"]')
        logger.info(f'Submitted signup for phone number: {local_number} (original: {phone_number})')
    except Exception as e:
        logger.error(f'Error submitting signup for {phone_number}: {e}')
        await airbnb_page.close()
        await context.close()
        await browser.close()
        return
    # Wait for code input to appear (new selector)
    try:
        await airbnb_page.wait_for_selector('input#phone-verification-code-form__code-input', timeout=30000)
    except Exception as e:
        logger.error(f'Code input did not appear for {phone_number}: {e}')
        await airbnb_page.close()
        await context.close()
        await browser.close()
        return
    # Open SMSCDRStats in a new tab and get code
    sms_page = await context.new_page()
    await sms_page.goto('http://91.232.105.47/ints/client/SMSCDRStats')
    await sms_page.wait_for_load_state('networkidle')
    code = await get_fresh_verification_code_for_number(handler, sms_page, phone_number, timeout=120)
    await sms_page.close()
    if not code:
        logger.error(f'No verification code found for {phone_number}')
        await airbnb_page.close()
        await context.close()
        await browser.close()
        return
    # Enter code in Airbnb (new selector)
    try:
        await airbnb_page.fill('input#phone-verification-code-form__code-input', code)
        # Find the enabled Continue button and click it
        buttons = await airbnb_page.query_selector_all('button')
        for btn in buttons:
            text = await btn.text_content()
            disabled = await btn.get_attribute('disabled')
            if text and 'Continue' in text and not disabled:
                await btn.click()
                break
        logger.info(f'Entered verification code for {phone_number}: {code}')
        # Check for max confirmation attempts error
        await asyncio.sleep(2)
        page_text = await airbnb_page.content()
        if 'max confirmation attempts' in page_text.lower() or 'try again in 1 hour' in page_text.lower():
            logger.warning(f'Max confirmation attempts reached for {phone_number}. Skipping this number for now.')
            await airbnb_page.close()
            await context.close()
            await browser.close()
            return
    except Exception as e:
        logger.error(f'Error entering verification code for {phone_number}: {e}')
        await airbnb_page.close()
        await context.close()
        await browser.close()
        return
    # --- Complete the signup: Name, Email, Birthday ---
    # 1. Name
    try:
        await airbnb_page.wait_for_selector('input[name="firstName"]', timeout=20000)
        await airbnb_page.fill('input[name="firstName"]', first_name)
        await airbnb_page.fill('input[name="lastName"]', last_name)
        buttons = await airbnb_page.query_selector_all('button')
        for btn in buttons:
            text = await btn.text_content()
            if text and ('Continue' in text or 'Next' in text):
                await btn.click()
                break
        logger.info(f'Entered name for {phone_number}: {first_name} {last_name}')
    except Exception as e:
        logger.warning(f'Name step may be skipped or failed: {e}')
    # 2. Email
    try:
        await airbnb_page.wait_for_selector('input[type="email"]', timeout=20000)
        await airbnb_page.fill('input[type="email"]', email)
        buttons = await airbnb_page.query_selector_all('button')
        for btn in buttons:
            text = await btn.text_content()
            if text and ('Continue' in text or 'Next' in text):
                await btn.click()
                break
        logger.info(f'Entered email for {phone_number}: {email}')
    except Exception as e:
        logger.warning(f'Email step may be skipped or failed: {e}')
    # 3. Birthday
    try:
        await airbnb_page.wait_for_selector('input[name="birthdate"]', timeout=20000)
        await airbnb_page.fill('input[name="birthdate"]', birthday)
        buttons = await airbnb_page.query_selector_all('button')
        for btn in buttons:
            text = await btn.text_content()
            if text and ('Continue' in text or 'Next' in text):
                await btn.click()
                break
        logger.info(f'Entered birthday for {phone_number}: {birthday}')
    except Exception as e:
        logger.warning(f'Birthday step may be skipped or failed: {e}')
    await asyncio.sleep(3)
    logger.info(f'Account creation flow finished for {phone_number}')
    await airbnb_page.close()
    await context.close()
    await browser.close()

async def main():
    args = parse_args()
    # Update log level if provided via CLI
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    config = load_config()
    SMS_USERNAME = args.username or config.get('SMS_USERNAME', '')
    SMS_PASSWORD = args.password or config.get('SMS_PASSWORD', '')
    csv_path = args.csv
    if not SMS_USERNAME or not SMS_PASSWORD:
        logger.error('SMS_USERNAME and SMS_PASSWORD must be set via CLI, config.json, or environment variables.')
        return
    handler = SMSServiceHandler(SMS_USERNAME, SMS_PASSWORD)
    try:
        await handler.initialize()
        if await handler.login_to_sms_service():
            logger.info("Successfully logged into SMS service")
            if args.action == 'display':
                await handler.display_existing_messages()
            elif args.action == 'analyze':
                await handler.display_existing_messages()
                df = handler.read_csv(csv_path)
                await handler.analyze_phone_numbers(df)
                # After analysis, automate Airbnb signup for each phone number and get verification code
                async with async_playwright() as playwright:
                    for _, row in df.iterrows():
                        phone_number = str(row['Number']).strip()
                        country_name = str(row['Range']).strip()
                        # If max confirmation attempts is hit, the function will log and skip to the next number
                        await create_full_airbnb_account_with_sms(playwright, handler, phone_number, country_name)
            elif args.action == 'monitor':
                await handler.display_existing_messages()
                await handler.monitor_new_messages(duration_minutes=args.duration)
            elif args.action == 'test':
                if args.test_numbers:
                    await handler.test_specific_phone_numbers(args.test_numbers)
                else:
                    logger.error('No phone numbers provided for test action.')
            else:
                logger.error(f'Unknown action: {args.action}')
            # After all SMS extraction/analysis, open Airbnb signup/login page in a new tab
            if handler.browser:
                airbnb_page = await handler.browser.new_page()  # This opens a new tab
                await airbnb_page.goto('https://www.airbnb.com/signup_login')
                logger.info('Opened Airbnb signup/login page in a new browser tab.')
        else:
            logger.error("Failed to login to SMS service")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await handler.cleanup()
        # Explicitly close the asyncio event loop to avoid Windows resource warnings
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.stop()
        except Exception:
            pass

if __name__ == "__main__":
    import sys
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Interrupted by user. Exiting...')
    finally:
        # Clean up all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
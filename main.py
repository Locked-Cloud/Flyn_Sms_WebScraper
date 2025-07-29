import asyncio
import pandas as pd
import re
import time
import os
import glob
from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
import json
import argparse
import random
import string
import traceback
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.orm import sessionmaker
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import dotenv
import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich.table import Table
from pathlib import Path
import sys
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid
from contextlib import asynccontextmanager

# Load environment variables
dotenv.load_dotenv()

# Initialize console for rich output
console = Console()

class ServiceType(str, Enum):
    VALR = "valr"

class MessageStatus(str, Enum):
    PENDING = "pending"
    RECEIVED = "received"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ProxyConfig:
    """Proxy configuration"""
    host: str
    port: int
    username: str
    password: str
    country: str = ""
    region: str = ""
    
    @property
    def proxy_url(self) -> str:
        return f"http://{self.username}:{self.password}@{self.host}:{self.port}"
    
    @classmethod
    def from_string(cls, proxy_string: str) -> "ProxyConfig":
        """Parse proxy string format: host:port:username:password|country|region|..."""
        parts = proxy_string.split('|')
        proxy_part = parts[0]
        
        # Parse proxy credentials
        proxy_components = proxy_part.split(':')
        if len(proxy_components) < 4:
            raise ValueError(f"Invalid proxy format: {proxy_string}")
        
        host = proxy_components[0]
        port = int(proxy_components[1])
        username = proxy_components[2]
        password = proxy_components[3]
        
        # Extract location info if available
        country = parts[1] if len(parts) > 1 else ""
        region = parts[2] if len(parts) > 2 else ""
        
        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            country=country,
            region=region
        )

@dataclass
class VerificationResult:
    phone_number: str
    service: ServiceType
    success: bool
    code: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    proxy_used: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

# Enhanced configuration
class SMSServiceConfig(BaseModel):
    username: str = Field(..., description="SMS service username")
    password: str = Field(..., description="SMS service password")
    base_url: str = Field("http://91.232.105.47/ints", description="Base URL for SMS service")
    timeout: int = Field(30, description="Timeout for SMS service requests", ge=10, le=300)
    max_retries: int = Field(3, description="Maximum number of retries", ge=1, le=10)
    session_timeout: int = Field(3600, description="Session timeout in seconds", ge=300)

class ServiceConfig(BaseModel):
    valr_url: str = Field("https://www.valr.com/en/verify/phone", description="VALR verification URL")
    max_attempts_per_number: int = Field(3, description="Maximum attempts per phone number", ge=1, le=10)
    wait_between_attempts: int = Field(5, description="Wait time between attempts", ge=1, le=60)
    verification_timeout: int = Field(180, description="Timeout for verification code", ge=30, le=600)
    rate_limit_delay: Tuple[float, float] = Field((2.0, 8.0), description="Random delay range between requests")

class BrowserConfig(BaseModel):
    headless: bool = Field(True, description="Run browser in headless mode")
    slow_mo: int = Field(100, description="Slow motion delay", ge=0, le=5000)
    viewport_width: int = Field(1920, description="Browser viewport width", ge=800, le=3840)
    viewport_height: int = Field(1080, description="Browser viewport height", ge=600, le=2160)
    user_agent_rotation: bool = Field(True, description="Rotate user agents")
    stealth_mode: bool = Field(True, description="Enable stealth mode")

class ProxyPoolConfig(BaseModel):
    enabled: bool = Field(False, description="Enable proxy pool")
    proxy_list: List[str] = Field(default_factory=list, description="List of proxy strings")
    rotation_strategy: str = Field("random", description="Proxy rotation strategy")
    max_failures_per_proxy: int = Field(3, description="Max failures before rotating proxy")

class Config(BaseModel):
    sms_service: SMSServiceConfig
    services: ServiceConfig
    browser: BrowserConfig
    proxy_pool: ProxyPoolConfig
    
    @classmethod
    def load_from_file(cls, path: str = "config.json") -> "Config":
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                return cls(**data)
            else:
                console.print(f"[yellow]Config file {path} not found, creating default config[/]")
                config = cls.create_default()
                config.save_to_file(path)
                return config
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/]")
            return cls.create_default()
    
    @classmethod
    def create_default(cls) -> "Config":
        return cls(
            sms_service=SMSServiceConfig(
                username=os.getenv("SMS_USERNAME", ""),
                password=os.getenv("SMS_PASSWORD", "")
            ),
            services=ServiceConfig(),
            browser=BrowserConfig(),
            proxy_pool=ProxyPoolConfig()
        )
    
    def save_to_file(self, path: str = "config.json"):
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2, default=str)

# Load configuration
config = Config.load_from_file()

# Enhanced database models
Base = declarative_base()

class SMSMessage(Base):
    __tablename__ = "sms_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True, nullable=False)
    service_type = Column(String, index=True, nullable=False)
    verification_code = Column(String, nullable=True)
    cli = Column(String, index=True)
    sms_content = Column(Text)
    status = Column(String, default=MessageStatus.PENDING)
    received_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    session_id = Column(String, index=True)

class PhoneNumber(Base):
    __tablename__ = "phone_numbers"
    
    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, unique=True, index=True, nullable=False)
    country = Column(String, nullable=False)
    country_code = Column(String, nullable=False)
    status = Column(String, default="active")
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
engine = create_engine(
    "sqlite:///valr_sms.db",
    echo=False,
    pool_size=20,
    pool_timeout=30,
    pool_pre_ping=True
)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Enhanced exception handling
class SMSServiceError(Exception):
    pass

class CaptchaError(SMSServiceError):
    pass

class LoginError(SMSServiceError):
    pass

class RateLimitError(SMSServiceError):
    pass

class VerificationTimeoutError(SMSServiceError):
    pass

class ProxyError(SMSServiceError):
    pass

# Proxy pool manager
class ProxyPoolManager:
    """Manages proxy rotation and health checking"""
    
    def __init__(self, proxy_strings: List[str]):
        self.proxies: List[ProxyConfig] = []
        self.proxy_failures: Dict[str, int] = {}
        self.current_index = 0
        
        # Parse proxy strings
        for proxy_string in proxy_strings:
            try:
                proxy = ProxyConfig.from_string(proxy_string.strip())
                self.proxies.append(proxy)
                self.proxy_failures[proxy.proxy_url] = 0
            except Exception as e:
                logger.warning(f"Failed to parse proxy: {proxy_string}, error: {e}")
        
        logger.info(f"Initialized proxy pool with {len(self.proxies)} proxies")
    
    def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next available proxy"""
        if not self.proxies:
            return None
        
        # Find a proxy with fewer than max failures
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            
            if self.proxy_failures[proxy.proxy_url] < config.proxy_pool.max_failures_per_proxy:
                return proxy
            
            attempts += 1
        
        # If all proxies have max failures, reset failure counts and try again
        logger.warning("All proxies have max failures, resetting failure counts")
        for proxy_url in self.proxy_failures:
            self.proxy_failures[proxy_url] = 0
        
        return self.proxies[0] if self.proxies else None
    
    def get_random_proxy(self) -> Optional[ProxyConfig]:
        """Get random available proxy"""
        if not self.proxies:
            return None
        
        available_proxies = [
            proxy for proxy in self.proxies
            if self.proxy_failures[proxy.proxy_url] < config.proxy_pool.max_failures_per_proxy
        ]
        
        if not available_proxies:
            # Reset all failure counts if no proxies available
            for proxy_url in self.proxy_failures:
                self.proxy_failures[proxy_url] = 0
            available_proxies = self.proxies
        
        return random.choice(available_proxies)
    
    def mark_proxy_failure(self, proxy: ProxyConfig):
        """Mark proxy as failed"""
        self.proxy_failures[proxy.proxy_url] += 1
        logger.warning(f"Proxy failure: {proxy.host}:{proxy.port} "
                      f"(failures: {self.proxy_failures[proxy.proxy_url]})")
    
    def mark_proxy_success(self, proxy: ProxyConfig):
        """Mark proxy as successful (reset failure count)"""
        self.proxy_failures[proxy.proxy_url] = 0

# User agent management
class UserAgentManager:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
    
    def get_random_agent(self) -> str:
        return random.choice(self.user_agents)
    
    def get_headers_for_agent(self, user_agent: str) -> Dict[str, str]:
        base_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        if 'Chrome' in user_agent:
            base_headers.update({
                'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            })
        
        base_headers['User-Agent'] = user_agent
        return base_headers

# Enhanced SMS service handler (keeping original implementation)
class EnhancedSMSServiceHandler:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.sms_context = None
        self.sms_page = None
        self.session_id = str(uuid.uuid4())
        self.user_agent_manager = UserAgentManager()
        self.message_cache: Dict[str, List[Dict]] = {}
        self.last_refresh = datetime.utcnow()
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PlaywrightTimeout, CaptchaError))
    )
    async def initialize(self):
        try:
            logger.info("Initializing SMS service handler", session_id=self.session_id)
            
            self.playwright = await async_playwright().start()
            
            browser_options = {
                "headless": config.browser.headless,
                "slow_mo": config.browser.slow_mo,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-extensions",
                    "--no-sandbox",
                    "--disable-dev-shm-usage"
                ]
            }
            
            self.browser = await self.playwright.chromium.launch(**browser_options)
            
            user_agent = self.user_agent_manager.get_random_agent()
            headers = self.user_agent_manager.get_headers_for_agent(user_agent)
            
            context_options = {
                "viewport": {
                    "width": config.browser.viewport_width,
                    "height": config.browser.viewport_height
                },
                "user_agent": user_agent,
                "extra_http_headers": headers,
                "locale": "en-US",
                "timezone_id": "America/New_York"
            }
            
            self.sms_context = await self.browser.new_context(**context_options)
            
            if config.browser.stealth_mode:
                await self._apply_enhanced_stealth()
            
            logger.info("Browser initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize browser", error=str(e))
            await self.cleanup()
            raise SMSServiceError(f"Browser initialization failed: {str(e)}")
    
    async def _apply_enhanced_stealth(self):
        stealth_script = """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
        
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {}
        };
        
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        """
        
        await self.sms_context.add_init_script(stealth_script)
    
    def _extract_captcha_numbers(self, captcha_text: str) -> Optional[str]:
        if not captcha_text:
            return None
            
        captcha_text = captcha_text.strip().lower()
        
        patterns = [
            r'what is (\d+) \+ (\d+)',
            r'(\d+) \+ (\d+) = \?',
            r'(\d+) plus (\d+)',
            r'add (\d+) and (\d+)',
            r'(\d+)\s*\+\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, captcha_text)
            if match:
                try:
                    num1, num2 = map(int, match.groups())
                    result = num1 + num2
                    logger.info(f"Solved captcha: {num1} + {num2} = {result}")
                    return str(result)
                except ValueError:
                    continue
        
        numbers = re.findall(r'\d+', captcha_text)
        if len(numbers) >= 2:
            try:
                result = sum(int(num) for num in numbers[:2])
                logger.info(f"Fallback captcha solution: {result}")
                return str(result)
            except ValueError:
                pass
        
        logger.warning("Could not solve captcha", text=captcha_text)
        return "10"
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def login_to_sms_service(self) -> bool:
        try:
            if not self.sms_context:
                raise SMSServiceError("SMS context not initialized")
            
            logger.info("Attempting login to SMS service")
            self.sms_page = await self.sms_context.new_page()
            
            await self.sms_page.goto(
                f"{config.sms_service.base_url}/login",
                timeout=config.sms_service.timeout * 1000,
                wait_until="networkidle"
            )
            
            await self.sms_page.wait_for_selector('input[name="username"]', timeout=15000)
            
            await self.sms_page.fill('input[name="username"]', self.username)
            await self.sms_page.fill('input[name="password"]', self.password)
            
            await self._handle_enhanced_captcha()
            await self._submit_login_form()
            
            if await self._verify_login_success():
                await self.sms_page.goto(
                    f"{config.sms_service.base_url}/client/SMSCDRStats",
                    timeout=config.sms_service.timeout * 1000,
                    wait_until="networkidle"
                )
                logger.info("Successfully logged in and navigated to SMS data page")
                return True
            else:
                raise LoginError("Login verification failed")
                
        except Exception as e:
            logger.error("Login failed", error=str(e))
            if isinstance(e, (PlaywrightTimeout, LoginError)):
                raise
            raise SMSServiceError(f"Login failed: {str(e)}")
    
    async def _handle_enhanced_captcha(self):
        captcha_selectors = [
            'div.wrap-input100',
            '.captcha-container', 
            '#captcha-text',
            'div:has-text("What is")',
            'label:has-text("=")'
        ]
        
        captcha_text = None
        
        for selector in captcha_selectors:
            try:
                element = await self.sms_page.wait_for_selector(selector, timeout=3000)
                if element:
                    text = await element.text_content()
                    if text and ('what is' in text.lower() or '+' in text or '=' in text):
                        captcha_text = text
                        break
            except:
                continue
        
        if not captcha_text:
            page_content = await self.sms_page.content()
            if 'what is' in page_content.lower() or 'captcha' in page_content.lower():
                soup_match = re.search(r'what is \d+ \+ \d+', page_content, re.IGNORECASE)
                if soup_match:
                    captcha_text = soup_match.group()
        
        if captcha_text:
            answer = self._extract_captcha_numbers(captcha_text)
            if answer:
                captcha_input = await self.sms_page.query_selector('input[name="capt"]')
                if captcha_input:
                    await captcha_input.fill(answer)
                    logger.info("Captcha solved and filled", answer=answer)
    
    async def _submit_login_form(self):
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]', 
            'button:has-text("Login")',
            'button:has-text("Sign In")',
            '.login-button',
            '#login-submit'
        ]
        
        for selector in submit_selectors:
            try:
                button = await self.sms_page.wait_for_selector(selector, timeout=3000)
                if button:
                    await button.click()
                    await self.sms_page.wait_for_load_state('networkidle')
                    return
            except:
                continue
                
        raise LoginError("No submit button found")
    
    async def _verify_login_success(self) -> bool:
        current_url = self.sms_page.url
        
        if 'login' not in current_url.lower() or 'dashboard' in current_url.lower():
            return True
        
        error_selectors = [
            '.error-message',
            '.alert-danger', 
            '#error-container',
            'div:has-text("Invalid")',
            'div:has-text("Error")'
        ]
        
        for selector in error_selectors:
            try:
                error_element = await self.sms_page.query_selector(selector)
                if error_element:
                    error_text = await error_element.text_content()
                    if error_text and any(word in error_text.lower() for word in ['invalid', 'error', 'failed']):
                        logger.error("Login error detected", error=error_text)
                        return False
            except:
                continue
        
        success_indicators = ['dashboard', 'welcome', 'sms', 'messages']
        page_content = await self.sms_page.content()
        
        return any(indicator in page_content.lower() for indicator in success_indicators)
    
    async def extract_sms_data_with_caching(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        cache_key = f"{self.session_id}_sms_data"
        
        if not force_refresh and (now - self.last_refresh).seconds < 30:
            if cache_key in self.message_cache:
                logger.debug("Using cached SMS data")
                return self.message_cache[cache_key]
        
        try:
            if not self.sms_page:
                raise SMSServiceError("SMS page not initialized")
            
            await self.sms_page.wait_for_selector('#dt tbody', timeout=15000)
            
            rows = await self.sms_page.query_selector_all('#dt tbody tr')
            sms_data = []
            
            for row in rows:
                try:
                    style = await row.get_attribute('style')
                    if style and 'display: none' in style:
                        continue
                    
                    cells = await row.query_selector_all('td')
                    if len(cells) < 6:
                        continue
                    
                    cell_contents = []
                    for cell in cells:
                        content = await cell.text_content()
                        cell_contents.append(content.strip() if content else '')
                    
                    if not cell_contents[0] or not cell_contents[2]:
                        continue
                    
                    if "0,0,0," in cell_contents[0]:
                        continue
                    
                    message_data = {
                        'date': cell_contents[0],
                        'range': cell_contents[1] if len(cell_contents) > 1 else '',
                        'number': cell_contents[2],
                        'cli': cell_contents[3] if len(cell_contents) > 3 else '',
                        'sms': cell_contents[4] if len(cell_contents) > 4 else '',
                        'currency': cell_contents[5] if len(cell_contents) > 5 else '',
                        'payout': cell_contents[6] if len(cell_contents) > 6 else ''
                    }
                    
                    sms_data.append(message_data)
                    
                except Exception as e:
                    logger.debug(f"Error extracting row data: {e}")
                    continue
            
            self.message_cache[cache_key] = sms_data
            self.last_refresh = now
            
            logger.info(f"Extracted {len(sms_data)} SMS messages")
            return sms_data
            
        except Exception as e:
            logger.error("Failed to extract SMS data", error=str(e))
            raise SMSServiceError(f"SMS data extraction failed: {str(e)}")
    
    async def get_verification_code_intelligent(
        self, 
        phone_number: str, 
        service_type: ServiceType = ServiceType.VALR,
        timeout: int = None
    ) -> Optional[VerificationResult]:
        
        if timeout is None:
            timeout = config.services.verification_timeout
        
        clean_phone = self._normalize_phone_number(phone_number)
        start_time = time.time()
        
        logger.info(f"Looking for {service_type.value} verification code", 
                   phone_number=phone_number, timeout=timeout)
        
        existing_messages = await self.extract_sms_data_with_caching()
        
        for message in existing_messages:
            if self._phone_numbers_match(clean_phone, message['number']):
                if self._is_valr_message(message['cli'], message['sms']):
                    code = self._extract_verification_code_enhanced(message['sms'])
                    if code:
                        logger.info(f"Found existing {service_type.value} code", 
                                  phone_number=phone_number, code=code)
                        return VerificationResult(
                            phone_number=phone_number,
                            service=service_type,
                            success=True,
                            code=code
                        )
        
        check_interval = 10
        max_iterations = timeout // check_interval
        
        for iteration in range(max_iterations):
            await asyncio.sleep(check_interval)
            
            current_messages = await self.extract_sms_data_with_caching(force_refresh=True)
            
            for message in current_messages[len(existing_messages):]:
                if self._phone_numbers_match(clean_phone, message['number']):
                    if self._is_valr_message(message['cli'], message['sms']):
                        code = self._extract_verification_code_enhanced(message['sms'])
                        if code:
                            await self._store_verification_result(
                                phone_number, service_type, code, message
                            )
                            
                            logger.info(f"Found new {service_type.value} code", 
                                      phone_number=phone_number, code=code)
                            return VerificationResult(
                                phone_number=phone_number,
                                service=service_type,
                                success=True,
                                code=code
                            )
            
            existing_messages = current_messages
            
            remaining_time = timeout - (time.time() - start_time)
            logger.debug(f"Still waiting for code, {remaining_time:.0f}s remaining")
        
        logger.warning(f"No {service_type.value} verification code found", 
                      phone_number=phone_number, timeout=timeout)
        
        return VerificationResult(
            phone_number=phone_number,
            service=service_type,
            success=False,
            error="Verification timeout"
        )
    
    def _normalize_phone_number(self, phone_number: str) -> str:
        return re.sub(r'[^\d]', '', phone_number)
    
    def _phone_numbers_match(self, phone1: str, phone2: str) -> bool:
        clean1 = self._normalize_phone_number(phone1)
        clean2 = self._normalize_phone_number(phone2)
        
        if clean1 == clean2:
            return True
        
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        if len(clean1) >= 8 and len(clean2) >= 8:
            if clean1[-8:] == clean2[-8:]:
                return True
        
        if len(clean1) >= 10 and len(clean2) >= 10:
            if clean1[-10:] == clean2[-10:]:
                return True
        
        return False
    
    def _is_valr_message(self, cli: str, sms_content: str) -> bool:
        if not cli and not sms_content:
            return False
        
        valr_indicators = ['valr', 'VALR', 'verification', 'code', 'otp']
        
        # Check CLI
        if cli:
            cli_lower = cli.lower()
            if any(indicator.lower() in cli_lower for indicator in valr_indicators):
                return True
        
        # Check SMS content
        if sms_content:
            sms_lower = sms_content.lower()
            if any(indicator.lower() in sms_lower for indicator in valr_indicators):
                return True
            
            # Generic verification patterns
            verification_patterns = [
                r'verification.*code',
                r'your.*code',
                r'security.*code',
                r'otp.*code',
                r'\d{4,6}.*verify'
            ]
            
            for pattern in verification_patterns:
                if re.search(pattern, sms_lower):
                    return True
        
        return False
    
    def _extract_verification_code_enhanced(self, sms_text: str) -> Optional[str]:
        if not sms_text:
            return None
        
        # VALR specific patterns
        patterns = [
            r'Your VALR verification code is (\d{4,6})',
            r'VALR.*?(\d{4,6})',
            r'Your verification code is (\d{4,6})',
            r'verification code.*?(\d{4,6})',
            r'code is (\d{4,6})',
            r'code: (\d{4,6})',
            r'your code is (\d{4,6})',
            r'(\d{4,6}) is your',
            r'OTP.*?(\d{4,6})',
            r'security code.*?(\d{4,6})',
            r'(\d{6})',  # 6-digit fallback
            r'(\d{4})'   # 4-digit fallback
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sms_text, re.IGNORECASE)
            if match:
                code = match.group(1)
                if 4 <= len(code) <= 6:
                    return code
        
        return None
    
    async def _store_verification_result(
        self, 
        phone_number: str, 
        service_type: ServiceType, 
        code: str, 
        message: Dict[str, Any]
    ):
        try:
            with SessionLocal() as db:
                try:
                    received_at = datetime.strptime(message['date'], '%Y-%m-%d %H:%M:%S')
                except:
                    received_at = datetime.utcnow()
                
                sms_message = SMSMessage(
                    phone_number=phone_number,
                    service_type=service_type.value,
                    verification_code=code,
                    cli=message['cli'],
                    sms_content=message['sms'],
                    status=MessageStatus.RECEIVED,
                    received_at=received_at,
                    processed_at=datetime.utcnow(),
                    session_id=self.session_id
                )
                
                db.add(sms_message)
                
                phone_record = db.query(PhoneNumber).filter(
                    PhoneNumber.number == phone_number
                ).first()
                
                if phone_record:
                    phone_record.success_count += 1
                    phone_record.last_used = datetime.utcnow()
                
                db.commit()
                logger.debug("Stored verification result in database")
                
        except Exception as e:
            logger.error("Failed to store verification result", error=str(e))
    
    async def cleanup(self):
        try:
            if self.sms_page:
                await self.sms_page.close()
            if self.sms_context:
                await self.sms_context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("SMS service handler cleaned up successfully")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))


# Enhanced VALR Automator Class with Proxy Support
class VALRAutomator:
    """Enhanced VALR automation with cookie support and proxy integration"""
    
    def __init__(self, proxy_pool_manager: Optional[ProxyPoolManager] = None):
        self.service_type = ServiceType.VALR
        self.user_agent_manager = UserAgentManager()
        self.proxy_pool_manager = proxy_pool_manager
        self.success_count = 0
        self.failure_count = 0
        
        # Fixed VALR cookies with proper sameSite handling
        self.valr_cookies = [
            {
                "name": "_ga_R7T9PJFPJ7",
                "value": "GS2.1.s1753749108$o1$g1$t1753749634$j58$l0$h0",
                "domain": ".valr.com",
                "path": "/",
                "secure": False,
                "httpOnly": False,
                "sameSite": "Lax"  # Fixed: Use proper sameSite value
            },
            {
                "name": "_gcl_au",
                "value": "1.1.379737871.1753749109",
                "domain": ".valr.com",
                "path": "/",
                "secure": False,
                "httpOnly": False,
                "sameSite": "Lax"
            },
            {
                "name": "_ga",
                "value": "GA1.1.2001278771.1753749109",
                "domain": ".valr.com",
                "path": "/",
                "secure": False,
                "httpOnly": False,
                "sameSite": "Lax"
            },
            {
                "name": "AMP_216eb084f8",
                "value": "JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjI4MGU3MDc3Ny1hZGEyLTQ5ZWUtYTEyOC0zMmI2YTFjOWU4NGUlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzUzNzQ4ODcyODE3JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTc1Mzc0OTYzNDc1MSUyQyUyMmxhc3RFdmVudElkJTIyJTNBNTglMkMlMjJwYWdlQ291bnRlciUyMiUzQTMlN0Q=",
                "domain": ".valr.com",
                "path": "/",
                "secure": False,
                "httpOnly": False,
                "sameSite": "Lax"
            },
            {
                "name": "AMP_MKTG_216eb084f8",
                "value": "JTdCJTdE",
                "domain": ".valr.com",
                "path": "/",
                "secure": False,
                "httpOnly": False,
                "sameSite": "Lax"
            }
        ]
    
    async def request_verification_code(
        self, 
        phone_number: str, 
        country: str = "EG",
        playwright_instance = None
    ) -> VerificationResult:
        """Request VALR verification code with intelligent detection and proxy support"""
        current_proxy = None
        try:
            formatted_number = self._format_phone_number(phone_number, country)
            
            user_agent = self.user_agent_manager.get_random_agent()
            headers = self.user_agent_manager.get_headers_for_agent(user_agent)
            
            # Get proxy if available
            if self.proxy_pool_manager:
                current_proxy = self.proxy_pool_manager.get_random_proxy()
                if current_proxy:
                    logger.info(f"Using proxy: {current_proxy.host}:{current_proxy.port} ({current_proxy.country})")
            
            # Use provided playwright instance or create new one
            if playwright_instance is None:
                playwright_instance = await async_playwright().start()
                should_close_playwright = True
            else:
                should_close_playwright = False
            
            browser_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-features=VizDisplayCompositor",
                "--disable-extensions",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor"
            ]
            
            # Add proxy to browser args if available
            if current_proxy:
                browser_args.append(f"--proxy-server={current_proxy.proxy_url}")
            
            browser = await playwright_instance.chromium.launch(
                headless=config.browser.headless,
                slow_mo=config.browser.slow_mo,
                args=browser_args
            )
            
            context_options = {
                "user_agent": user_agent,
                "extra_http_headers": headers,
                "viewport": {'width': config.browser.viewport_width, 'height': config.browser.viewport_height},
                "locale": "en-US",
                "timezone_id": "Africa/Cairo"  # Egypt timezone
            }
            
            # Add proxy authentication if needed
            if current_proxy:
                context_options["proxy"] = {
                    "server": f"http://{current_proxy.host}:{current_proxy.port}",
                    "username": current_proxy.username,
                    "password": current_proxy.password
                }
            
            context = await browser.new_context(**context_options)
            
            # Add enhanced stealth script
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en', 'ar']
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {},
                    csi: function() {},
                    app: {}
                };
                
                // Hide automation indicators
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """)
            
            page = await context.new_page()
            
            try:
                # Add cookies to the page with proper error handling
                try:
                    await context.add_cookies(self.valr_cookies)
                    logger.info("Successfully added VALR cookies to browser context")
                except Exception as cookie_error:
                    logger.warning(f"Failed to add cookies: {cookie_error}")
                    # Continue without cookies if they fail
                
                # Navigate to VALR phone verification page
                valr_url = f"{config.services.valr_url}?country={country}"
                logger.info(f"Navigating to VALR: {valr_url}")
                
                await page.goto(valr_url, timeout=30000, wait_until="networkidle")
                await asyncio.sleep(2)  # Wait for page to stabilize
                
                # Take screenshot for debugging
                screenshot_path = f"valr_page_{int(time.time())}.png"
                try:
                    await page.screenshot(path=screenshot_path)
                    logger.debug(f"Screenshot saved: {screenshot_path}")
                except:
                    pass  # Don't fail if screenshot fails
                
                # Try multiple strategies to find and fill phone input
                phone_filled = await self._fill_phone_number_intelligent(page, formatted_number, country)
                
                if not phone_filled:
                    raise Exception("Could not fill phone number - input field not found or not accessible")
                
                # Submit the form
                submit_success = await self._submit_verification_form(page)
                
                if not submit_success:
                    raise Exception("Could not submit verification form - submit button not found or not clickable")
                
                # Wait and check for success/error indicators
                await asyncio.sleep(3)
                
                # Check for various success/error indicators
                result = await self._analyze_submission_result(page, phone_number)
                result.proxy_used = f"{current_proxy.host}:{current_proxy.port}" if current_proxy else None
                
                if result.success:
                    self.success_count += 1
                    if current_proxy and self.proxy_pool_manager:
                        self.proxy_pool_manager.mark_proxy_success(current_proxy)
                    logger.info(f"Successfully requested VALR verification for {phone_number}")
                else:
                    self.failure_count += 1
                    if current_proxy and self.proxy_pool_manager:
                        self.proxy_pool_manager.mark_proxy_failure(current_proxy)
                
                return result
                
            finally:
                try:
                    await page.close()
                    await context.close()
                    await browser.close()
                    if should_close_playwright:
                        await playwright_instance.stop()
                except:
                    pass  # Don't fail on cleanup errors
                
        except Exception as e:
            self.failure_count += 1
            if current_proxy and self.proxy_pool_manager:
                self.proxy_pool_manager.mark_proxy_failure(current_proxy)
            logger.error(f"VALR verification request failed for {phone_number}: {e}")
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.VALR,
                success=False,
                error=str(e),
                proxy_used=f"{current_proxy.host}:{current_proxy.port}" if current_proxy else None
            )
    
    async def _fill_phone_number_intelligent(self, page: Page, phone_number: str, country: str) -> bool:
        """Intelligently find and fill phone number input with multiple strategies"""
        
        # Strategy 1: Common phone input selectors
        phone_selectors = [
            'input[type="tel"]',
            'input[name*="phone"]',
            'input[placeholder*="phone"]',
            'input[placeholder*="Phone"]',
            'input[id*="phone"]',
            'input[class*="phone"]',
            'input[data-testid*="phone"]',
            'input[aria-label*="phone"]',
            'input[name*="mobile"]',
            'input[placeholder*="mobile"]'
        ]
        
        for selector in phone_selectors:
            try:
                phone_input = await page.wait_for_selector(selector, timeout=3000)
                if phone_input:
                    # Check if input is visible and enabled
                    is_visible = await phone_input.is_visible()
                    is_enabled = await phone_input.is_enabled()
                    
                    if is_visible and is_enabled:
                        await phone_input.clear()
                        await phone_input.fill(phone_number)
                        
                        # Verify the value was set
                        filled_value = await phone_input.input_value()
                        if filled_value:
                            logger.info(f"Successfully filled phone number using selector: {selector}")
                            return True
            except:
                continue
        
        # Strategy 2: Find by text content (labels, placeholders)
        try:
            # Look for elements containing phone-related text
            phone_texts = ['phone', 'mobile', 'number', 'تليفون', 'رقم']
            
            for text in phone_texts:
                elements = await page.query_selector_all(f'*:has-text("{text}")')
                for element in elements:
                    # Find associated input
                    input_field = await element.query_selector('input')
                    if not input_field:
                        # Check if this element is an input itself
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        if tag_name == 'input':
                            input_field = element
                    
                    if input_field:
                        is_visible = await input_field.is_visible()
                        is_enabled = await input_field.is_enabled()
                        
                        if is_visible and is_enabled:
                            await input_field.clear()
                            await input_field.fill(phone_number)
                            
                            filled_value = await input_field.input_value()
                            if filled_value:
                                logger.info(f"Successfully filled phone number using text search: {text}")
                                return True
        except Exception as e:
            logger.debug(f"Text-based search failed: {e}")
        
        # Strategy 3: Try all visible input fields
        try:
            all_inputs = await page.query_selector_all('input')
            for input_field in all_inputs:
                try:
                    input_type = await input_field.get_attribute('type')
                    is_visible = await input_field.is_visible()
                    is_enabled = await input_field.is_enabled()
                    
                    # Skip if not suitable for phone number
                    if input_type in ['hidden', 'submit', 'button', 'checkbox', 'radio']:
                        continue
                    
                    if is_visible and is_enabled:
                        placeholder = await input_field.get_attribute('placeholder') or ''
                        name = await input_field.get_attribute('name') or ''
                        
                        # Check if this looks like a phone input
                        if any(keyword in (placeholder + name).lower() for keyword in ['phone', 'mobile', 'tel', 'number']):
                            await input_field.clear()
                            await input_field.fill(phone_number)
                            
                            filled_value = await input_field.input_value()
                            if filled_value:
                                logger.info("Successfully filled phone number using fallback strategy")
                                return True
                except:
                    continue
        except Exception as e:
            logger.debug(f"Fallback input search failed: {e}")
        
        logger.warning("Could not find suitable phone input field")
        return False
    
    async def _submit_verification_form(self, page: Page) -> bool:
        """Intelligently find and click submit button"""
        
        # Strategy 1: Common submit button selectors
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Send")',
            'button:has-text("Submit")',
            'button:has-text("Verify")',
            'button:has-text("Continue")',
            'button:has-text("Next")',
            'button:has-text("Send Code")',
            'button:has-text("Get Code")',
            'button[data-testid*="submit"]',
            'button[data-testid*="send"]',
            'button[data-testid*="verify"]',
            '.submit-button',
            '.send-button',
            '.verify-button'
        ]
        
        for selector in submit_selectors:
            try:
                button = await page.wait_for_selector(selector, timeout=3000)
                if button:
                    is_visible = await button.is_visible()
                    is_enabled = await button.is_enabled()
                    
                    if is_visible and is_enabled:
                        await button.click()
                        logger.info(f"Successfully clicked submit button using selector: {selector}")
                        await page.wait_for_load_state('networkidle', timeout=10000)
                        return True
            except:
                continue
        
        # Strategy 2: Find buttons by text content
        button_texts = ['send', 'submit', 'verify', 'continue', 'next', 'get code', 'إرسال']
        
        for text in button_texts:
            try:
                buttons = await page.query_selector_all(f'button:has-text("{text}")')
                for button in buttons:
                    is_visible = await button.is_visible()
                    is_enabled = await button.is_enabled()
                    
                    if is_visible and is_enabled:
                        await button.click()
                        logger.info(f"Successfully clicked submit button using text: {text}")
                        await page.wait_for_load_state('networkidle', timeout=10000)
                        return True
            except:
                continue
        
        # Strategy 3: Try form submission via Enter key
        try:
            phone_input = await page.query_selector('input[type="tel"], input[name*="phone"]')
            if phone_input:
                await phone_input.press('Enter')
                logger.info("Attempted form submission via Enter key")
                await page.wait_for_load_state('networkidle', timeout=10000)
                return True
        except:
            pass
        
        logger.warning("Could not find or click submit button")
        return False
    
    async def _analyze_submission_result(self, page: Page, phone_number: str) -> VerificationResult:
        """Analyze the result of form submission"""
        
        try:
            # Get current page content
            page_content = await page.content()
            current_url = page.url
            
            # Take screenshot for debugging
            screenshot_path = f"valr_result_{int(time.time())}.png"
            try:
                await page.screenshot(path=screenshot_path)
                logger.debug(f"Result screenshot saved: {screenshot_path}")
            except:
                pass
            
            # Check for success indicators
            success_indicators = [
                'code sent',
                'verification code',
                'sms sent',
                'check your phone',
                'enter the code',
                'verify your phone',
                'otp sent',
                'message sent'
            ]
            
            # Check for error indicators
            error_indicators = [
                'error',
                'invalid',
                'failed',
                'not valid',
                'incorrect',
                'blocked',
                'try again',
                'خطأ',
                'غير صحيح'
            ]
            
            page_content_lower = page_content.lower()
            
            # Check for success
            for indicator in success_indicators:
                if indicator in page_content_lower:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=True,
                        error=None
                    )
            
            # Check for specific error messages
            for indicator in error_indicators:
                if indicator in page_content_lower:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=False,
                        error=f"Error detected: {indicator}"
                    )
            
            # Check if URL changed (might indicate success)
            if 'verify' in current_url or 'code' in current_url:
                return VerificationResult(
                    phone_number=phone_number,
                    service=ServiceType.VALR,
                    success=True,
                    error=None
                )
            
            # Check for input fields that might indicate next step
            try:
                code_inputs = await page.query_selector_all('input[type="text"], input[type="number"], input[maxlength="6"], input[maxlength="4"]')
                if code_inputs:
                    # Look for code input patterns
                    for input_field in code_inputs:
                        placeholder = await input_field.get_attribute('placeholder') or ''
                        name = await input_field.get_attribute('name') or ''
                        
                        if any(keyword in (placeholder + name).lower() for keyword in ['code', 'otp', 'verify', 'pin']):
                            return VerificationResult(
                                phone_number=phone_number,
                                service=ServiceType.VALR,
                                success=True,
                                error=None
                            )
            except:
                pass
            
            # Default to success if no clear error indicators
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.VALR,
                success=True,
                error=None
            )
            
        except Exception as e:
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.VALR,
                success=False,
                error=f"Analysis failed: {str(e)}"
            )
    
    def _format_phone_number(self, phone_number: str, country: str) -> str:
        """Format phone number for VALR (Egypt focused)"""
        
        # Clean phone number
        clean_number = re.sub(r'[^\d]', '', phone_number)
        
        # Egypt country code handling
        if country.upper() == 'EG':
            # Remove Egypt country code if present
            if clean_number.startswith('20'):
                clean_number = clean_number[2:]
            
            # Ensure it starts with proper Egyptian mobile prefix
            egyptian_prefixes = ['10', '11', '12', '15']
            
            if not any(clean_number.startswith(prefix) for prefix in egyptian_prefixes):
                # If it doesn't start with a valid prefix, assume it's missing and add '10'
                if len(clean_number) == 8:
                    clean_number = '10' + clean_number
            
            # Return with country code
            return f"+20{clean_number}"
        
        # Default country code handling
        country_codes = {
            'US': '+1',
            'UK': '+44',
            'SA': '+966',
            'AE': '+971',
            'KW': '+965'
        }
        
        country_code = country_codes.get(country.upper(), '+20')  # Default to Egypt
        
        # Remove country code if already present
        code_digits = country_code[1:]
        if clean_number.startswith(code_digits):
            clean_number = clean_number[len(code_digits):]
        
        return f"{country_code}{clean_number}"


# Enhanced Verification Orchestrator with Proxy Support
class EnhancedVerificationOrchestrator:
    """Orchestrates the entire VALR verification process with proxy support"""
    
    def __init__(self, proxy_strings: List[str] = None):
        self.sms_handler = None
        self.proxy_pool_manager = None
        self.valr_automator = None
        self.results: List[VerificationResult] = []
        self.session_id = str(uuid.uuid4())
        
        # Initialize proxy pool if proxy strings provided
        if proxy_strings:
            self.proxy_pool_manager = ProxyPoolManager(proxy_strings)
            logger.info(f"Initialized with {len(proxy_strings)} proxies")
        
        # Initialize VALR automator with proxy support
        self.valr_automator = VALRAutomator(self.proxy_pool_manager)
        
    async def initialize(self, username: str, password: str):
        """Initialize the orchestrator"""
        self.sms_handler = EnhancedSMSServiceHandler(username, password)
        await self.sms_handler.__aenter__()
        
        # Login to SMS service
        if not await self.sms_handler.login_to_sms_service():
            raise LoginError("Failed to login to SMS service")
        
        logger.info("Enhanced VALR verification orchestrator initialized successfully")
    
    async def process_phone_numbers(
        self, 
        phone_data: List[Dict[str, str]]
    ) -> List[VerificationResult]:
        """Process multiple phone numbers for VALR verification with proxy rotation"""
        
        results = []
        
        async with async_playwright() as playwright:
            # Process numbers with controlled concurrency
            semaphore = asyncio.Semaphore(2)  # Limit concurrent requests
            
            tasks = [
                self._process_single_number(
                    phone_info, playwright, semaphore
                )
                for phone_info in phone_data
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                else:
                    results.append(result)
                    self.results.append(result)
        
        return results
    
    async def _process_single_number(
        self,
        phone_info: Dict[str, str],
        playwright_instance,
        semaphore: asyncio.Semaphore
    ) -> VerificationResult:
        """Process a single phone number with proxy support"""
        
        async with semaphore:
            phone_number = phone_info.get('Number', phone_info.get('phone', ''))
            country = phone_info.get('Range', phone_info.get('Country', phone_info.get('country', 'EG')))
            
            try:
                logger.info(f"Processing {phone_number} for VALR verification")
                
                # Request verification code from VALR with proxy support
                request_result = await self.valr_automator.request_verification_code(
                    phone_number, country, playwright_instance
                )
                
                if not request_result.success:
                    return request_result
                
                # Wait for SMS verification code
                verification_result = await self.sms_handler.get_verification_code_intelligent(
                    phone_number, ServiceType.VALR
                )
                
                if verification_result and verification_result.success:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=True,
                        code=verification_result.code,
                        proxy_used=request_result.proxy_used
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {phone_number}: {e}")
                return VerificationResult(
                    phone_number=phone_number,
                    service=ServiceType.VALR,
                    success=False,
                    error=str(e)
                )
            finally:
                # Rate limiting
                delay = random.uniform(*config.services.rate_limit_delay)
                await asyncio.sleep(delay)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report with proxy statistics"""
        if not self.results:
            return {'status': 'No results to report'}
        
        total_attempts = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_attempts - successful
        success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0
        
        # Error analysis
        error_types = {}
        for result in self.results:
            if not result.success and result.error:
                error_type = result.error[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Proxy statistics
        proxy_stats = {}
        for result in self.results:
            if result.proxy_used:
                proxy_stats[result.proxy_used] = proxy_stats.get(result.proxy_used, {'total': 0, 'success': 0})
                proxy_stats[result.proxy_used]['total'] += 1
                if result.success:
                    proxy_stats[result.proxy_used]['success'] += 1
        
        # Calculate proxy success rates
        for proxy, stats in proxy_stats.items():
            stats['success_rate'] = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        # Successful codes
        successful_codes = [
            {
                'phone_number': r.phone_number,
                'service': r.service.value,
                'code': r.code,
                'timestamp': r.timestamp.isoformat(),
                'proxy_used': r.proxy_used
            }
            for r in self.results if r.success and r.code
        ]
        
        return {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'success_rate': round(success_rate, 2)
            },
            'error_analysis': error_types,
            'proxy_statistics': proxy_stats,
            'successful_verifications': successful_codes
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sms_handler:
            await self.sms_handler.__aexit__(None, None, None)


# CSV Processing (keeping original implementation)
class CSVProcessor:
    """Enhanced CSV processing"""
    
    @staticmethod
    def load_and_validate_csv(file_path: str) -> pd.DataFrame:
        """Load and validate CSV file"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            # Find phone number column
            phone_columns = ['Number', 'Phone', 'Mobile', 'phone', 'number', 'mobile']
            phone_col = None
            
            for col in phone_columns:
                if col in df.columns:
                    phone_col = col
                    break
            
            if not phone_col:
                # Try to find column with phone-like content
                for col in df.columns:
                    sample_value = str(df[col].iloc[0]) if len(df) > 0 else ""
                    if re.search(r'\d{8,}', sample_value):
                        phone_col = col
                        break
            
            if not phone_col:
                raise ValueError("No phone number column found")
            
            # Rename to standard name
            if phone_col != 'Number':
                df = df.rename(columns={phone_col: 'Number'})
            
            # Add default country if not present
            if 'Country' not in df.columns and 'Range' not in df.columns:
                df['Country'] = 'EG'  # Default to Egypt
            
            # Clean phone numbers
            df['Number'] = df['Number'].astype(str).str.strip()
            df = df[df['Number'].str.len() > 5]
            df = df.dropna(subset=['Number'])
            df = df.drop_duplicates(subset=['Number'])
            
            logger.info(f"Loaded {len(df)} valid phone numbers from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def find_csv_files(directory: str = ".") -> List[str]:
        """Find CSV files in directory"""
        patterns = ["*.csv"]
        found_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            found_files.extend(files)
        
        unique_files = list(set(found_files))
        unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return unique_files


# Enhanced logging setup
def setup_enhanced_logging():
    """Setup enhanced logging"""
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )

# Initialize logging
setup_enhanced_logging()
logger = structlog.get_logger()


# CLI interface with proxy support
def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with proxy support"""
    parser = argparse.ArgumentParser(
        description="VALR SMS Verification Automation Tool with Proxy Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv phones.csv
  %(prog)s --test-number +201234567890
  %(prog)s --display-messages
  %(prog)s --csv phones.csv --proxy-file proxies.txt
  %(prog)s --test-number +201234567890 --proxy "163.172.65.133:48103:6c65:6c65|United Kingdom"
        """
    )
    
    # Authentication
    auth_group = parser.add_argument_group('Authentication')
    auth_group.add_argument('--username', type=str, help='SMS service username')
    auth_group.add_argument('--password', type=str, help='SMS service password')
    
    # Input data
    data_group = parser.add_argument_group('Input Data')
    data_group.add_argument('--csv', type=str, help='Path to CSV file with phone numbers')
    data_group.add_argument('--test-number', type=str, help='Single phone number to test')
    
    # Proxy options
    proxy_group = parser.add_argument_group('Proxy Options')
    proxy_group.add_argument('--proxy', type=str, help='Single proxy string (host:port:user:pass|country|...)')
    proxy_group.add_argument('--proxy-file', type=str, help='File containing proxy strings (one per line)')
    proxy_group.add_argument('--no-proxy', action='store_true', help='Disable proxy usage')
    
    # Actions
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument(
        '--action', 
        type=str, 
        choices=['verify', 'display-messages', 'test-single'], 
        default='verify',
        help='Action to perform'
    )
    
    # Browser options
    browser_group = parser.add_argument_group('Browser Options')
    browser_group.add_argument('--headless', action='store_true', help='Run browsers in headless mode')
    browser_group.add_argument('--slow-mo', type=int, default=100, help='Slow motion delay in ms')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-format', choices=['json', 'csv', 'table'], default='table', help='Output format')
    output_group.add_argument('--output-file', type=str, help='Output file path')
    output_group.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser


def load_proxy_strings(proxy_file: str) -> List[str]:
    """Load proxy strings from file"""
    try:
        with open(proxy_file, 'r') as f:
            proxy_strings = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(proxy_strings)} proxies from {proxy_file}")
        return proxy_strings
    except Exception as e:
        logger.error(f"Failed to load proxy file: {e}")
        return []


def display_results_table(results: List[VerificationResult]):
    """Display results in a formatted table with proxy information"""
    table = Table(title="VALR Verification Results")
    
    table.add_column("Phone Number", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Code", style="yellow")
    table.add_column("Proxy", style="magenta")
    table.add_column("Error", style="red")
    table.add_column("Timestamp", style="blue")
    
    for result in results:
        status = "✅ Success" if result.success else "❌ Failed"
        code = result.code or "-"
        proxy = result.proxy_used or "-"
        error = result.error[:40] + "..." if result.error and len(result.error) > 40 else (result.error or "-")
        timestamp = result.timestamp.strftime("%H:%M:%S") if result.timestamp else "-"
        
        table.add_row(
            result.phone_number,
            status,
            code,
            proxy,
            error,
            timestamp
        )
    
    console.print(table)


def save_results(results: List[VerificationResult], format: str, file_path: str = None):
    """Save results in specified format with proxy information"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'json':
        file_path = file_path or f'valr_verification_results_{timestamp}.json'
        data = [
            {
                'phone_number': r.phone_number,
                'service': r.service.value,
                'success': r.success,
                'code': r.code,
                'proxy_used': r.proxy_used,
                'error': r.error,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None
            }
            for r in results
        ]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    elif format == 'csv':
        file_path = file_path or f'valr_verification_results_{timestamp}.csv'
        data = []
        for r in results:
            data.append({
                'phone_number': r.phone_number,
                'service': r.service.value,
                'success': r.success,
                'code': r.code,
                'proxy_used': r.proxy_used,
                'error': r.error,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    console.print(f"[green]Results saved to {file_path}[/]")


# Main execution function with proxy support
async def main():
    """Enhanced main function with proxy support"""
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Update config with CLI arguments
        if args.headless:
            config.browser.headless = True
        if args.slow_mo:
            config.browser.slow_mo = args.slow_mo
        
        # Set verbose logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load proxy strings
        proxy_strings = []
        if not args.no_proxy:
            if args.proxy:
                proxy_strings = [args.proxy]
            elif args.proxy_file:
                proxy_strings = load_proxy_strings(args.proxy_file)
            else:
                # Try to find default proxy file
                default_proxy_files = ['proxies.txt', 'proxy.txt', 'proxy_list.txt']
                for file_name in default_proxy_files:
                    if os.path.exists(file_name):
                        proxy_strings = load_proxy_strings(file_name)
                        break
        
        # Add default proxy from example if none provided
        if not proxy_strings and not args.no_proxy:
            proxy_strings = ["163.172.65.133:48103:6c65:6c65|United Kingdom|England|Huddersfield"]
            console.print("[yellow]Using default proxy from example[/]")
        
        # Get credentials
        username = args.username or config.sms_service.username or os.getenv('SMS_USERNAME')
        password = args.password or config.sms_service.password or os.getenv('SMS_PASSWORD')
        
        if not username or not password:
            console.print("[red]Error: SMS service credentials not provided[/]")
            console.print("Use --username and --password, or set SMS_USERNAME and SMS_PASSWORD environment variables")
            return 1
        
        # Display header
        console.rule("[bold blue]Enhanced VALR SMS Verification Automation[/]")
        console.print(f"[bold]Session ID:[/] {uuid.uuid4()}")
        console.print(f"[bold]Action:[/] {args.action}")
        console.print(f"[bold]Proxies:[/] {len(proxy_strings)} loaded" if proxy_strings else "[bold]Proxies:[/] Disabled")
        
        # Initialize orchestrator with proxy support
        orchestrator = EnhancedVerificationOrchestrator(proxy_strings)
        
        try:
            await orchestrator.initialize(username, password)
            
            if args.action == 'display-messages':
                # Display existing messages
                console.print("\n[bold yellow]Displaying existing SMS messages...[/]")
                messages = await orchestrator.sms_handler.extract_sms_data_with_caching()
                
                if messages:
                    table = Table(title="Recent SMS Messages")
                    table.add_column("Date", style="cyan")
                    table.add_column("Number", style="green")
                    table.add_column("CLI", style="yellow")
                    table.add_column("SMS Content", style="white")
                    
                    for msg in messages[-20:]:  # Show last 20 messages
                        table.add_row(
                            msg.get('date', ''),
                            msg.get('number', ''),
                            msg.get('cli', ''),
                            msg.get('sms', '')[:100] + "..." if len(msg.get('sms', '')) > 100 else msg.get('sms', '')
                        )
                    
                    console.print(table)
                else:
                    console.print("[yellow]No SMS messages found[/]")
                
            elif args.action == 'test-single':
                if not args.test_number:
                    console.print("[red]Error: --test-number required for test-single action[/]")
                    return 1
                
                console.print(f"\n[bold yellow]Testing single number: {args.test_number}[/]")
                
                phone_data = [{'Number': args.test_number, 'Country': 'EG'}]
                
                results = await orchestrator.process_phone_numbers(phone_data)
                
                # Display results
                display_results_table(results)
                
            elif args.action == 'verify':
                # Load phone numbers
                if args.csv:
                    csv_file = args.csv
                elif args.test_number:
                    # Create temporary data for single number
                    phone_data = [{'Number': args.test_number, 'Country': 'EG'}]
                else:
                    # Find CSV file automatically
                    csv_files = CSVProcessor.find_csv_files()
                    if not csv_files:
                        console.print("[red]Error: No CSV file found. Use --csv to specify file path[/]")
                        return 1
                    csv_file = csv_files[0]
                    console.print(f"[yellow]Using CSV file: {csv_file}[/]")
                
                if not args.test_number:
                    # Load from CSV
                    df = CSVProcessor.load_and_validate_csv(csv_file)
                    phone_data = df.to_dict('records')
                
                console.print(f"\n[bold green]Processing {len(phone_data)} phone numbers for VALR verification...[/]")
                
                # Process phone numbers
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing phone numbers...", total=len(phone_data))
                    
                    results = await orchestrator.process_phone_numbers(phone_data)
                    progress.update(task, completed=len(phone_data))
                
                # Display results
                console.print("\n[bold blue]Results Summary[/]")
                if args.output_format == 'table':
                    display_results_table(results)
                
                # Generate and display report
                report = orchestrator.generate_report()
                
                console.print(f"\n[bold green]Success Rate: {report['summary']['success_rate']:.1f}%[/]")
                console.print(f"[bold]Total Attempts:[/] {report['summary']['total_attempts']}")
                console.print(f"[bold]Successful:[/] {report['summary']['successful']}")
                console.print(f"[bold]Failed:[/] {report['summary']['failed']}")
                
                # Display proxy statistics if available
                if report.get('proxy_statistics'):
                    console.print(f"\n[bold magenta]Proxy Performance:[/]")
                    proxy_table = Table()
                    proxy_table.add_column("Proxy", style="cyan")
                    proxy_table.add_column("Total", style="blue")
                    proxy_table.add_column("Success", style="green")
                    proxy_table.add_column("Success Rate", style="yellow")
                    
                    for proxy, stats in report['proxy_statistics'].items():
                        proxy_table.add_row(
                            proxy,
                            str(stats['total']),
                            str(stats['success']),
                            f"{stats['success_rate']:.1f}%"
                        )
                    
                    console.print(proxy_table)
                
                # Display successful codes
                if report['successful_verifications']:
                    console.print(f"\n[bold green]Successful Verification Codes:[/]")
                    for verification in report['successful_verifications']:
                        proxy_info = f" (via {verification['proxy_used']})" if verification['proxy_used'] else ""
                        console.print(f"[green]{verification['phone_number']}: {verification['code']}{proxy_info}[/]")
                
                # Save results
                if args.output_file or args.output_format != 'table':
                    save_results(results, args.output_format, args.output_file)
                
                # Save report
                report_file = f"valr_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                console.print(f"[green]Detailed report saved to {report_file}[/]")
            
        finally:
            await orchestrator.cleanup()
        
        console.rule("[bold green]Process Completed Successfully[/]")
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/]")
        logger.error("Fatal error occurred", error=str(e), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
# VALR SMS Verification Automation Tool with Proxy Support
# This script automates the SMS verification process for VALR with enhanced features and proxy support.
# It includes intelligent SMS handling, proxy rotation, and comprehensive reporting.
# Author: Ibrahim Ahmed
# Version: 1.0.0
# License: MIT
# Source: https://github.com/ibrahim-ahmed/valr_sms_verification
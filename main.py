import asyncio
import pandas as pd
import re
import time
import os
import glob
from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
import json
import argparse
import functools
from tqdm import tqdm
import random
import string
import traceback
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.orm import sessionmaker, Session
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
    UBER = "uber"
    AIRBNB = "airbnb"
    LYFT = "lyft"
    DOORDASH = "doordash"

class MessageStatus(str, Enum):
    PENDING = "pending"
    RECEIVED = "received"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class VerificationResult:
    phone_number: str
    service: ServiceType
    success: bool
    code: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

# Enhanced configuration with better validation
class SMSServiceConfig(BaseModel):
    username: str = Field(..., description="SMS service username")
    password: str = Field(..., description="SMS service password")
    base_url: str = Field("http://91.232.105.47/ints", description="Base URL for SMS service")
    timeout: int = Field(30, description="Timeout for SMS service requests", ge=10, le=300)
    max_retries: int = Field(3, description="Maximum number of retries", ge=1, le=10)
    session_timeout: int = Field(3600, description="Session timeout in seconds", ge=300)

class ServiceConfig(BaseModel):
    signup_urls: Dict[ServiceType, str] = Field(default={
        ServiceType.UBER: "https://auth.uber.com/login/",
        ServiceType.AIRBNB: "https://www.airbnb.com/signup_login",
        ServiceType.LYFT: "https://account.lyft.com/signup",
        ServiceType.DOORDASH: "https://identity.doordash.com/auth/user/signup"
    })
    max_attempts_per_number: int = Field(3, description="Maximum attempts per phone number", ge=1, le=10)
    wait_between_attempts: int = Field(5, description="Wait time between attempts", ge=1, le=60)
    verification_timeout: int = Field(180, description="Timeout for verification code", ge=30, le=600)
    rate_limit_delay: Tuple[float, float] = Field((2.0, 8.0), description="Random delay range between requests")

class ProxyConfig(BaseModel):
    enabled: bool = Field(False, description="Whether to use proxies")
    proxy_list: List[str] = Field(default=[], description="List of proxy servers")
    test_url: str = Field("https://httpbin.org/ip", description="URL to test proxies")
    test_timeout: int = Field(15, description="Timeout for proxy testing", ge=5, le=60)
    rotate_every: int = Field(5, description="Number of requests before rotating proxy", ge=1, le=50)
    health_check_interval: int = Field(300, description="Proxy health check interval in seconds")

class DatabaseConfig(BaseModel):
    url: str = Field("sqlite:///enhanced_sms.db", description="Database URL")
    echo: bool = Field(False, description="Echo SQL statements")
    pool_size: int = Field(20, description="Connection pool size", ge=5, le=100)
    pool_timeout: int = Field(30, description="Pool timeout", ge=10, le=300)

class BrowserConfig(BaseModel):
    headless: bool = Field(True, description="Run browser in headless mode")
    slow_mo: int = Field(100, description="Slow motion delay", ge=0, le=5000)
    viewport_width: int = Field(1920, description="Browser viewport width", ge=800, le=3840)
    viewport_height: int = Field(1080, description="Browser viewport height", ge=600, le=2160)
    user_agent_rotation: bool = Field(True, description="Rotate user agents")
    stealth_mode: bool = Field(True, description="Enable stealth mode")

class ConcurrencyConfig(BaseModel):
    max_workers: int = Field(3, description="Maximum concurrent workers", ge=1, le=20)
    batch_size: int = Field(10, description="Batch size for processing", ge=1, le=100)
    queue_timeout: int = Field(300, description="Queue timeout in seconds", ge=60, le=1800)

class Config(BaseModel):
    sms_service: SMSServiceConfig
    services: ServiceConfig
    proxy: ProxyConfig
    database: DatabaseConfig
    browser: BrowserConfig
    concurrency: ConcurrencyConfig
    
    @classmethod
    def load_from_file(cls, path: str = "config.json") -> "Config":
        """Load configuration with better error handling and defaults"""
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
        """Create default configuration"""
        return cls(
            sms_service=SMSServiceConfig(
                username=os.getenv("SMS_USERNAME", ""),
                password=os.getenv("SMS_PASSWORD", "")
            ),
            services=ServiceConfig(),
            proxy=ProxyConfig(),
            database=DatabaseConfig(),
            browser=BrowserConfig(),
            concurrency=ConcurrencyConfig()
        )
    
    def save_to_file(self, path: str = "config.json"):
        """Save configuration to file"""
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
    success_rate = Column(Float, default=0.0)
    
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

class ProxyServer(Base):
    __tablename__ = "proxy_servers"
    
    id = Column(Integer, primary_key=True, index=True)
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    username = Column(String, nullable=True)
    password = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    success_rate = Column(Float, default=1.0)
    last_tested = Column(DateTime, nullable=True)
    response_time = Column(Float, nullable=True)

# Enhanced database setup
engine = create_engine(
    config.database.url,
    echo=config.database.echo,
    pool_size=config.database.pool_size,
    pool_timeout=config.database.pool_timeout,
    pool_pre_ping=True
)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Enhanced exception handling
class SMSServiceError(Exception):
    """Base exception for SMS service errors"""
    pass

class ProxyError(SMSServiceError):
    """Proxy-related errors"""
    pass

class CaptchaError(SMSServiceError):
    """Captcha-related errors"""
    pass

class LoginError(SMSServiceError):
    """Login-related errors"""
    pass

class RateLimitError(SMSServiceError):
    """Rate limit errors"""
    pass

class VerificationTimeoutError(SMSServiceError):
    """Verification timeout errors"""
    pass

# Enhanced user agent and header management
class UserAgentManager:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
        self.current_index = 0
    
    def get_random_agent(self) -> str:
        return random.choice(self.user_agents)
    
    def get_next_agent(self) -> str:
        agent = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return agent
    
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

# Enhanced proxy management
class ProxyManager:
    def __init__(self):
        self.proxies: List[Dict[str, Any]] = []
        self.current_index = 0
        self.health_status: Dict[str, bool] = {}
        self.response_times: Dict[str, float] = {}
        self.last_health_check = datetime.utcnow()
        
    async def load_proxies_from_db(self):
        """Load proxies from database"""
        with SessionLocal() as db:
            proxy_records = db.query(ProxyServer).filter(ProxyServer.is_active == True).all()
            self.proxies = [
                {
                    'host': p.host,
                    'port': p.port,
                    'username': p.username,
                    'password': p.password,
                    'success_rate': p.success_rate
                }
                for p in proxy_records
            ]
    
    async def test_proxy(self, proxy: Dict[str, Any]) -> Tuple[bool, float]:
        """Test a single proxy"""
        proxy_url = f"http://{proxy['host']}:{proxy['port']}"
        
        try:
            connector = aiohttp.TCPConnector()
            proxy_auth = None
            
            if proxy.get('username'):
                proxy_auth = aiohttp.BasicAuth(proxy['username'], proxy['password'])
            
            start_time = time.time()
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    config.proxy.test_url,
                    proxy=proxy_url,
                    proxy_auth=proxy_auth,
                    timeout=aiohttp.ClientTimeout(total=config.proxy.test_timeout)
                ) as response:
                    if response.status == 200:
                        response_time = time.time() - start_time
                        return True, response_time
                    
        except Exception as e:
            logger.debug(f"Proxy test failed for {proxy_url}: {e}")
            
        return False, float('inf')
    
    async def health_check(self):
        """Perform health check on all proxies"""
        if not self.proxies:
            return
            
        now = datetime.utcnow()
        if (now - self.last_health_check).seconds < config.proxy.health_check_interval:
            return
            
        logger.info("Performing proxy health check...")
        
        with Progress() as progress:
            task = progress.add_task("Testing proxies...", total=len(self.proxies))
            
            for proxy in self.proxies:
                proxy_key = f"{proxy['host']}:{proxy['port']}"
                is_healthy, response_time = await self.test_proxy(proxy)
                
                self.health_status[proxy_key] = is_healthy
                self.response_times[proxy_key] = response_time
                
                # Update database
                with SessionLocal() as db:
                    db_proxy = db.query(ProxyServer).filter(
                        ProxyServer.host == proxy['host'],
                        ProxyServer.port == proxy['port']
                    ).first()
                    
                    if db_proxy:
                        db_proxy.is_active = is_healthy
                        db_proxy.response_time = response_time
                        db_proxy.last_tested = now
                        db.commit()
                
                progress.advance(task)
        
        self.last_health_check = now
        healthy_count = sum(1 for status in self.health_status.values() if status)
        logger.info(f"Proxy health check complete: {healthy_count}/{len(self.proxies)} healthy")
    
    def get_best_proxy(self) -> Optional[Dict[str, Any]]:
        """Get the best performing healthy proxy"""
        if not self.proxies:
            return None
            
        healthy_proxies = []
        for proxy in self.proxies:
            proxy_key = f"{proxy['host']}:{proxy['port']}"
            if self.health_status.get(proxy_key, True):  # Assume healthy if not tested
                proxy_with_metrics = proxy.copy()
                proxy_with_metrics['response_time'] = self.response_times.get(proxy_key, 1.0)
                healthy_proxies.append(proxy_with_metrics)
        
        if not healthy_proxies:
            return None
            
        # Sort by success rate and response time
        healthy_proxies.sort(key=lambda p: (-p['success_rate'], p['response_time']))
        return healthy_proxies[0]

# Enhanced SMS service handler
class EnhancedSMSServiceHandler:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.sms_context = None
        self.sms_page = None
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.utcnow()
        self.user_agent_manager = UserAgentManager()
        self.proxy_manager = ProxyManager()
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
        retry=retry_if_exception_type((PlaywrightTimeout, ProxyError, CaptchaError))
    )
    async def initialize(self):
        """Initialize with enhanced error handling and proxy support"""
        try:
            logger.info("Initializing SMS service handler", session_id=self.session_id)
            
            self.playwright = await async_playwright().start()
            
            # Load and test proxies if enabled
            if config.proxy.enabled:
                await self.proxy_manager.load_proxies_from_db()
                await self.proxy_manager.health_check()
            
            # Browser launch options
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
            
            # Add proxy if enabled and available
            if config.proxy.enabled:
                best_proxy = self.proxy_manager.get_best_proxy()
                if best_proxy:
                    proxy_config = {
                        "server": f"http://{best_proxy['host']}:{best_proxy['port']}"
                    }
                    if best_proxy.get('username'):
                        proxy_config.update({
                            "username": best_proxy['username'],
                            "password": best_proxy['password']
                        })
                    browser_options["proxy"] = proxy_config
                    logger.info("Using proxy", proxy=f"{best_proxy['host']}:{best_proxy['port']}")
            
            self.browser = await self.playwright.chromium.launch(**browser_options)
            
            # Create context with enhanced stealth
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
                "timezone_id": "America/New_York",
                "permissions": ["geolocation"],
                "geolocation": {"latitude": 40.7128, "longitude": -74.0060}
            }
            
            self.sms_context = await self.browser.new_context(**context_options)
            
            # Apply stealth scripts
            if config.browser.stealth_mode:
                await self._apply_enhanced_stealth()
            
            logger.info("Browser initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize browser", error=str(e))
            await self.cleanup()
            raise SMSServiceError(f"Browser initialization failed: {str(e)}")
    
    async def _apply_enhanced_stealth(self):
        """Apply comprehensive stealth measures"""
        stealth_script = """
        // Remove webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        
        // Mock languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        
        // Mock plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });
        
        // Mock chrome object
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {}
        };
        
        // Mock permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Override the `plugins` property to use a custom getter.
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Pass the Webdriver test
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        """
        
        await self.sms_context.add_init_script(stealth_script)
    
    def _extract_captcha_numbers(self, captcha_text: str) -> Optional[str]:
        """Enhanced captcha solving with multiple patterns"""
        if not captcha_text:
            return None
            
        # Clean the text
        captcha_text = captcha_text.strip().lower()
        
        # Multiple patterns for different captcha formats
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
        
        # Fallback to simple number extraction
        numbers = re.findall(r'\d+', captcha_text)
        if len(numbers) >= 2:
            try:
                result = sum(int(num) for num in numbers[:2])
                logger.info(f"Fallback captcha solution: {result}")
                return str(result)
            except ValueError:
                pass
        
        logger.warning("Could not solve captcha", text=captcha_text)
        return "10"  # Conservative fallback
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def login_to_sms_service(self) -> bool:
        """Enhanced login with better error handling"""
        try:
            if not self.sms_context:
                raise SMSServiceError("SMS context not initialized")
            
            logger.info("Attempting login to SMS service")
            self.sms_page = await self.sms_context.new_page()
            
            # Navigate to login page
            await self.sms_page.goto(
                f"{config.sms_service.base_url}/login",
                timeout=config.sms_service.timeout * 1000,
                wait_until="networkidle"
            )
            
            # Wait for login form
            await self.sms_page.wait_for_selector('input[name="username"]', timeout=15000)
            
            # Fill credentials
            await self.sms_page.fill('input[name="username"]', self.username)
            await self.sms_page.fill('input[name="password"]', self.password)
            
            # Handle captcha if present
            await self._handle_enhanced_captcha()
            
            # Submit form
            await self._submit_login_form()
            
            # Verify login success
            if await self._verify_login_success():
                # Navigate to SMS data page
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
        """Enhanced captcha handling with multiple strategies"""
        captcha_selectors = [
            'div.wrap-input100',
            '.captcha-container', 
            '#captcha-text',
            'div:has-text("What is")',
            'label:has-text("=")'
        ]
        
        captcha_text = None
        
        # Try to find captcha element
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
        
        # Fallback to page content search
        if not captcha_text:
            page_content = await self.sms_page.content()
            if 'what is' in page_content.lower() or 'captcha' in page_content.lower():
                # Extract relevant text
                soup_match = re.search(r'what is \d+ \+ \d+', page_content, re.IGNORECASE)
                if soup_match:
                    captcha_text = soup_match.group()
        
        # Solve and fill captcha
        if captcha_text:
            answer = self._extract_captcha_numbers(captcha_text)
            if answer:
                captcha_input = await self.sms_page.query_selector('input[name="capt"]')
                if captcha_input:
                    await captcha_input.fill(answer)
                    logger.info("Captcha solved and filled", answer=answer)
    
    async def _submit_login_form(self):
        """Submit login form with multiple button strategies"""
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
        """Verify login was successful"""
        current_url = self.sms_page.url
        
        # Check URL for success indicators
        if 'login' not in current_url.lower() or 'dashboard' in current_url.lower():
            return True
        
        # Check for error messages
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
        
        # Check for successful login indicators
        success_indicators = ['dashboard', 'welcome', 'sms', 'messages']
        page_content = await self.sms_page.content()
        
        return any(indicator in page_content.lower() for indicator in success_indicators)
    
    async def extract_sms_data_with_caching(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Extract SMS data with intelligent caching"""
        now = datetime.utcnow()
        cache_key = f"{self.session_id}_sms_data"
        
        # Check cache validity
        if not force_refresh and (now - self.last_refresh).seconds < 30:
            if cache_key in self.message_cache:
                logger.debug("Using cached SMS data")
                return self.message_cache[cache_key]
        
        try:
            if not self.sms_page:
                raise SMSServiceError("SMS page not initialized")
            
            # Wait for table with timeout
            await self.sms_page.wait_for_selector('#dt tbody', timeout=15000)
            
            # Extract table rows
            rows = await self.sms_page.query_selector_all('#dt tbody tr')
            sms_data = []
            
            for row in rows:
                try:
                    # Skip hidden rows
                    style = await row.get_attribute('style')
                    if style and 'display: none' in style:
                        continue
                    
                    # Extract cell data
                    cells = await row.query_selector_all('td')
                    if len(cells) < 6:
                        continue
                    
                    # Extract cell contents with validation
                    cell_contents = []
                    for cell in cells:
                        content = await cell.text_content()
                        cell_contents.append(content.strip() if content else '')
                    
                    # Skip invalid rows
                    if not cell_contents[0] or not cell_contents[2]:  # date and number required
                        continue
                    
                    # Skip total rows
                    if "0,0,0," in cell_contents[0]:
                        continue
                    
                    # Structure the data
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
            
            # Update cache
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
        service_type: ServiceType,
        timeout: int = None
    ) -> Optional[VerificationResult]:
        """Intelligently get verification code with enhanced pattern matching"""
        
        if timeout is None:
            timeout = config.services.verification_timeout
        
        clean_phone = self._normalize_phone_number(phone_number)
        start_time = time.time()
        
        logger.info(f"Looking for {service_type.value} verification code", 
                   phone_number=phone_number, timeout=timeout)
        
        # Check existing messages first
        existing_messages = await self.extract_sms_data_with_caching()
        
        # Look for existing verification code
        for message in existing_messages:
            if self._phone_numbers_match(clean_phone, message['number']):
                if self._is_service_message(message['cli'], service_type):
                    code = self._extract_verification_code_enhanced(message['sms'], service_type)
                    if code:
                        logger.info(f"Found existing {service_type.value} code", 
                                  phone_number=phone_number, code=code)
                        return VerificationResult(
                            phone_number=phone_number,
                            service=service_type,
                            success=True,
                            code=code
                        )
        
        # Monitor for new messages
        check_interval = 10
        max_iterations = timeout // check_interval
        
        for iteration in range(max_iterations):
            await asyncio.sleep(check_interval)
            
            # Refresh and check for new messages
            current_messages = await self.extract_sms_data_with_caching(force_refresh=True)
            
            # Check new messages
            for message in current_messages[len(existing_messages):]:
                if self._phone_numbers_match(clean_phone, message['number']):
                    if self._is_service_message(message['cli'], service_type):
                        code = self._extract_verification_code_enhanced(message['sms'], service_type)
                        if code:
                            # Store in database
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
            
            # Update existing messages for next iteration
            existing_messages = current_messages
            
            # Log progress
            remaining_time = timeout - (time.time() - start_time)
            logger.debug(f"Still waiting for code, {remaining_time:.0f}s remaining")
        
        # Timeout reached
        logger.warning(f"No {service_type.value} verification code found", 
                      phone_number=phone_number, timeout=timeout)
        
        return VerificationResult(
            phone_number=phone_number,
            service=service_type,
            success=False,
            error="Verification timeout"
        )
    
    def _normalize_phone_number(self, phone_number: str) -> str:
        """Normalize phone number for comparison"""
        return re.sub(r'[^\d]', '', phone_number)
    
    def _phone_numbers_match(self, phone1: str, phone2: str) -> bool:
        """Enhanced phone number matching"""
        clean1 = self._normalize_phone_number(phone1)
        clean2 = self._normalize_phone_number(phone2)
        
        # Exact match
        if clean1 == clean2:
            return True
        
        # One contains the other
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        # Last 8 digits match (common for international numbers)
        if len(clean1) >= 8 and len(clean2) >= 8:
            if clean1[-8:] == clean2[-8:]:
                return True
        
        # Last 10 digits match
        if len(clean1) >= 10 and len(clean2) >= 10:
            if clean1[-10:] == clean2[-10:]:
                return True
        
        return False
    
    def _is_service_message(self, cli: str, service_type: ServiceType) -> bool:
        """Check if message is from the expected service"""
        if not cli:
            return False
        
        cli_lower = cli.lower()
        service_identifiers = {
            ServiceType.UBER: ['uber'],
            ServiceType.AIRBNB: ['airbnb'],
            ServiceType.LYFT: ['lyft'],
            ServiceType.DOORDASH: ['doordash', 'door dash']
        }
        
        identifiers = service_identifiers.get(service_type, [])
        return any(identifier in cli_lower for identifier in identifiers)
    
    def _extract_verification_code_enhanced(self, sms_text: str, service_type: ServiceType) -> Optional[str]:
        """Enhanced verification code extraction with service-specific patterns"""
        if not sms_text:
            return None
        
        # Service-specific patterns
        service_patterns = {
            ServiceType.UBER: [
                r'Your Uber code is (\d{4,6})',
                r'Your verification code is (\d{4,6})',
                r'Uber.*?(\d{4,6})',
            ],
            ServiceType.AIRBNB: [
                r'Your Airbnb verification code is (\d{4,6})',
                r'verification code is (\d{4,6})',
                r'Airbnb.*?(\d{4,6})',
            ],
            ServiceType.LYFT: [
                r'Your Lyft code is (\d{4,6})',
                r'Lyft.*?(\d{4,6})',
            ],
            ServiceType.DOORDASH: [
                r'Your DoorDash verification code is (\d{4,6})',
                r'DoorDash.*?(\d{4,6})',
            ]
        }
        
        # Try service-specific patterns first
        patterns = service_patterns.get(service_type, [])
        
        # Add generic patterns
        patterns.extend([
            r'verification code is (\d{4,6})',
            r'code is (\d{4,6})',
            r'code: (\d{4,6})',
            r'your code is (\d{4,6})',
            r'(\d{4,6}) is your',
            r'(\d{6})',  # Fallback 6-digit pattern
            r'(\d{4})'   # Fallback 4-digit pattern
        ])
        
        for pattern in patterns:
            match = re.search(pattern, sms_text, re.IGNORECASE)
            if match:
                code = match.group(1)
                # Validate code length
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
        """Store verification result in database"""
        try:
            with SessionLocal() as db:
                # Parse timestamp
                try:
                    received_at = datetime.strptime(message['date'], '%Y-%m-%d %H:%M:%S')
                except:
                    received_at = datetime.utcnow()
                
                # Create SMS message record
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
                
                # Update phone number record
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
        """Enhanced cleanup with proper resource management"""
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


# Enhanced service automation classes
class ServiceAutomator:
    """Base class for service automation"""
    
    def __init__(self, service_type: ServiceType):
        self.service_type = service_type
        self.user_agent_manager = UserAgentManager()
        self.success_count = 0
        self.failure_count = 0
    
    async def request_verification_code(
        self, 
        phone_number: str, 
        country: str,
        playwright_instance
    ) -> VerificationResult:
        """Request verification code - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _generate_fake_profile(self) -> Dict[str, str]:
        """Generate fake user profile"""
        first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Mary', 'Patricia', 'Jennifer', 'Linda']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        email_providers = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
        email = f"{first_name.lower()}.{last_name.lower()}{random.randint(100, 999)}@{random.choice(email_providers)}"
        
        return {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'birth_year': random.randint(1980, 2000),
            'birth_month': random.randint(1, 12),
            'birth_day': random.randint(1, 28)
        }


class UberAutomator(ServiceAutomator):
    """Enhanced Uber automation"""
    
    def __init__(self):
        super().__init__(ServiceType.UBER)
    
    async def request_verification_code(
        self, 
        phone_number: str, 
        country: str,
        playwright_instance
    ) -> VerificationResult:
        """Request Uber verification code"""
        try:
            # Format phone number
            formatted_number = self._format_phone_number(phone_number, country)
            
            # Set up browser context
            user_agent = self.user_agent_manager.get_random_agent()
            headers = self.user_agent_manager.get_headers_for_agent(user_agent)
            
            browser = await playwright_instance.chromium.launch(
                headless=config.browser.headless,
                slow_mo=config.browser.slow_mo,
                args=["--disable-blink-features=AutomationControlled"]
            )
            
            context = await browser.new_context(
                user_agent=user_agent,
                extra_http_headers=headers,
                viewport={'width': config.browser.viewport_width, 'height': config.browser.viewport_height}
            )
            
            page = await context.new_page()
            
            try:
                # Navigate to Uber auth page
                await page.goto("https://auth.uber.com/v2/?breeze_local_zone=dca1", timeout=30000)
                await page.wait_for_load_state('networkidle')
                
                # Handle phone number input
                phone_selectors = [
                    '#PHONE_NUMBER',
                    'input[type="tel"]',
                    'input[name="phoneNumber"]',
                    'input[data-testid="phone-number-input"]'
                ]
                
                phone_input = None
                for selector in phone_selectors:
                    try:
                        phone_input = await page.wait_for_selector(selector, timeout=5000)
                        if phone_input:
                            break
                    except:
                        continue
                
                if not phone_input:
                    raise Exception("Phone input field not found")
                
                await phone_input.fill(formatted_number)
                logger.info(f"Filled phone number: {formatted_number}")
                
                # Click continue button
                continue_selectors = [
                    'button[type="submit"]',
                    'button:has-text("Continue")',
                    'button:has-text("Next")',
                    'button[data-testid="forward-button"]'
                ]
                
                for selector in continue_selectors:
                    try:
                        button = await page.wait_for_selector(selector, timeout=3000)
                        if button:
                            await button.click()
                            break
                    except:
                        continue
                
                # Wait for verification code input or error
                await asyncio.sleep(3)
                
                # Check for errors
                error_selectors = [
                    '[data-testid="error-message"]',
                    '.error-message',
                    'div:has-text("error")',
                    'div:has-text("invalid")'
                ]
                
                for selector in error_selectors:
                    try:
                        error_element = await page.query_selector(selector)
                        if error_element:
                            error_text = await error_element.text_content()
                            if error_text and len(error_text.strip()) > 0:
                                self.failure_count += 1
                                return VerificationResult(
                                    phone_number=phone_number,
                                    service=ServiceType.UBER,
                                    success=False,
                                    error=error_text.strip()
                                )
                    except:
                        continue
                
                # Success if no errors found
                self.success_count += 1
                logger.info(f"Successfully requested Uber verification for {phone_number}")
                
                return VerificationResult(
                    phone_number=phone_number,
                    service=ServiceType.UBER,
                    success=True
                )
                
            finally:
                await page.close()
                await context.close()
                await browser.close()
                
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Uber verification request failed for {phone_number}: {e}")
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.UBER,
                success=False,
                error=str(e)
            )
    
    def _format_phone_number(self, phone_number: str, country: str) -> str:
        """Format phone number for Uber"""
        # Country code mapping
        country_codes = {
            'United States': '+1',
            'United Kingdom': '+44',
            'Canada': '+1',
            'Australia': '+61',
            'Germany': '+49',
            'France': '+33',
            'Italy': '+39',
            'Spain': '+34',
            'Brazil': '+55',
            'India': '+91',
            'China': '+86',
            'Japan': '+81',
            'South Korea': '+82',
            'Mexico': '+52',
            'Egypt': '+20',
            'Nigeria': '+234',
            'Ghana': '+233'
        }
        
        # Clean phone number
        clean_number = re.sub(r'[^\d]', '', phone_number)
        
        # Get country code
        country_code = country_codes.get(country, '+1')
        
        # Remove country code if already present
        country_digits = country_code[1:]
        if clean_number.startswith(country_digits):
            clean_number = clean_number[len(country_digits):]
        
        return f"{country_code}{clean_number}"


class AirbnbAutomator(ServiceAutomator):
    """Enhanced Airbnb automation"""
    
    def __init__(self):
        super().__init__(ServiceType.AIRBNB)
    
    async def request_verification_code(
        self, 
        phone_number: str, 
        country: str,
        playwright_instance
    ) -> VerificationResult:
        """Request Airbnb verification code"""
        try:
            formatted_number = self._format_phone_number(phone_number, country)
            profile = self._generate_fake_profile()
            
            user_agent = self.user_agent_manager.get_random_agent()
            headers = self.user_agent_manager.get_headers_for_agent(user_agent)
            
            browser = await playwright_instance.chromium.launch(
                headless=config.browser.headless,
                slow_mo=config.browser.slow_mo
            )
            
            context = await browser.new_context(
                user_agent=user_agent,
                extra_http_headers=headers,
                viewport={'width': config.browser.viewport_width, 'height': config.browser.viewport_height}
            )
            
            page = await context.new_page()
            
            try:
                await page.goto("https://www.airbnb.com/signup_login", timeout=30000)
                await page.wait_for_load_state('networkidle')
                
                # Click phone tab if needed
                try:
                    phone_tab = await page.wait_for_selector('button[data-testid="signup-login-phone-tab"]', timeout=5000)
                    if phone_tab:
                        await phone_tab.click()
                        await asyncio.sleep(1)
                except:
                    pass
                
                # Select country code
                try:
                    country_select = await page.query_selector('select[data-testid="login-signup-countrycode"]')
                    if country_select:
                        country_code_value = self._get_airbnb_country_code(country)
                        await country_select.select_option(value=country_code_value)
                except Exception as e:
                    logger.debug(f"Could not select country code: {e}")
                
                # Fill phone number
                phone_input_selectors = [
                    'input[name="phoneInputphone-login"]',
                    'input[data-testid="phone-input"]',
                    'input[type="tel"]'
                ]
                
                phone_input = None
                for selector in phone_input_selectors:
                    try:
                        phone_input = await page.wait_for_selector(selector, timeout=3000)
                        if phone_input:
                            break
                    except:
                        continue
                
                if not phone_input:
                    raise Exception("Phone input not found")
                
                # Get local number part
                local_number = self._get_local_number(formatted_number, country)
                await phone_input.fill(local_number)
                
                # Submit form
                submit_button = await page.wait_for_selector('button[type="submit"]', timeout=10000)
                await submit_button.click()
                
                # Wait and check for success/error
                await asyncio.sleep(3)
                
                # Check for errors
                page_content = await page.content()
                if any(error in page_content.lower() for error in ['error', 'invalid', 'blocked']):
                    self.failure_count += 1
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.AIRBNB,
                        success=False,
                        error="Form submission error detected"
                    )
                
                self.success_count += 1
                logger.info(f"Successfully requested Airbnb verification for {phone_number}")
                
                return VerificationResult(
                    phone_number=phone_number,
                    service=ServiceType.AIRBNB,
                    success=True
                )
                
            finally:
                await page.close()
                await context.close()
                await browser.close()
                
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Airbnb verification request failed for {phone_number}: {e}")
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.AIRBNB,
                success=False,
                error=str(e)
            )
    
    def _format_phone_number(self, phone_number: str, country: str) -> str:
        """Format phone number for Airbnb"""
        # Similar to Uber formatting
        country_codes = {
            'United States': '1',
            'Egypt': '20',
            'Nigeria': '234',
            'Ghana': '233',
            'United Kingdom': '44',
            'India': '91'
        }
        
        clean_number = re.sub(r'[^\d]', '', phone_number)
        country_code = country_codes.get(country, '1')
        
        if clean_number.startswith(country_code):
            clean_number = clean_number[len(country_code):]
        
        return f"+{country_code}{clean_number}"
    
    def _get_airbnb_country_code(self, country: str) -> str:
        """Get Airbnb-specific country code value"""
        country_mapping = {
            'United States': '1US',
            'Egypt': '20EG',
            'Nigeria': '234NG',
            'Ghana': '233GH',
            'United Kingdom': '44GB',
            'India': '91IN'
        }
        return country_mapping.get(country, '1US')
    
    def _get_local_number(self, formatted_number: str, country: str) -> str:
        """Extract local number part"""
        country_codes = {
            'United States': '+1',
            'Egypt': '+20',
            'Nigeria': '+234',
            'Ghana': '+233',
            'United Kingdom': '+44',
            'India': '+91'
        }
        
        country_code = country_codes.get(country, '+1')
        if formatted_number.startswith(country_code):
            return formatted_number[len(country_code):].lstrip('0')
        
        return formatted_number


# Enhanced orchestration class
class EnhancedVerificationOrchestrator:
    """Orchestrates the entire verification process"""
    
    def __init__(self):
        self.sms_handler = None
        self.automators = {
            ServiceType.UBER: UberAutomator(),
            ServiceType.AIRBNB: AirbnbAutomator()
        }
        self.results: List[VerificationResult] = []
        self.session_id = str(uuid.uuid4())
        
    async def initialize(self, username: str, password: str):
        """Initialize the orchestrator"""
        self.sms_handler = EnhancedSMSServiceHandler(username, password)
        await self.sms_handler.__aenter__()
        
        # Login to SMS service
        if not await self.sms_handler.login_to_sms_service():
            raise LoginError("Failed to login to SMS service")
        
        logger.info("Verification orchestrator initialized successfully")
    
    async def process_phone_numbers(
        self, 
        phone_data: List[Dict[str, str]], 
        service_type: ServiceType
    ) -> List[VerificationResult]:
        """Process multiple phone numbers"""
        
        if service_type not in self.automators:
            raise ValueError(f"Unsupported service type: {service_type}")
        
        automator = self.automators[service_type]
        results = []
        
        async with async_playwright() as playwright:
            # Process in batches to manage resources
            batch_size = config.concurrency.batch_size
            
            for i in range(0, len(phone_data), batch_size):
                batch = phone_data[i:i + batch_size]
                
                # Create semaphore for concurrency control
                semaphore = asyncio.Semaphore(config.concurrency.max_workers)
                
                # Process batch
                tasks = [
                    self._process_single_number(
                        phone_info, service_type, automator, playwright, semaphore
                    )
                    for phone_info in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                    else:
                        results.append(result)
                        self.results.append(result)
                
                # Progress update
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(phone_data) + batch_size - 1)//batch_size}")
                
                # Rate limiting between batches
                if i + batch_size < len(phone_data):
                    delay = random.uniform(*config.services.rate_limit_delay)
                    await asyncio.sleep(delay)
        
        return results
    
    async def _process_single_number(
        self,
        phone_info: Dict[str, str],
        service_type: ServiceType,
        automator: ServiceAutomator,
        playwright_instance,
        semaphore: asyncio.Semaphore
    ) -> VerificationResult:
        """Process a single phone number"""
        
        async with semaphore:
            phone_number = phone_info['Number']
            country = phone_info.get('Range', phone_info.get('Country', 'United States'))
            
            try:
                logger.info(f"Processing {phone_number} for {service_type.value}")
                
                # Request verification code
                request_result = await automator.request_verification_code(
                    phone_number, country, playwright_instance
                )
                
                if not request_result.success:
                    return request_result
                
                # Wait for SMS verification code
                verification_result = await self.sms_handler.get_verification_code_intelligent(
                    phone_number, service_type
                )
                
                # Merge results
                if verification_result and verification_result.success:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=service_type,
                        success=True,
                        code=verification_result.code
                    )
                else:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=service_type,
                        success=False,
                        error="SMS verification code not received"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {phone_number}: {e}")
                return VerificationResult(
                    phone_number=phone_number,
                    service=service_type,
                    success=False,
                    error=str(e)
                )
            finally:
                # Rate limiting
                delay = random.uniform(*config.services.rate_limit_delay)
                await asyncio.sleep(delay)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        if not self.results:
            return {'status': 'No results to report'}
        
        # Calculate statistics
        total_attempts = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_attempts - successful
        success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0
        
        # Group by service
        service_stats = {}
        for service_type in ServiceType:
            service_results = [r for r in self.results if r.service == service_type]
            if service_results:
                service_successful = sum(1 for r in service_results if r.success)
                service_stats[service_type.value] = {
                    'total': len(service_results),
                    'successful': service_successful,
                    'failed': len(service_results) - service_successful,
                    'success_rate': (service_successful / len(service_results)) * 100
                }
        
        # Error analysis
        error_types = {}
        for result in self.results:
            if not result.success and result.error:
                error_type = result.error[:50]  # First 50 chars
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Successful codes
        successful_codes = [
            {
                'phone_number': r.phone_number,
                'service': r.service.value,
                'code': r.code,
                'timestamp': r.timestamp.isoformat()
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
            'service_breakdown': service_stats,
            'error_analysis': error_types,
            'successful_verifications': successful_codes,
            'processing_time': (datetime.utcnow() - datetime.fromisoformat(self.session_id.split('-')[0])).total_seconds() if '-' in self.session_id else 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.sms_handler:
            await self.sms_handler.__aexit__(None, None, None)


# Enhanced CSV processing
class CSVProcessor:
    """Enhanced CSV processing with validation and normalization"""
    
    @staticmethod
    def load_and_validate_csv(file_path: str) -> pd.DataFrame:
        """Load and validate CSV file"""
        try:
            # Try different encodings
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
            
            # Validate required columns
            required_columns = ['Number']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try to find similar column names
                available_cols = df.columns.tolist()
                phone_candidates = [col for col in available_cols if 
                                 any(keyword in col.lower() for keyword in ['phone', 'number', 'mobile', 'cell'])]
                
                if phone_candidates:
                    df = df.rename(columns={phone_candidates[0]: 'Number'})
                    logger.info(f"Mapped column '{phone_candidates[0]}' to 'Number'")
                else:
                    raise ValueError(f"Required columns missing: {missing_columns}")
            
            # Add default country column if missing
            if 'Range' not in df.columns and 'Country' not in df.columns:
                df['Range'] = 'United States'  # Default country
                logger.info("Added default country column")
            
            # Clean and validate phone numbers
            df['Number'] = df['Number'].astype(str).str.strip()
            df = df[df['Number'].str.len() > 5]  # Remove obviously invalid numbers
            df = df.dropna(subset=['Number'])
            
            # Remove duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=['Number'])
            removed_duplicates = initial_count - len(df)
            
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate phone numbers")
            
            logger.info(f"Loaded {len(df)} valid phone numbers from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    @staticmethod
    def find_csv_files(directory: str = ".") -> List[str]:
        """Find CSV files in directory"""
        patterns = ["*SMS*.csv", "*Number*.csv", "*Phone*.csv", "*.csv"]
        found_files = []
        
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            found_files.extend(files)
        
        # Remove duplicates and sort by modification time
        unique_files = list(set(found_files))
        unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return unique_files


# Enhanced logging setup
def setup_enhanced_logging():
    """Setup enhanced logging with structured output"""
    
    # Configure structlog
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
    
    # Setup rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )

# Initialize logging
setup_enhanced_logging()
logger = structlog.get_logger()


# Enhanced CLI interface
def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser"""
    parser = argparse.ArgumentParser(
        description="Enhanced SMS Verification Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --service uber --csv phones.csv
  %(prog)s --service airbnb --csv phones.csv --headless
  %(prog)s --display-messages
  %(prog)s --test-number +1234567890 --service uber
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
    
    # Service selection
    service_group = parser.add_argument_group('Service Options')
    service_group.add_argument(
        '--service', 
        type=str, 
        choices=[s.value for s in ServiceType], 
        default=ServiceType.UBER.value,
        help='Service to request verification codes from'
    )
    
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
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--max-workers', type=int, default=3, help='Maximum concurrent workers')
    perf_group.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    perf_group.add_argument('--timeout', type=int, default=180, help='Verification timeout in seconds')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-format', choices=['json', 'csv', 'table'], default='table', help='Output format')
    output_group.add_argument('--output-file', type=str, help='Output file path')
    output_group.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser


def display_results_table(results: List[VerificationResult]):
    """Display results in a formatted table"""
    table = Table(title="Verification Results")
    
    table.add_column("Phone Number", style="cyan")
    table.add_column("Service", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Code", style="yellow")
    table.add_column("Error", style="red")
    table.add_column("Timestamp", style="blue")
    
    for result in results:
        status = "✅ Success" if result.success else "❌ Failed"
        code = result.code or "-"
        error = result.error[:30] + "..." if result.error and len(result.error) > 30 else (result.error or "-")
        timestamp = result.timestamp.strftime("%H:%M:%S") if result.timestamp else "-"
        
        table.add_row(
            result.phone_number,
            result.service.value.upper(),
            status,
            code,
            error,
            timestamp
        )
    
    console.print(table)


def save_results(results: List[VerificationResult], format: str, file_path: str = None):
    """Save results in specified format"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'json':
        file_path = file_path or f'verification_results_{timestamp}.json'
        data = [
            {
                'phone_number': r.phone_number,
                'service': r.service.value,
                'success': r.success,
                'code': r.code,
                'error': r.error,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None
            }
            for r in results
        ]
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    elif format == 'csv':
        file_path = file_path or f'verification_results_{timestamp}.csv'
        data = []
        for r in results:
            data.append({
                'phone_number': r.phone_number,
                'service': r.service.value,
                'success': r.success,
                'code': r.code,
                'error': r.error,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    console.print(f"[green]Results saved to {file_path}[/]")


# Main execution function
async def main():
    """Enhanced main function with comprehensive error handling"""
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Update config with CLI arguments
        if args.headless:
            config.browser.headless = True
        if args.slow_mo:
            config.browser.slow_mo = args.slow_mo
        if args.max_workers:
            config.concurrency.max_workers = args.max_workers
        if args.batch_size:
            config.concurrency.batch_size = args.batch_size
        if args.timeout:
            config.services.verification_timeout = args.timeout
        
        # Set verbose logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Get credentials
        username = args.username or config.sms_service.username or os.getenv('SMS_USERNAME')
        password = args.password or config.sms_service.password or os.getenv('SMS_PASSWORD')
        
        if not username or not password:
            console.print("[red]Error: SMS service credentials not provided[/]")
            console.print("Use --username and --password, or set SMS_USERNAME and SMS_PASSWORD environment variables")
            return 1
        
        # Display header
        console.rule("[bold blue]Enhanced SMS Verification Automation[/]")
        console.print(f"[bold]Session ID:[/] {uuid.uuid4()}")
        console.print(f"[bold]Service:[/] {args.service.upper()}")
        console.print(f"[bold]Action:[/] {args.action}")
        
        # Initialize orchestrator
        orchestrator = EnhancedVerificationOrchestrator()
        
        try:
            await orchestrator.initialize(username, password)
            
            if args.action == 'display-messages':
                # Display existing messages
                console.print("\n[bold yellow]Displaying existing SMS messages...[/]")
                await orchestrator.sms_handler.display_existing_messages()
                
            elif args.action == 'test-single':
                if not args.test_number:
                    console.print("[red]Error: --test-number required for test-single action[/]")
                    return 1
                
                console.print(f"\n[bold yellow]Testing single number: {args.test_number}[/]")
                
                phone_data = [{'Number': args.test_number, 'Range': 'United States'}]
                service_type = ServiceType(args.service)
                
                results = await orchestrator.process_phone_numbers(phone_data, service_type)
                
                # Display results
                display_results_table(results)
                
            elif args.action == 'verify':
                # Load phone numbers
                if args.csv:
                    csv_file = args.csv
                elif args.test_number:
                    # Create temporary data for single number
                    phone_data = [{'Number': args.test_number, 'Range': 'United States'}]
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
                
                console.print(f"\n[bold green]Processing {len(phone_data)} phone numbers...[/]")
                
                # Process phone numbers
                service_type = ServiceType(args.service)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing phone numbers...", total=len(phone_data))
                    
                    results = await orchestrator.process_phone_numbers(phone_data, service_type)
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
                
                # Save results
                if args.output_file or args.output_format != 'table':
                    save_results(results, args.output_format, args.output_file)
                
                # Save report
                report_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    try:
        # Set event loop policy for Windows
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except Exception as e:
        console.print(f"[red]Critical error: {e}[/]")
        sys.exit(1)
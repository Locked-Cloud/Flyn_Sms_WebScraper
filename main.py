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
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
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

# Setup logging
def setup_logging():
    """Setup structured logging with rich console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Global config variable
config = None

# Enhanced logging setup
def setup_enhanced_logging():
    """Setup enhanced logging with better formatting"""
    
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
        try:
            parts = proxy_string.strip().split('|')
            proxy_part = parts[0]
            
            # Parse proxy credentials
            proxy_components = proxy_part.split(':')
            if len(proxy_components) < 4:
                raise ValueError(f"Invalid proxy format. Expected host:port:username:password, got: {proxy_string}")
            
            host = proxy_components[0].strip()
            port = int(proxy_components[1].strip())
            username = proxy_components[2].strip()
            password = proxy_components[3].strip()
            
            # Validate components
            if not host or not username or not password:
                raise ValueError(f"Invalid proxy credentials in: {proxy_string}")
            
            # Extract location info if available
            country = parts[1].strip() if len(parts) > 1 else ""
            region = parts[2].strip() if len(parts) > 2 else ""
            
            return cls(
                host=host,
                port=port,
                username=username,
                password=password,
                country=country,
                region=region
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse proxy string '{proxy_string}': {str(e)}")

@dataclass
class AccountCreationResult:
    """Result of account creation attempt"""
    success: bool
    email: str
    password: str
    error: Optional[str] = None
    error_type: Optional[str] = None
    proxy_used: Optional[str] = None

@dataclass
class VerificationResult:
    phone_number: str
    service: ServiceType
    success: bool
    code: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    proxy_used: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    account_email: Optional[str] = None
    
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
    valr_signup_url: str = Field("https://www.valr.com/en/signup", description="VALR signup URL")
    valr_verify_url: str = Field("https://www.valr.com/en/verify/phone", description="VALR verification URL")
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
try:
    config = Config.load_from_file()
except Exception as e:
    console.print(f"[red]Error loading config: {e}[/]")
    config = Config.create_default()

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

class VALRAccount(Base):
    __tablename__ = "valr_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    phone_number = Column(String, nullable=True)
    verification_status = Column(String, default="pending")
    proxy_used = Column(String, nullable=True)
    session_id = Column(String, index=True)

# Database setup with enhanced error handling
try:
    engine = create_engine(
        "sqlite:///valr_sms.db",
        echo=False,
        pool_size=20,
        pool_timeout=30,
        pool_pre_ping=True
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
except Exception as e:
    console.print(f"[red]Database initialization error: {e}[/]")
    raise

# Enhanced exception handling
class SMSServiceError(Exception):
    """Base SMS service error"""
    def __init__(self, message: str, error_type: str = "SMS_SERVICE_ERROR", original_error: Exception = None):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

class AccountCreationError(SMSServiceError):
    def __init__(self, message: str = "Account creation failed", original_error: Exception = None):
        super().__init__(message, "ACCOUNT_CREATION_ERROR", original_error)

class CaptchaError(SMSServiceError):
    def __init__(self, message: str = "Captcha solving failed", original_error: Exception = None):
        super().__init__(message, "CAPTCHA_ERROR", original_error)

class LoginError(SMSServiceError):
    def __init__(self, message: str = "Login failed", original_error: Exception = None):
        super().__init__(message, "LOGIN_ERROR", original_error)

class RateLimitError(SMSServiceError):
    def __init__(self, message: str = "Rate limit exceeded", original_error: Exception = None):
        super().__init__(message, "RATE_LIMIT_ERROR", original_error)

class VerificationTimeoutError(SMSServiceError):
    def __init__(self, message: str = "Verification timeout", original_error: Exception = None):
        super().__init__(message, "VERIFICATION_TIMEOUT_ERROR", original_error)

class ProxyError(SMSServiceError):
    def __init__(self, message: str = "Proxy error", original_error: Exception = None):
        super().__init__(message, "PROXY_ERROR", original_error)

class BrowserError(SMSServiceError):
    def __init__(self, message: str = "Browser error", original_error: Exception = None):
        super().__init__(message, "BROWSER_ERROR", original_error)

class NetworkError(SMSServiceError):
    def __init__(self, message: str = "Network error", original_error: Exception = None):
        super().__init__(message, "NETWORK_ERROR", original_error)

# Error handler decorator
def handle_errors(error_types: Dict[type, str] = None, max_retries: int = 3):
    """Enhanced error handling decorator"""
    if error_types is None:
        error_types = {
            PlaywrightTimeout: "PLAYWRIGHT_TIMEOUT",
            ConnectionError: "CONNECTION_ERROR",
            TimeoutError: "TIMEOUT_ERROR",
            Exception: "GENERAL_ERROR"
        }
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except tuple(error_types.keys()) as e:
                    error_type = error_types.get(type(e), "UNKNOWN_ERROR")
                    last_error = e
                    
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...", 
                                     error_type=error_type)
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)} (Type: {error_type})")
                        
                        # Convert to our custom error types
                        if isinstance(e, PlaywrightTimeout):
                            raise VerificationTimeoutError(f"Operation timed out after {max_retries} attempts", e)
                        elif isinstance(e, (ConnectionError, TimeoutError)):
                            raise NetworkError(f"Network error after {max_retries} attempts", e)
                        else:
                            raise SMSServiceError(f"Operation failed after {max_retries} attempts: {str(e)}", 
                                                "GENERAL_ERROR", e)
            
            raise last_error
        return wrapper
    return decorator


def display_welcome_banner():
    """Display enhanced welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🚀 Enhanced VALR SMS Verification with Account Creation    ║
║                           v3.0                               ║
║                                                              ║
║  Features:                                                   ║
║  • Automatic VALR account creation                           ║
║  • Smart proxy rotation and management                       ║
║  • Enhanced error handling and recovery                      ║
║  • Interactive configuration mode                            ║
║  • Comprehensive reporting and analytics                     ║
║  • Real-time progress tracking                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Enhanced VALR SMS Verification Automation Tool with Account Creation and Proxy Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv phones.csv
  %(prog)s --test-number +201234567890
  %(prog)s --display-messages
  %(prog)s --csv phones.csv --proxy-file proxies.txt
  %(prog)s --test-number +201234567890 --proxy "163.172.65.133:48103:6c65:6c65|United Kingdom"
  %(prog)s --interactive  # Interactive mode with proxy selection
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
    proxy_group.add_argument('--interactive', '-i', action='store_true', help='Interactive mode with proxy selection')
    
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


# Enhanced Verification Orchestrator with Account Creation
class EnhancedVerificationOrchestrator:
    """Orchestrates the entire VALR verification process with account creation and proxy support"""
    
    def __init__(self, proxy_strings: List[str] = None, use_proxies: bool = True):
        self.sms_handler = None
        self.proxy_pool_manager = None
        self.valr_automator = None
        self.results: List[VerificationResult] = []
        self.created_accounts: List[AccountCreationResult] = []
        self.session_id = str(uuid.uuid4())
        self.use_proxies = use_proxies
        self.error_counts = {
            'total_errors': 0,
            'proxy_errors': 0,
            'network_errors': 0,
            'timeout_errors': 0,
            'verification_errors': 0,
            'account_creation_errors': 0,
            'other_errors': 0
        }
        
        # Initialize proxy pool if proxy strings provided and proxies are enabled
        if proxy_strings and use_proxies:
            try:
                self.proxy_pool_manager = ProxyPoolManager(proxy_strings)
                logger.info(f"Initialized with {len(proxy_strings)} proxies")
            except Exception as e:
                logger.error(f"Failed to initialize proxy pool: {e}")
                self.proxy_pool_manager = None
        
        # Initialize VALR automator with proxy support
        self.valr_automator = VALRAutomator(self.proxy_pool_manager)
        
    async def initialize(self, username: str, password: str):
        """Initialize the orchestrator with enhanced error handling"""
        try:
            self.sms_handler = EnhancedSMSServiceHandler(username, password)
            await self.sms_handler.__aenter__()
            
            # Login to SMS service
            if not await self.sms_handler.login_to_sms_service():
                raise LoginError("Failed to login to SMS service")
            
            logger.info("Enhanced VALR verification orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.cleanup()
            raise
    
    @handle_errors({
        SMSServiceError: "SMS_SERVICE_ERROR",
        NetworkError: "NETWORK_ERROR",
        Exception: "PROCESSING_ERROR"
    }, max_retries=1)
    async def process_phone_numbers_with_account_creation(
        self, 
        phone_data: List[Dict[str, str]]
    ) -> List[VerificationResult]:
        """Process multiple phone numbers for VALR verification with account creation"""
        
        results = []
        
        try:
            async with async_playwright() as playwright:
                # Process numbers with controlled concurrency
                semaphore = asyncio.Semaphore(2)  # Limit concurrent requests
                
                tasks = [
                    self._process_single_number_with_account(
                        phone_info, playwright, semaphore
                    )
                    for phone_info in phone_data
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        self.error_counts['total_errors'] += 1
                        self.error_counts['other_errors'] += 1
                        
                        # Create a failed result for the exception
                        failed_result = VerificationResult(
                            phone_number="unknown",
                            service=ServiceType.VALR,
                            success=False,
                            error=str(result),
                            error_type="TASK_EXCEPTION"
                        )
                        results.append(failed_result)
                    else:
                        results.append(result)
                        self.results.append(result)
                        
                        # Track error statistics
                        if not result.success:
                            self.error_counts['total_errors'] += 1
                            if 'proxy' in result.error_type.lower():
                                self.error_counts['proxy_errors'] += 1
                            elif 'network' in result.error_type.lower():
                                self.error_counts['network_errors'] += 1
                            elif 'timeout' in result.error_type.lower():
                                self.error_counts['timeout_errors'] += 1
                            elif 'verification' in result.error_type.lower():
                                self.error_counts['verification_errors'] += 1
                            elif 'account' in result.error_type.lower():
                                self.error_counts['account_creation_errors'] += 1
                            else:
                                self.error_counts['other_errors'] += 1
        
        except Exception as e:
            logger.error(f"Failed to process phone numbers: {e}")
            raise
        
        return results
    
    async def _process_single_number_with_account(
        self,
        phone_info: Dict[str, str],
        playwright_instance,
        semaphore: asyncio.Semaphore
    ) -> VerificationResult:
        """Process a single phone number with account creation"""
        
        async with semaphore:
            try:
                phone_number = phone_info.get('phone_number', phone_info.get('number', ''))
                country = phone_info.get('country', 'EG')
                
                if not phone_number:
                    return VerificationResult(
                        phone_number="unknown",
                        service=ServiceType.VALR,
                        success=False,
                        error="No phone number provided",
                        error_type="INVALID_INPUT"
                    )
                
                logger.info(f"Processing phone number: {phone_number}")
                
                # Step 1: Create VALR account
                account_result = await self.valr_automator.create_valr_account(playwright_instance)
                
                if not account_result.success:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=False,
                        error=account_result.error,
                        error_type=account_result.error_type,
                        account_email=account_result.email
                    )
                
                # Store account creation result
                self.created_accounts.append(account_result)
                
                # Step 2: Request verification code using the created account
                verification_result = await self.valr_automator.request_verification_code_with_account(
                    phone_number=phone_number,
                    account_result=account_result,
                    country=country,
                    playwright_instance=playwright_instance
                )
                
                # Add account email to verification result
                verification_result.account_email = account_result.email
                
                return verification_result
                
            except Exception as e:
                logger.error(f"Error processing {phone_info.get('phone_number', 'unknown')}: {e}")
                return VerificationResult(
                    phone_number=phone_info.get('phone_number', 'unknown'),
                    service=ServiceType.VALR,
                    success=False,
                    error=str(e),
                    error_type="PROCESSING_ERROR"
                )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report with enhanced analytics"""
        total_processed = len(self.results)
        successful = len([r for r in self.results if r.success])
        failed = total_processed - successful
        
        # Calculate success rate
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
        
        # Account creation statistics
        total_accounts = len(self.created_accounts)
        successful_accounts = len([a for a in self.created_accounts if a.success])
        account_success_rate = (successful_accounts / total_accounts * 100) if total_accounts > 0 else 0
        
        # Proxy statistics
        proxy_stats = {}
        if self.proxy_pool_manager:
            proxy_stats = self.proxy_pool_manager.get_statistics()
        
        return {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'total_processed': total_processed,
            'successful_verifications': successful,
            'failed_verifications': failed,
            'success_rate': success_rate,
            'total_accounts_created': total_accounts,
            'successful_accounts': successful_accounts,
            'account_success_rate': account_success_rate,
            'error_breakdown': self.error_counts,
            'proxy_statistics': proxy_stats,
            'results': [
                {
                    'phone_number': r.phone_number,
                    'success': r.success,
                    'code': r.code,
                    'error': r.error,
                    'error_type': r.error_type,
                    'account_email': r.account_email,
                    'proxy_used': r.proxy_used,
                    'timestamp': r.timestamp.isoformat() if r.timestamp else None
                }
                for r in self.results
            ],
            'accounts': [
                {
                    'email': a.email,
                    'success': a.success,
                    'error': a.error,
                    'error_type': a.error_type,
                    'proxy_used': a.proxy_used
                }
                for a in self.created_accounts
            ]
        }
    
    async def cleanup(self):
        """Cleanup resources with enhanced error handling"""
        try:
            if self.sms_handler:
                await self.sms_handler.cleanup()
                await self.sms_handler.__aexit__(None, None, None)
            logger.info("Enhanced orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Main execution function with enhanced account creation support
async def main():
    """Enhanced main function with account creation and interactive proxy selection"""
    try:
        # Display welcome banner
        display_welcome_banner()
        
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
            console.print("[dim]Verbose logging enabled[/]")
        
        # Determine proxy configuration
        use_proxies = True
        proxy_strings = []
        
        if args.interactive:
            # Interactive mode
            use_proxies, proxy_strings = interactive_proxy_setup()
        elif args.no_proxy:
            # Explicitly disable proxies
            use_proxies = False
            console.print("[yellow]⚠️ Proxies disabled by user[/]")
        else:
            # Command line proxy configuration
            if args.proxy:
                try:
                    ProxyConfig.from_string(args.proxy)  # Validate
                    proxy_strings = [args.proxy]
                    console.print("[green]✅ Using single proxy from command line[/]")
                except Exception as e:
                    console.print(f"[red]❌ Invalid proxy format: {e}[/]")
                    return 1
            elif args.proxy_file:
                try:
                    proxy_strings = load_proxy_strings(args.proxy_file)
                    console.print(f"[green]✅ Loaded {len(proxy_strings)} proxies from file[/]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to load proxy file: {e}[/]")
                    return 1
            else:
                # Try to find default proxy file
                default_proxy_files = ['proxies.txt', 'proxy.txt', 'proxy_list.txt']
                for file_name in default_proxy_files:
                    if os.path.exists(file_name):
                        try:
                            proxy_strings = load_proxy_strings(file_name)
                            console.print(f"[green]✅ Auto-loaded {len(proxy_strings)} proxies from {file_name}[/]")
                            break
                        except:
                            continue
                
                # If no proxies found, ask user
                if not proxy_strings:
                    console.print("[yellow]⚠️ No proxy configuration found[/]")
                    if Confirm.ask("Would you like to run without proxies?", default=True):
                        use_proxies = False
                        console.print("[green]✅ Running without proxies[/]")
                    else:
                        console.print("[blue]Please provide proxy configuration and try again[/]")
                        return 1
        
        # Get credentials with enhanced prompts
        username = args.username or config.sms_service.username or os.getenv('SMS_USERNAME')
        password = args.password or config.sms_service.password or os.getenv('SMS_PASSWORD')
        
        if not username:
            username = Prompt.ask("\n[bold yellow]Enter SMS service username[/]")
        if not password:
            password = Prompt.ask("\n[bold yellow]Enter SMS service password[/]", password=True)
        
        if not username or not password:
            console.print("[red]❌ SMS service credentials are required[/]")
            console.print("Use --username and --password, or set SMS_USERNAME and SMS_PASSWORD environment variables")
            return 1
        
        # Display configuration summary
        console.rule("[bold blue]Configuration Summary[/]")
        config_table = Table.grid(padding=1)
        config_table.add_column(style="bold blue")
        config_table.add_column()
        
        config_table.add_row("Session ID:", str(uuid.uuid4())[:8])
        config_table.add_row("Action:", args.action)
        config_table.add_row("Account Creation:", "Enabled")
        config_table.add_row("Proxies:", f"{len(proxy_strings)} loaded" if proxy_strings else "Disabled")
        config_table.add_row("Browser Mode:", "Headless" if config.browser.headless else "GUI")
        config_table.add_row("Verification Timeout:", f"{config.services.verification_timeout}s")
        
        console.print(config_table)
        console.print()
        
        # Initialize orchestrator with enhanced configuration
        orchestrator = EnhancedVerificationOrchestrator(proxy_strings, use_proxies)
        
        try:
            # Initialize with progress indication
            with console.status("[bold green]Initializing automation system..."):
                await orchestrator.initialize(username, password)
                console.print("[green]✅ System initialized successfully[/]")
            
            if args.action == 'display-messages':
                # Display existing messages
                console.print("\n[bold yellow]📱 Displaying recent SMS messages...[/]")
                
                try:
                    messages = await orchestrator.sms_handler.extract_sms_data_with_caching()
                    
                    if messages:
                        table = Table(title="Recent SMS Messages", show_lines=True)
                        table.add_column("Date", style="cyan", no_wrap=True)
                        table.add_column("Number", style="green", no_wrap=True)
                        table.add_column("CLI", style="yellow", max_width=15)
                        table.add_column("SMS Content", style="white", max_width=50)
                        
                        for msg in messages[-20:]:  # Show last 20 messages
                            sms_content = msg.get('sms', '')
                            if len(sms_content) > 50:
                                sms_content = sms_content[:47] + "..."
                            
                            table.add_row(
                                msg.get('date', ''),
                                msg.get('number', ''),
                                msg.get('cli', ''),
                                sms_content
                            )
                        
                        console.print(table)
                        
                        # Show VALR messages specifically
                        valr_messages = [msg for msg in messages if 
                                       orchestrator.sms_handler._is_valr_message(msg.get('cli', ''), msg.get('sms', ''))]
                        
                        if valr_messages:
                            console.print(f"\n[bold green]Found {len(valr_messages)} VALR-related messages[/]")
                        
                    else:
                        console.print("[yellow]📭 No SMS messages found[/]")
                        
                except Exception as e:
                    console.print(f"[red]❌ Error retrieving messages: {e}[/]")
                    return 1
                
            elif args.action == 'test-single':
                if not args.test_number:
                    args.test_number = Prompt.ask("\n[bold yellow]Enter phone number to test[/]")
                
                console.print(f"\n[bold yellow]🧪 Testing single number with account creation: {args.test_number}[/]")
                
                phone_data = [{'Number': args.test_number, 'Country': 'EG'}]
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Creating account and processing number...", total=1)
                    results = await orchestrator.process_phone_numbers_with_account_creation(phone_data)
                    progress.update(task, completed=1)
                
                # Display results
                display_results_table(results)
                
                # Display account creation results
                if orchestrator.created_accounts:
                    console.print("\n[bold cyan]📧 Account Creation Results:[/]")
                    display_account_creation_table(orchestrator.created_accounts)
                
                if results and not results[0].success:
                    display_error_analysis(results)
                
            elif args.action == 'verify':
                # Load phone numbers
                phone_data = []
                
                if args.csv:
                    csv_file = args.csv
                elif args.test_number:
                    # Create temporary data for single number
                    phone_data = [{'Number': args.test_number, 'Country': 'EG'}]
                else:
                    # Find CSV file automatically
                    csv_files = CSVProcessor.find_csv_files()
                    if not csv_files:
                        console.print("[red]❌ No CSV file found. Use --csv to specify file path[/]")
                        return 1
                    csv_file = csv_files[0]
                    console.print(f"[yellow]📁 Using CSV file: {csv_file}[/]")
                
                if not phone_data:
                    # Load from CSV
                    try:
                        with console.status("[bold blue]Loading and validating CSV..."):
                            df = CSVProcessor.load_and_validate_csv(csv_file)
                            phone_data = df.to_dict('records')
                            console.print(f"[green]✅ Loaded {len(phone_data)} phone numbers[/]")
                    except Exception as e:
                        console.print(f"[red]❌ Error loading CSV: {e}[/]")
                        return 1
                
                console.print(f"\n[bold green]🚀 Processing {len(phone_data)} phone numbers with VALR account creation...[/]")
                
                # Confirm before processing if many numbers
                if len(phone_data) > 10:
                    if not Confirm.ask(f"Process {len(phone_data)} phone numbers? This will create {len(phone_data)} VALR accounts.", default=True):
                        console.print("[yellow]Operation cancelled by user[/]")
                        return 0
                
                # Process phone numbers with progress tracking
                results = []
                start_time = time.time()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Creating accounts and processing phone numbers...", total=len(phone_data))
                    
                    try:
                        results = await orchestrator.process_phone_numbers_with_account_creation(phone_data)
                        progress.update(task, completed=len(phone_data))
                    except Exception as e:
                        console.print(f"\n[red]❌ Processing failed: {e}[/]")
                        logger.error(f"Processing error: {e}", exc_info=True)
                        return 1
                
                processing_time = time.time() - start_time
                
                # Display results
                console.print("\n" + "="*60)
                console.print("[bold blue]📊 RESULTS SUMMARY[/]", justify="center")
                console.print("="*60)
                
                if args.output_format == 'table':
                    display_results_table(results)
                
                # Display account creation results
                if orchestrator.created_accounts:
                    console.print("\n[bold cyan]📧 Account Creation Results:[/]")
                    display_account_creation_table(orchestrator.created_accounts)
                
                # Display error analysis
                display_error_analysis(results)
                
                # Generate and display comprehensive report
                report = orchestrator.generate_report()
                
                # Performance metrics
                console.print(f"\n[bold magenta]⚡ Performance Metrics:[/]")
                perf_table = Table.grid(padding=1)
                perf_table.add_column(style="bold cyan")
                perf_table.add_column()
                
                perf_table.add_row("Processing Time:", f"{processing_time:.1f}s")
                perf_table.add_row("Success Rate:", f"{report['summary']['success_rate']:.1f}%")
                perf_table.add_row("Total Attempts:", str(report['summary']['total_attempts']))
                perf_table.add_row("Successful:", str(report['summary']['successful']))
                perf_table.add_row("Failed:", str(report['summary']['failed']))
                perf_table.add_row("Accounts Created:", str(report['account_creation']['total_accounts_created']))
                perf_table.add_row("Account Success Rate:", f"{report['account_creation']['account_creation_rate']:.1f}%")
                
                console.print(perf_table)
                
                # Display proxy statistics if available
                if report.get('proxy_statistics'):
                    console.print(f"\n[bold magenta]🌐 Proxy Performance:[/]")
                    proxy_table = Table()
                    proxy_table.add_column("Proxy", style="cyan", no_wrap=True)
                    proxy_table.add_column("Total", style="blue", justify="center")
                    proxy_table.add_column("Success", style="green", justify="center")
                    proxy_table.add_column("Success Rate", style="yellow", justify="center")
                    proxy_table.add_column("Top Error", style="red", max_width=20)
                    
                    for proxy, stats in report['proxy_statistics'].items():
                        top_error = "None"
                        if stats.get('errors'):
                            top_error = max(stats['errors'].items(), key=lambda x: x[1])[0]
                        
                        proxy_table.add_row(
                            proxy.split(':')[0] + ":***",  # Hide credentials
                            str(stats['total']),
                            str(stats['success']),
                            f"{stats['success_rate']:.1f}%",
                            top_error
                        )
                    
                    console.print(proxy_table)
                
                # Display successful codes
                successful_codes = report['successful_verifications']
                if successful_codes:
                    console.print(f"\n[bold green]✅ Successful Verification Codes ({len(successful_codes)}):[/]")
                    
                    codes_table = Table()
                    codes_table.add_column("Phone Number", style="cyan")
                    codes_table.add_column("Code", style="bold green", justify="center")
                    codes_table.add_column("Account Email", style="blue", max_width=25)
                    codes_table.add_column("Proxy", style="magenta", no_wrap=True)
                    codes_table.add_column("Time", style="blue", no_wrap=True)
                    
                    for verification in successful_codes:
                        proxy_display = "Direct"
                        if verification['proxy_used']:
                            proxy_display = verification['proxy_used'].split(':')[0] + ":***"
                        
                        account_email = verification['account_email'][:22] + "..." if verification['account_email'] and len(verification['account_email']) > 25 else (verification['account_email'] or "-")
                        timestamp = datetime.fromisoformat(verification['timestamp']).strftime("%H:%M:%S")
                        
                        codes_table.add_row(
                            verification['phone_number'],
                            verification['code'],
                            account_email,
                            proxy_display,
                            timestamp
                        )
                    
                    console.print(codes_table)
                
                # Display created accounts summary
                successful_accounts = report['account_creation']['account_details']
                if successful_accounts:
                    successful_account_count = len([a for a in successful_accounts if a['success']])
                    console.print(f"\n[bold cyan]📧 Created {successful_account_count} successful VALR accounts[/]")
                
                # Save results if requested
                if args.output_file or args.output_format != 'table':
                    saved_file = save_results(results, args.output_format, args.output_file)
                    if saved_file:
                        console.print(f"\n[green]💾 Results saved to: {saved_file}[/]")
                
                # Save detailed report
                try:
                    report_file = f"valr_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(report_file, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                    console.print(f"[green]📋 Detailed report saved to: {report_file}[/]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Could not save report: {e}[/]")
                
                # Save accounts to CSV
                if orchestrator.created_accounts:
                    try:
                        accounts_file = f"valr_created_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        accounts_data = []
                        for account in orchestrator.created_accounts:
                            accounts_data.append({
                                'email': account.email,
                                'password': account.password,
                                'success': account.success,
                                'error': account.error,
                                'proxy_used': account.proxy_used
                            })
                        
                        accounts_df = pd.DataFrame(accounts_data)
                        accounts_df.to_csv(accounts_file, index=False)
                        console.print(f"[green]📧 Created accounts saved to: {accounts_file}[/]")
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Could not save accounts: {e}[/]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Process interrupted by user[/]")
            return 1
        except Exception as e:
            console.print(f"\n[red]❌ Fatal error during processing: {e}[/]")
            logger.error(f"Fatal error occurred: {str(e)}\n{traceback.format_exc()}")
            return 1
        finally:
            # Cleanup
            with console.status("[dim]Cleaning up resources..."):
                await orchestrator.cleanup()
        
        # Success message
        console.rule("[bold green]Process Completed Successfully! 🎉[/]")
        console.print("\n[dim]Thank you for using the Enhanced VALR SMS Verification Tool with Account Creation![/]")
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Process interrupted by user[/]")
        return 1
    except Exception as e:
        console.print(f"\n[red]❌ Critical system error: {e}[/]")
        logger.error(f"Critical error occurred: {str(e)}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]💥 System crash: {e}[/]")
        sys.exit(1)

# Proxy pool manager with enhanced error handling (keeping existing implementation)
class ProxyPoolManager:
    """Manages proxy rotation and health checking with enhanced error handling"""
    
    def __init__(self, proxy_strings: List[str]):
        self.proxies: List[ProxyConfig] = []
        self.proxy_failures: Dict[str, int] = {}
        self.proxy_success_count: Dict[str, int] = {}
        self.current_index = 0
        self.total_requests = 0
        self.failed_proxies: List[str] = []
        
        # Parse proxy strings with error handling
        for proxy_string in proxy_strings:
            try:
                proxy = ProxyConfig.from_string(proxy_string.strip())
                self.proxies.append(proxy)
                self.proxy_failures[proxy.proxy_url] = 0
                self.proxy_success_count[proxy.proxy_url] = 0
                logger.info(f"Added proxy: {proxy.host}:{proxy.port} ({proxy.country})")
            except Exception as e:
                logger.warning(f"Failed to parse proxy: {proxy_string}, error: {e}")
        
        if not self.proxies:
            raise ProxyError("No valid proxies loaded")
        
        logger.info(f"Initialized proxy pool with {len(self.proxies)} valid proxies")
    
    def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next available proxy with error handling"""
        if not self.proxies:
            return None
        
        # Find a proxy with fewer than max failures
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            
            if self.proxy_failures[proxy.proxy_url] < config.proxy_pool.max_failures_per_proxy:
                self.total_requests += 1
                return proxy
            
            attempts += 1
        
        # If all proxies have max failures, reset failure counts and try again
        logger.warning("All proxies have max failures, resetting failure counts")
        self._reset_failure_counts()
        
        return self.proxies[0] if self.proxies else None
    
    def get_random_proxy(self) -> Optional[ProxyConfig]:
        """Get random available proxy with error handling"""
        if not self.proxies:
            return None
        
        available_proxies = [
            proxy for proxy in self.proxies
            if self.proxy_failures[proxy.proxy_url] < config.proxy_pool.max_failures_per_proxy
        ]
        
        if not available_proxies:
            self._reset_failure_counts()
            available_proxies = self.proxies
        
        if available_proxies:
            self.total_requests += 1
            return random.choice(available_proxies)
        
        return None
    
    def mark_proxy_failure(self, proxy: ProxyConfig, error: Exception = None):
        """Mark proxy as failed with detailed error tracking"""
        self.proxy_failures[proxy.proxy_url] += 1
        
        error_msg = str(error) if error else "Unknown error"
        logger.warning(f"Proxy failure: {proxy.host}:{proxy.port} "
                      f"(failures: {self.proxy_failures[proxy.proxy_url]}) - {error_msg}")
        
        # Mark proxy as completely failed if it exceeds max failures
        if self.proxy_failures[proxy.proxy_url] >= config.proxy_pool.max_failures_per_proxy:
            if proxy.proxy_url not in self.failed_proxies:
                self.failed_proxies.append(proxy.proxy_url)
                logger.error(f"Proxy permanently failed: {proxy.host}:{proxy.port}")
    
    def mark_proxy_success(self, proxy: ProxyConfig):
        """Mark proxy as successful (reset failure count)"""
        self.proxy_failures[proxy.proxy_url] = 0
        self.proxy_success_count[proxy.proxy_url] += 1
        logger.debug(f"Proxy success: {proxy.host}:{proxy.port} "
                    f"(successes: {self.proxy_success_count[proxy.proxy_url]})")
    
    def _reset_failure_counts(self):
        """Reset all failure counts"""
        for proxy_url in self.proxy_failures:
            self.proxy_failures[proxy_url] = 0
        self.failed_proxies.clear()
        logger.info("Reset all proxy failure counts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get proxy pool statistics"""
        total_proxies = len(self.proxies)
        active_proxies = len([p for p in self.proxies if self.proxy_failures[p.proxy_url] < config.proxy_pool.max_failures_per_proxy])
        failed_proxies = len(self.failed_proxies)
        
        return {
            'total_proxies': total_proxies,
            'active_proxies': active_proxies,
            'failed_proxies': failed_proxies,
            'total_requests': self.total_requests,
            'success_rate': self._calculate_overall_success_rate()
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate"""
        total_successes = sum(self.proxy_success_count.values())
        total_failures = sum(self.proxy_failures.values())
        total_operations = total_successes + total_failures
        
        return (total_successes / total_operations * 100) if total_operations > 0 else 0

# User agent management (keeping existing implementation)
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

# Email Generator
class EmailGenerator:
    """Generate random email addresses for account creation"""
    
    @staticmethod
    def generate_random_email() -> str:
        """Generate a random email address"""
        domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'protonmail.com', 'tempmail.org', '10minutemail.com'
        ]
        
        # Generate random username
        username_length = random.randint(8, 15)
        username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=username_length))
        
        # Add random numbers to make it more unique
        username += str(random.randint(100, 9999))
        
        domain = random.choice(domains)
        return f"{username}@{domain}"
    
    @staticmethod
    def generate_strong_password() -> str:
        """Generate a strong password"""
        # Ensure password has at least one of each required character type
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*"
        
        # Start with at least one character from each category
        password = [
            random.choice(lowercase),
            random.choice(uppercase),
            random.choice(digits),
            random.choice(special)
        ]
        
        # Fill remaining length with random characters
        all_chars = lowercase + uppercase + digits + special
        remaining_length = random.randint(8, 16) - 4
        password.extend(random.choices(all_chars, k=remaining_length))
        
        # Shuffle the password
        random.shuffle(password)
        return ''.join(password)

# Enhanced SMS service handler (keeping existing implementation with small modifications)
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
        self.retry_count = 0
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    @handle_errors({
        PlaywrightTimeout: "BROWSER_TIMEOUT",
        ConnectionError: "CONNECTION_ERROR",
        Exception: "INITIALIZATION_ERROR"
    }, max_retries=3)
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
            logger.error(f"Failed to initialize browser: {str(e)}")
            await self.cleanup()
            raise BrowserError(f"Browser initialization failed: {str(e)}", original_error=e)
    
    async def _apply_enhanced_stealth(self):
        """Apply stealth mode with error handling"""
        try:
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
            logger.debug("Stealth mode applied successfully")
        except Exception as e:
            logger.warning(f"Failed to apply stealth mode: {e}")
    
    def _extract_captcha_numbers(self, captcha_text: str) -> Optional[str]:
        """Extract captcha numbers with enhanced error handling"""
        if not captcha_text:
            return None
            
        try:
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
            return "10"  # Default fallback
            
        except Exception as e:
            logger.error(f"Error extracting captcha numbers: {e}")
            return "10"
    
    @handle_errors({
        PlaywrightTimeout: "LOGIN_TIMEOUT",
        CaptchaError: "CAPTCHA_ERROR",
        Exception: "LOGIN_ERROR"
    }, max_retries=5)
    async def login_to_sms_service(self) -> bool:
        try:
            if not self.sms_context:
                raise LoginError("SMS context not initialized")
            
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
            logger.error(f"Login failed: {str(e)}")
            if isinstance(e, (PlaywrightTimeout, LoginError, CaptchaError)):
                raise
            raise LoginError(f"Login failed: {str(e)}", original_error=e)
    
    async def _handle_enhanced_captcha(self):
        """Handle captcha with enhanced error handling"""
        try:
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
                    else:
                        logger.warning("Captcha input field not found")
            else:
                logger.debug("No captcha detected")
                
        except Exception as e:
            logger.error(f"Captcha handling failed: {e}")
            raise CaptchaError(f"Failed to handle captcha: {str(e)}", original_error=e)
    
    async def _submit_login_form(self):
        """Submit login form with error handling"""
        try:
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
                except Exception as e:
                    logger.debug(f"Failed to click submit button with selector {selector}: {e}")
                    continue
                    
            raise LoginError("No submit button found")
        except Exception as e:
            logger.error(f"Failed to submit login form: {e}")
            raise LoginError(f"Login form submission failed: {str(e)}", original_error=e)
    
    async def _verify_login_success(self) -> bool:
        """Verify login success with enhanced error handling"""
        try:
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
                            logger.error(f"Login error detected: {error_text}")
                            return False
                except:
                    continue
            
            success_indicators = ['dashboard', 'welcome', 'sms', 'messages']
            page_content = await self.sms_page.content()
            
            return any(indicator in page_content.lower() for indicator in success_indicators)
            
        except Exception as e:
            logger.error(f"Error verifying login success: {e}")
            return False
    
    @handle_errors({
        PlaywrightTimeout: "SMS_EXTRACTION_TIMEOUT",
        Exception: "SMS_EXTRACTION_ERROR"
    }, max_retries=3)
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
            logger.error(f"Failed to extract SMS data: {str(e)}")
            raise SMSServiceError(f"SMS data extraction failed: {str(e)}", original_error=e)
    
    @handle_errors({
        VerificationTimeoutError: "VERIFICATION_TIMEOUT",
        SMSServiceError: "SMS_SERVICE_ERROR",
        Exception: "VERIFICATION_ERROR"
    }, max_retries=2)
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
        
        try:
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
                error="Verification timeout",
                error_type="VERIFICATION_TIMEOUT"
            )
            
        except Exception as e:
            logger.error(f"Error getting verification code: {e}")
            return VerificationResult(
                phone_number=phone_number,
                service=service_type,
                success=False,
                error=str(e),
                error_type="VERIFICATION_ERROR"
            )
    
    def _normalize_phone_number(self, phone_number: str) -> str:
        """Normalize phone number with error handling"""
        try:
            return re.sub(r'[^\d]', '', phone_number)
        except Exception as e:
            logger.error(f"Error normalizing phone number {phone_number}: {e}")
            return phone_number
    
    def _phone_numbers_match(self, phone1: str, phone2: str) -> bool:
        """Check if phone numbers match with enhanced logic"""
        try:
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
        except Exception as e:
            logger.error(f"Error matching phone numbers {phone1} and {phone2}: {e}")
            return False
    
    def _is_valr_message(self, cli: str, sms_content: str) -> bool:
        """Check if message is from VALR with error handling"""
        try:
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
        except Exception as e:
            logger.error(f"Error checking VALR message: {e}")
            return False
    
    def _extract_verification_code_enhanced(self, sms_text: str) -> Optional[str]:
        """Extract verification code with enhanced error handling"""
        try:
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
        except Exception as e:
            logger.error(f"Error extracting verification code from '{sms_text}': {e}")
            return None
    
    async def _store_verification_result(
        self, 
        phone_number: str, 
        service_type: ServiceType, 
        code: str, 
        message: Dict[str, Any]
    ):
        """Store verification result with error handling"""
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
            logger.error(f"Failed to store verification result: {str(e)}")
    
    async def cleanup(self):
        """Cleanup with enhanced error handling"""
        cleanup_errors = []
        
        try:
            if self.sms_page:
                await self.sms_page.close()
        except Exception as e:
            cleanup_errors.append(f"Page cleanup: {e}")
        
        try:
            if self.sms_context:
                await self.sms_context.close()
        except Exception as e:
            cleanup_errors.append(f"Context cleanup: {e}")
        
        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            cleanup_errors.append(f"Browser cleanup: {e}")
        
        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            cleanup_errors.append(f"Playwright cleanup: {e}")
        
        if cleanup_errors:
            logger.warning(f"Cleanup errors: {'; '.join(cleanup_errors)}")
        else:
            logger.info("SMS service handler cleaned up successfully")


# Global variable to store VALRAutomator class for circular import resolution
_VALRAutomatorClass = None

def get_valr_automator_class():
    """Get VALRAutomator class, creating it if needed"""
    global _VALRAutomatorClass
    if _VALRAutomatorClass is None:
        # Import VALRAutomator class dynamically
        import sys
        current_module = sys.modules[__name__]
        _VALRAutomatorClass = getattr(current_module, 'VALRAutomator', None)
    return _VALRAutomatorClass

# Enhanced VALR Automator Class with Account Creation
class VALRAutomator:
    """Enhanced VALR automation with account creation and verification"""
    
    def __init__(self, proxy_pool_manager: Optional[ProxyPoolManager] = None):
        self.service_type = ServiceType.VALR
        self.user_agent_manager = UserAgentManager()
        self.proxy_pool_manager = proxy_pool_manager
        self.email_generator = EmailGenerator()
        self.success_count = 0
        self.failure_count = 0
        self.created_accounts: List[AccountCreationResult] = []
        
    @handle_errors({
        PlaywrightTimeout: "ACCOUNT_CREATION_TIMEOUT",
        ProxyError: "PROXY_ERROR",
        NetworkError: "NETWORK_ERROR",
        Exception: "ACCOUNT_CREATION_ERROR"
    }, max_retries=3)
    async def create_valr_account(
        self, 
        playwright_instance = None
    ) -> AccountCreationResult:
        """Create a new VALR account"""
        current_proxy = None
        browser = None
        context = None
        page = None
        
        try:
            # Generate account credentials
            email = self.email_generator.generate_random_email()
            password = self.email_generator.generate_strong_password()
            
            logger.info(f"Creating VALR account with email: {email}")
            
            user_agent = self.user_agent_manager.get_random_agent()
            headers = self.user_agent_manager.get_headers_for_agent(user_agent)
            
            # Get proxy if available
            if self.proxy_pool_manager:
                current_proxy = self.proxy_pool_manager.get_random_proxy()
                if current_proxy:
                    logger.info(f"Using proxy for account creation: {current_proxy.host}:{current_proxy.port}")
            
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
                "--disable-web-security"
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
                "timezone_id": "Africa/Cairo"
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
                
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """)
            
            page = await context.new_page()
            
            # Navigate to VALR signup page
            logger.info(f"Navigating to VALR signup: {config.services.valr_signup_url}")
            await page.goto(config.services.valr_signup_url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(3)  # Wait for page to stabilize
            
            # Take screenshot for debugging
            screenshot_path = f"valr_signup_{int(time.time())}.png"
            try:
                await page.screenshot(path=screenshot_path)
                logger.debug(f"Signup screenshot saved: {screenshot_path}")
            except:
                pass
            
            # Fill signup form
            await self._fill_signup_form(page, email, password)
            
            # Submit the form
            if not await self._submit_signup_form(page):
                raise AccountCreationError("Could not submit signup form")
            
            # Wait and check for success/error indicators
            await asyncio.sleep(5)
            
            # Analyze result
            creation_result = await self._analyze_signup_result(page, email, password)
            creation_result.proxy_used = f"{current_proxy.host}:{current_proxy.port}" if current_proxy else None
            
            if creation_result.success:
                # Store account in database
                await self._store_account_result(creation_result)
                self.created_accounts.append(creation_result)
                if current_proxy and self.proxy_pool_manager:
                    self.proxy_pool_manager.mark_proxy_success(current_proxy)
                logger.info(f"Successfully created VALR account: {email}")
            else:
                if current_proxy and self.proxy_pool_manager:
                    self.proxy_pool_manager.mark_proxy_failure(current_proxy)
            
            return creation_result
                
        except Exception as e:
            if current_proxy and self.proxy_pool_manager:
                self.proxy_pool_manager.mark_proxy_failure(current_proxy, e)
            
            error_type = "UNKNOWN_ERROR"
            if isinstance(e, PlaywrightTimeout):
                error_type = "TIMEOUT_ERROR"
            elif "proxy" in str(e).lower():
                error_type = "PROXY_ERROR"
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_type = "NETWORK_ERROR"
            
            logger.error(f"VALR account creation failed: {e}")
            return AccountCreationResult(
                success=False,
                email="",
                password="",
                error=str(e),
                error_type=error_type,
                proxy_used=f"{current_proxy.host}:{current_proxy.port}" if current_proxy else None
            )
        finally:
            # Cleanup resources
            cleanup_errors = []
            
            try:
                if page:
                    await page.close()
            except Exception as e:
                cleanup_errors.append(f"Page: {e}")
            
            try:
                if context:
                    await context.close()
            except Exception as e:
                cleanup_errors.append(f"Context: {e}")
            
            try:
                if browser:
                    await browser.close()
            except Exception as e:
                cleanup_errors.append(f"Browser: {e}")
            
            try:
                if should_close_playwright and playwright_instance:
                    await playwright_instance.stop()
            except Exception as e:
                cleanup_errors.append(f"Playwright: {e}")
            
            if cleanup_errors:
                logger.debug(f"Account creation cleanup errors: {'; '.join(cleanup_errors)}")
    
    async def _fill_signup_form(self, page: Page, email: str, password: str):
        """Fill the VALR signup form with intelligent field detection"""
        try:
            # Email field strategies
            email_selectors = [
                'input[type="email"]',
                'input[name*="email"]',
                'input[placeholder*="email"]',
                'input[placeholder*="Email"]',
                'input[id*="email"]',
                'input[class*="email"]',
                'input[data-testid*="email"]'
            ]
            
            email_filled = False
            for selector in email_selectors:
                try:
                    email_input = await page.wait_for_selector(selector, timeout=3000)
                    if email_input and await email_input.is_visible() and await email_input.is_enabled():
                        await email_input.clear()
                        await email_input.fill(email)
                        filled_value = await email_input.input_value()
                        if filled_value:
                            logger.info(f"Email filled using selector: {selector}")
                            email_filled = True
                            break
                except:
                    continue
            
            if not email_filled:
                raise AccountCreationError("Could not fill email field")
            
            # Password field strategies
            password_selectors = [
                'input[type="password"]',
                'input[name*="password"]',
                'input[placeholder*="password"]',
                'input[placeholder*="Password"]',
                'input[id*="password"]',
                'input[class*="password"]'
            ]
            
            password_fields = []
            for selector in password_selectors:
                try:
                    fields = await page.query_selector_all(selector)
                    for field in fields:
                        if await field.is_visible() and await field.is_enabled():
                            password_fields.append(field)
                except:
                    continue
            
            if not password_fields:
                raise AccountCreationError("Could not find password fields")
            
            # Fill password fields (usually password and confirm password)
            for i, field in enumerate(password_fields[:2]):  # Limit to first 2 password fields
                await field.clear()
                await field.fill(password)
                logger.info(f"Password field {i+1} filled")
            
            # Handle checkbox/terms agreement if present
            try:
                checkbox_selectors = [
                    'input[type="checkbox"]',
                    'input[name*="terms"]',
                    'input[name*="agree"]',
                    'input[id*="terms"]',
                    'input[id*="agree"]'
                ]
                
                for selector in checkbox_selectors:
                    try:
                        checkbox = await page.wait_for_selector(selector, timeout=2000)
                        if checkbox and await checkbox.is_visible():
                            if not await checkbox.is_checked():
                                await checkbox.click()
                                logger.info(f"Checked agreement checkbox: {selector}")
                    except:
                        continue
            except Exception as e:
                logger.debug(f"No agreement checkbox found or failed to check: {e}")
            
            logger.info("Signup form filled successfully")
            
        except Exception as e:
            logger.error(f"Failed to fill signup form: {e}")
            raise AccountCreationError(f"Form filling failed: {str(e)}", original_error=e)
    
    async def _submit_signup_form(self, page: Page) -> bool:
        """Submit the signup form with intelligent button detection"""
        try:
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Sign Up")',
                'button:has-text("Create Account")',
                'button:has-text("Register")',
                'button:has-text("Join")',
                'button:has-text("Get Started")',
                'button[data-testid*="submit"]',
                'button[data-testid*="signup"]',
                '.signup-button',
                '.submit-button',
                '.register-button'
            ]
            
            for selector in submit_selectors:
                try:
                    button = await page.wait_for_selector(selector, timeout=3000)
                    if button:
                        is_visible = await button.is_visible()
                        is_enabled = await button.is_enabled()
                        
                        if is_visible and is_enabled:
                            await button.click()
                            logger.info(f"Successfully clicked signup button using selector: {selector}")
                            await page.wait_for_load_state('networkidle', timeout=10000)
                            return True
                except:
                    continue
            
            # Try form submission via Enter key
            try:
                email_input = await page.query_selector('input[type="email"]')
                if email_input:
                    await email_input.press('Enter')
                    logger.info("Attempted form submission via Enter key")
                    await page.wait_for_load_state('networkidle', timeout=10000)
                    return True
            except:
                pass
            
            logger.warning("Could not find or click signup button")
            return False
            
        except Exception as e:
            logger.error(f"Failed to submit signup form: {e}")
            return False
    
    async def _analyze_signup_result(self, page: Page, email: str, password: str) -> AccountCreationResult:
        """Analyze the result of signup form submission"""
        try:
            current_url = page.url
            page_content = await page.content()
            
            # Take result screenshot
            screenshot_path = f"valr_signup_result_{int(time.time())}.png"
            try:
                await page.screenshot(path=screenshot_path)
                logger.debug(f"Signup result screenshot saved: {screenshot_path}")
            except:
                pass
            
            # Check for success indicators
            success_indicators = [
                'welcome',
                'account created',
                'registration successful',
                'verify your email',
                'check your email',
                'confirmation sent',
                'dashboard',
                'profile',
                'verification'
            ]
            
            # Check for error indicators
            error_indicators = [
                'error',
                'invalid',
                'failed',
                'already exists',
                'email already',
                'user already',
                'try again',
                'registration failed'
            ]
            
            page_content_lower = page_content.lower()
            
            # Check for success
            for indicator in success_indicators:
                if indicator in page_content_lower:
                    logger.info(f"Account creation success indicator found: {indicator}")
                    return AccountCreationResult(
                        success=True,
                        email=email,
                        password=password
                    )
            
            # Check for specific error messages
            for indicator in error_indicators:
                if indicator in page_content_lower:
                    return AccountCreationResult(
                        success=False,
                        email=email,
                        password=password,
                        error=f"Error detected: {indicator}",
                        error_type="SIGNUP_FORM_ERROR"
                    )
            
            # Check if URL changed (might indicate success)
            if current_url != config.services.valr_signup_url:
                if any(keyword in current_url for keyword in ['dashboard', 'verify', 'welcome', 'profile']):
                    return AccountCreationResult(
                        success=True,
                        email=email,
                        password=password
                    )
            
            # Default to success if no clear error indicators
            return AccountCreationResult(
                success=True,
                email=email,
                password=password
            )
            
        except Exception as e:
            return AccountCreationResult(
                success=False,
                email=email,
                password=password,
                error=f"Analysis failed: {str(e)}",
                error_type="ANALYSIS_ERROR"
            )
    
    async def _store_account_result(self, result: AccountCreationResult):
        """Store account creation result in database"""
        try:
            with SessionLocal() as db:
                account = VALRAccount(
                    email=result.email,
                    password=result.password,
                    proxy_used=result.proxy_used,
                    session_id=str(uuid.uuid4())
                )
                
                db.add(account)
                db.commit()
                logger.debug("Stored account creation result in database")
                
        except Exception as e:
            logger.error(f"Failed to store account result: {str(e)}")
    
    @handle_errors({
        PlaywrightTimeout: "VALR_TIMEOUT",
        ProxyError: "PROXY_ERROR",
        NetworkError: "NETWORK_ERROR",
        Exception: "VALR_ERROR"
    }, max_retries=3)
    async def request_verification_code_with_account(
        self, 
        phone_number: str, 
        account_result: AccountCreationResult,
        country: str = "EG",
        playwright_instance = None
    ) -> VerificationResult:
        """Request VALR verification code using a created account"""
        current_proxy = None
        browser = None
        context = None
        page = None
        
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
                "--disable-web-security"
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
                "timezone_id": "Africa/Cairo"
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
                
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """)
            
            page = await context.new_page()
            
            # First login to the account if needed
            login_success = await self._login_to_account(page, account_result)
            if not login_success:
                logger.warning("Could not login to account, proceeding without login")
            
            # Navigate to phone verification page
            verify_url = f"{config.services.valr_verify_url}?country={country}"
            logger.info(f"Navigating to VALR phone verification: {verify_url}")
            
            await page.goto(verify_url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(2)
            
            # Take screenshot for debugging
            screenshot_path = f"valr_verify_{int(time.time())}.png"
            try:
                await page.screenshot(path=screenshot_path)
                logger.debug(f"Verification screenshot saved: {screenshot_path}")
            except:
                pass
            
            # Fill phone number
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
            result.account_email = account_result.email
            
            if result.success:
                self.success_count += 1
                if current_proxy and self.proxy_pool_manager:
                    self.proxy_pool_manager.mark_proxy_success(current_proxy)
                logger.info(f"Successfully requested VALR verification for {phone_number} using account {account_result.email}")
            else:
                self.failure_count += 1
                if current_proxy and self.proxy_pool_manager:
                    self.proxy_pool_manager.mark_proxy_failure(current_proxy)
            
            return result
                
        except Exception as e:
            self.failure_count += 1
            if current_proxy and self.proxy_pool_manager:
                self.proxy_pool_manager.mark_proxy_failure(current_proxy, e)
            
            error_type = "UNKNOWN_ERROR"
            if isinstance(e, PlaywrightTimeout):
                error_type = "TIMEOUT_ERROR"
            elif "proxy" in str(e).lower():
                error_type = "PROXY_ERROR"
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                error_type = "NETWORK_ERROR"
            
            logger.error(f"VALR verification request failed for {phone_number}: {e}")
            return VerificationResult(
                phone_number=phone_number,
                service=ServiceType.VALR,
                success=False,
                error=str(e),
                error_type=error_type,
                proxy_used=f"{current_proxy.host}:{current_proxy.port}" if current_proxy else None,
                account_email=account_result.email
            )
        finally:
            # Cleanup resources
            cleanup_errors = []
            
            try:
                if page:
                    await page.close()
            except Exception as e:
                cleanup_errors.append(f"Page: {e}")
            
            try:
                if context:
                    await context.close()
            except Exception as e:
                cleanup_errors.append(f"Context: {e}")
            
            try:
                if browser:
                    await browser.close()
            except Exception as e:
                cleanup_errors.append(f"Browser: {e}")
            
            try:
                if should_close_playwright and playwright_instance:
                    await playwright_instance.stop()
            except Exception as e:
                cleanup_errors.append(f"Playwright: {e}")
            
            if cleanup_errors:
                logger.debug(f"VALR cleanup errors: {'; '.join(cleanup_errors)}")
    
    async def _login_to_account(self, page: Page, account_result: AccountCreationResult) -> bool:
        """Login to the created VALR account"""
        try:
            # Navigate to login page
            login_url = "https://www.valr.com/en/login"
            await page.goto(login_url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(2)
            
            # Fill login form
            email_input = await page.wait_for_selector('input[type="email"]', timeout=10000)
            if email_input:
                await email_input.fill(account_result.email)
            
            password_input = await page.wait_for_selector('input[type="password"]', timeout=10000)
            if password_input:
                await password_input.fill(account_result.password)
            
            # Submit login
            login_button = await page.wait_for_selector('button[type="submit"]', timeout=10000)
            if login_button:
                await login_button.click()
                await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Check if login was successful
            current_url = page.url
            if 'dashboard' in current_url or 'profile' in current_url or 'login' not in current_url:
                logger.info("Successfully logged into VALR account")
                return True
            else:
                logger.warning("Login may have failed or requires additional verification")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to login to account: {e}")
            return False
    
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
                buttons = await page.query_selector_all('button')
                for button in buttons:
                    button_text = await button.text_content()
                    if button_text and text.lower() in button_text.lower():
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
                        error=f"Error detected: {indicator}",
                        error_type="VALR_FORM_ERROR"
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
                error=f"Analysis failed: {str(e)}",
                error_type="ANALYSIS_ERROR"
            )
    
    def _format_phone_number(self, phone_number: str, country: str) -> str:
        """Format phone number for VALR (Egypt focused)"""
        
        try:
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
        
        except Exception as e:
            logger.error(f"Error formatting phone number {phone_number}: {e}")
            return phone_number


# Interactive Proxy Configuration (keeping existing implementation)
def interactive_proxy_setup() -> Tuple[bool, List[str]]:
    """Interactive proxy setup with enhanced options"""
    console.print("\n" + "="*60)
    console.print("[bold blue]🌐 PROXY CONFIGURATION[/]", justify="center")
    console.print("="*60)
    
    proxy_strings = []
    
    # Ask if user wants to use proxies
    use_proxy = Confirm.ask(
        "\n[bold yellow]Do you want to use proxies for this session?[/]",
        default=False
    )
    
    if not use_proxy:
        console.print("[green]✅ Running without proxies[/]")
        return False, []
    
    console.print("\n[bold cyan]Proxy Setup Options:[/]")
    console.print("1. Load from file")
    console.print("2. Enter proxy manually")
    console.print("3. Use default example proxy")
    
    choice = Prompt.ask(
        "\n[bold yellow]Choose an option[/]",
        choices=["1", "2", "3"],
        default="1"
    )
    
    if choice == "1":
        # Load from file
        default_files = ['proxies.txt', 'proxy.txt', 'proxy_list.txt']
        found_files = [f for f in default_files if os.path.exists(f)]
        
        if found_files:
            console.print(f"\n[green]Found proxy files: {', '.join(found_files)}[/]")
            file_choice = Prompt.ask(
                "Enter filename or press Enter for first found file",
                default=found_files[0]
            )
        else:
            file_choice = Prompt.ask(
                "\n[yellow]No default proxy files found. Enter proxy file path[/]",
                default="proxies.txt"
            )
        
        try:
            if os.path.exists(file_choice):
                with open(file_choice, 'r') as f:
                    proxy_strings = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                console.print(f"[green]✅ Loaded {len(proxy_strings)} proxies from {file_choice}[/]")
            else:
                console.print(f"[red]❌ File not found: {file_choice}[/]")
                return interactive_proxy_setup()  # Retry
        except Exception as e:
            console.print(f"[red]❌ Error reading proxy file: {e}[/]")
            return interactive_proxy_setup()  # Retry
    
    elif choice == "2":
        # Enter manually
        console.print("\n[bold cyan]Enter proxy in format: host:port:username:password|country|region[/]")
        console.print("[dim]Example: 163.172.65.133:48103:6c65:6c65|United Kingdom|England[/]")
        
        while True:
            proxy_input = Prompt.ask("\n[yellow]Enter proxy (or 'done' to finish)[/]")
            
            if proxy_input.lower() == 'done':
                break
            
            try:
                # Validate proxy format
                ProxyConfig.from_string(proxy_input)
                proxy_strings.append(proxy_input)
                console.print(f"[green]✅ Added proxy: {proxy_input.split('|')[0]}[/]")
            except Exception as e:
                console.print(f"[red]❌ Invalid proxy format: {e}[/]")
                continue
        
        if not proxy_strings:
            console.print("[yellow]⚠️ No proxies added. Running without proxies.[/]")
            return False, []
    
    elif choice == "3":
        # Use default example
        proxy_strings = ["163.172.65.133:48103:6c65:6c65|United Kingdom|England|Huddersfield"]
        console.print("[green]✅ Using default example proxy[/]")
    
    # Validate all proxies
    valid_proxies = []
    for proxy_string in proxy_strings:
        try:
            ProxyConfig.from_string(proxy_string)
            valid_proxies.append(proxy_string)
        except Exception as e:
            console.print(f"[red]❌ Invalid proxy removed: {proxy_string} - {e}[/]")
    
    if not valid_proxies:
        console.print("[red]❌ No valid proxies found. Would you like to try again?[/]")
        if Confirm.ask("Try again?", default=True):
            return interactive_proxy_setup()
        else:
            return False, []
    
    # Display proxy summary
    proxy_table = Table(title="Loaded Proxies")
    proxy_table.add_column("Host:Port", style="cyan")
    proxy_table.add_column("Country", style="green")
    proxy_table.add_column("Region", style="yellow")
    
    for proxy_string in valid_proxies:
        try:
            proxy = ProxyConfig.from_string(proxy_string)
            proxy_table.add_row(
                f"{proxy.host}:{proxy.port}",
                proxy.country or "Unknown",
                proxy.region or "Unknown"
            )
        except:
            continue
    
    console.print(proxy_table)
    
    if Confirm.ask(f"\n[bold green]Use these {len(valid_proxies)} proxies?[/]", default=True):
        return True, valid_proxies
    else:
        return interactive_proxy_setup()


# Enhanced Verification Orchestrator with Account Creation
class EnhancedVerificationOrchestrator:
    """Orchestrates the entire VALR verification process with account creation and proxy support"""
    
    def __init__(self, proxy_strings: List[str] = None, use_proxies: bool = True):
        self.sms_handler = None
        self.proxy_pool_manager = None
        self.valr_automator = None
        self.results: List[VerificationResult] = []
        self.created_accounts: List[AccountCreationResult] = []
        self.session_id = str(uuid.uuid4())
        self.use_proxies = use_proxies
        self.error_counts = {
            'total_errors': 0,
            'proxy_errors': 0,
            'network_errors': 0,
            'timeout_errors': 0,
            'verification_errors': 0,
            'account_creation_errors': 0,
            'other_errors': 0
        }
        
        # Initialize proxy pool if proxy strings provided and proxies are enabled
        if proxy_strings and use_proxies:
            try:
                self.proxy_pool_manager = ProxyPoolManager(proxy_strings)
                logger.info(f"Initialized with {len(proxy_strings)} proxies")
            except Exception as e:
                logger.error(f"Failed to initialize proxy pool: {e}")
                self.proxy_pool_manager = None
        
        # Initialize VALR automator with proxy support
        self.valr_automator = VALRAutomator(self.proxy_pool_manager)
        
    async def initialize(self, username: str, password: str):
        """Initialize the orchestrator with enhanced error handling"""
        try:
            self.sms_handler = EnhancedSMSServiceHandler(username, password)
            await self.sms_handler.__aenter__()
            
            # Login to SMS service
            if not await self.sms_handler.login_to_sms_service():
                raise LoginError("Failed to login to SMS service")
            
            logger.info("Enhanced VALR verification orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.cleanup()
            raise
    
    @handle_errors({
        SMSServiceError: "SMS_SERVICE_ERROR",
        NetworkError: "NETWORK_ERROR",
        Exception: "PROCESSING_ERROR"
    }, max_retries=1)
    async def process_phone_numbers_with_account_creation(
        self, 
        phone_data: List[Dict[str, str]]
    ) -> List[VerificationResult]:
        """Process multiple phone numbers for VALR verification with account creation"""
        
        results = []
        
        try:
            async with async_playwright() as playwright:
                # Process numbers with controlled concurrency
                semaphore = asyncio.Semaphore(2)  # Limit concurrent requests
                
                tasks = [
                    self._process_single_number_with_account(
                        phone_info, playwright, semaphore
                    )
                    for phone_info in phone_data
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        self.error_counts['total_errors'] += 1
                        self.error_counts['other_errors'] += 1
                        
                        # Create a failed result for the exception
                        failed_result = VerificationResult(
                            phone_number="unknown",
                            service=ServiceType.VALR,
                            success=False,
                            error=str(result),
                            error_type="TASK_EXCEPTION"
                        )
                        results.append(failed_result)
                    else:
                        results.append(result)
                        self.results.append(result)
                        
                        # Track error statistics
                        if not result.success:
                            self.error_counts['total_errors'] += 1
                            if result.error_type:
                                if 'PROXY' in result.error_type:
                                    self.error_counts['proxy_errors'] += 1
                                elif 'NETWORK' in result.error_type:
                                    self.error_counts['network_errors'] += 1
                                elif 'TIMEOUT' in result.error_type:
                                    self.error_counts['timeout_errors'] += 1
                                elif 'VERIFICATION' in result.error_type:
                                    self.error_counts['verification_errors'] += 1
                                elif 'ACCOUNT' in result.error_type:
                                    self.error_counts['account_creation_errors'] += 1
                                else:
                                    self.error_counts['other_errors'] += 1
        
        except Exception as e:
            logger.error(f"Critical error in processing phone numbers: {e}")
            raise
        
        return results
    
    async def _process_single_number_with_account(
        self,
        phone_info: Dict[str, str],
        playwright_instance,
        semaphore: asyncio.Semaphore
    ) -> VerificationResult:
        """Process a single phone number with account creation and enhanced error handling"""
        
        phone_number = phone_info.get('Number', phone_info.get('phone', ''))
        country = phone_info.get('Range', phone_info.get('Country', phone_info.get('country', 'EG')))
        
        async with semaphore:
            try:
                logger.info(f"Processing {phone_number} for VALR verification with account creation")
                
                # Step 1: Create VALR account
                account_result = await self.valr_automator.create_valr_account(playwright_instance)
                
                if not account_result.success:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=False,
                        error=f"Account creation failed: {account_result.error}",
                        error_type="ACCOUNT_CREATION_FAILED",
                        proxy_used=account_result.proxy_used
                    )
                
                logger.info(f"Successfully created account: {account_result.email}")
                self.created_accounts.append(account_result)
                
                # Step 2: Request verification code from VALR using the created account
                request_result = await self.valr_automator.request_verification_code_with_account(
                    phone_number, account_result, country, playwright_instance
                )
                
                if not request_result.success:
                    return request_result
                
                # Step 3: Wait for SMS verification code
                verification_result = await self.sms_handler.get_verification_code_intelligent(
                    phone_number, ServiceType.VALR
                )
                
                if verification_result and verification_result.success:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=True,
                        code=verification_result.code,
                        proxy_used=request_result.proxy_used,
                        account_email=account_result.email
                    )
                else:
                    return VerificationResult(
                        phone_number=phone_number,
                        service=ServiceType.VALR,
                        success=False,
                        error=verification_result.error if verification_result else "No verification result",
                        error_type="VERIFICATION_FAILED",
                        proxy_used=request_result.proxy_used,
                        account_email=account_result.email
                    )
                    
            except Exception as e:
                error_type = "PROCESSING_ERROR"
                if isinstance(e, SMSServiceError):
                    error_type = e.error_type
                elif isinstance(e, (NetworkError, ConnectionError)):
                    error_type = "NETWORK_ERROR"
                elif isinstance(e, (PlaywrightTimeout, VerificationTimeoutError)):
                    error_type = "TIMEOUT_ERROR"
                
                logger.error(f"Error processing {phone_number}: {e}")
                return VerificationResult(
                    phone_number=phone_number,
                    service=ServiceType.VALR,
                    success=False,
                    error=str(e),
                    error_type=error_type
                )
            finally:
                # Rate limiting
                try:
                    delay = random.uniform(*config.services.rate_limit_delay)
                    await asyncio.sleep(delay)
                except:
                    await asyncio.sleep(3)  # Fallback delay
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report with enhanced error analysis and account creation stats"""
        if not self.results:
            return {'status': 'No results to report'}
        
        total_attempts = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_attempts - successful
        success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0
        
        # Account creation statistics
        total_accounts_created = len(self.created_accounts)
        successful_accounts = sum(1 for a in self.created_accounts if a.success)
        account_creation_rate = (successful_accounts / total_accounts_created) * 100 if total_accounts_created > 0 else 0
        
        # Enhanced error analysis
        error_types = {}
        error_details = {}
        
        for result in self.results:
            if not result.success and result.error:
                error_type = result.error_type or "UNKNOWN_ERROR"
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Store detailed error for analysis
                if error_type not in error_details:
                    error_details[error_type] = []
                error_details[error_type].append({
                    'phone': result.phone_number,
                    'error': result.error[:100],  # Truncate long errors
                    'proxy': result.proxy_used,
                    'account': result.account_email
                })
        
        # Proxy statistics with enhanced metrics
        proxy_stats = {}
        if self.proxy_pool_manager:
            for result in self.results:
                if result.proxy_used:
                    if result.proxy_used not in proxy_stats:
                        proxy_stats[result.proxy_used] = {
                            'total': 0, 
                            'success': 0, 
                            'errors': {},
                            'success_rate': 0
                        }
                    
                    proxy_stats[result.proxy_used]['total'] += 1
                    if result.success:
                        proxy_stats[result.proxy_used]['success'] += 1
                    else:
                        error_type = result.error_type or 'UNKNOWN'
                        proxy_stats[result.proxy_used]['errors'][error_type] = \
                            proxy_stats[result.proxy_used]['errors'].get(error_type, 0) + 1
            
            # Calculate success rates
            for proxy, stats in proxy_stats.items():
                stats['success_rate'] = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        # Get proxy pool statistics if available
        proxy_pool_stats = None
        if self.proxy_pool_manager:
            proxy_pool_stats = self.proxy_pool_manager.get_statistics()
        
        # Successful codes with enhanced details
        successful_codes = [
            {
                'phone_number': r.phone_number,
                'service': r.service.value,
                'code': r.code,
                'timestamp': r.timestamp.isoformat(),
                'proxy_used': r.proxy_used,
                'account_email': r.account_email
            }
            for r in self.results if r.success and r.code
        ]
        
        # Account creation details
        account_details = [
            {
                'email': a.email,
                'success': a.success,
                'error': a.error,
                'proxy_used': a.proxy_used
            }
            for a in self.created_accounts
        ]
        
        # Time-based analysis
        if self.results:
            processing_times = []
            start_time = min(r.timestamp for r in self.results)
            end_time = max(r.timestamp for r in self.results)
            total_processing_time = (end_time - start_time).total_seconds()
        else:
            total_processing_time = 0
        
        return {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_seconds': total_processing_time,
            'summary': {
                'total_attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'success_rate': round(success_rate, 2)
            },
            'account_creation': {
                'total_accounts_created': total_accounts_created,
                'successful_accounts': successful_accounts,
                'account_creation_rate': round(account_creation_rate, 2),
                'account_details': account_details
            },
            'error_analysis': {
                'error_counts': self.error_counts,
                'error_types': error_types,
                'error_details': error_details
            },
            'proxy_statistics': proxy_stats,
            'proxy_pool_stats': proxy_pool_stats,
            'successful_verifications': successful_codes,
            'configuration': {
                'proxies_enabled': self.use_proxies,
                'proxy_count': len(self.proxy_pool_manager.proxies) if self.proxy_pool_manager else 0,
                'browser_headless': config.browser.headless,
                'verification_timeout': config.services.verification_timeout
            }
        }
    
    async def cleanup(self):
        """Enhanced cleanup with error handling"""
        cleanup_errors = []
        
        try:
            if self.sms_handler:
                await self.sms_handler.__aexit__(None, None, None)
        except Exception as e:
            cleanup_errors.append(f"SMS handler: {e}")
        
        if cleanup_errors:
            logger.warning(f"Cleanup errors: {'; '.join(cleanup_errors)}")
        else:
            logger.info("Orchestrator cleanup completed successfully")


# CSV Processing (keeping original implementation with enhanced error handling)
class CSVProcessor:
    """Enhanced CSV processing with better error handling"""
    
    @staticmethod
    def load_and_validate_csv(file_path: str) -> pd.DataFrame:
        """Load and validate CSV file with enhanced error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    encoding_used = encoding
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load CSV with {encoding}: {e}")
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
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
                    if len(df) > 0:
                        sample_value = str(df[col].iloc[0])
                        if re.search(r'\d{8,}', sample_value):
                            phone_col = col
                            break
            
            if not phone_col:
                available_columns = list(df.columns)
                raise ValueError(f"No phone number column found. Available columns: {available_columns}")
            
            # Rename to standard name
            if phone_col != 'Number':
                df = df.rename(columns={phone_col: 'Number'})
            
            # Add default country if not present
            if 'Country' not in df.columns and 'Range' not in df.columns:
                df['Country'] = 'EG'  # Default to Egypt
            
            # Clean and validate phone numbers
            original_count = len(df)
            
            # Convert to string and strip whitespace
            df['Number'] = df['Number'].astype(str).str.strip()
            
            # Remove rows with invalid phone numbers
            df = df[df['Number'].str.len() > 5]
            df = df[df['Number'] != 'nan']
            df = df[df['Number'] != '']
            df = df.dropna(subset=['Number'])
            df = df.drop_duplicates(subset=['Number'])
            
            final_count = len(df)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} invalid/duplicate phone numbers")
            
            if final_count == 0:
                raise ValueError("No valid phone numbers found in CSV")
            
            logger.info(f"Loaded {final_count} valid phone numbers from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file '{file_path}': {e}")
            raise
    
    @staticmethod
    def find_csv_files(directory: str = ".") -> List[str]:
        """Find CSV files in directory with error handling"""
        try:
            patterns = ["*.csv"]
            found_files = []
            
            for pattern in patterns:
                files = glob.glob(os.path.join(directory, pattern))
                found_files.extend(files)
            
            unique_files = list(set(found_files))
            unique_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            return unique_files
        except Exception as e:
            logger.error(f"Error finding CSV files in '{directory}': {e}")
            return []





def load_proxy_strings(proxy_file: str) -> List[str]:
    """Load proxy strings from file with enhanced error handling"""
    try:
        if not os.path.exists(proxy_file):
            raise FileNotFoundError(f"Proxy file not found: {proxy_file}")
        
        with open(proxy_file, 'r') as f:
            proxy_strings = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # Validate proxy format
                        ProxyConfig.from_string(line)
                        proxy_strings.append(line)
                    except Exception as e:
                        logger.warning(f"Invalid proxy on line {line_num}: {line} - {e}")
        
        if not proxy_strings:
            raise ValueError("No valid proxies found in file")
        
        logger.info(f"Loaded {len(proxy_strings)} valid proxies from {proxy_file}")
        return proxy_strings
        
    except Exception as e:
        logger.error(f"Failed to load proxy file '{proxy_file}': {e}")
        raise


def display_results_table(results: List[VerificationResult]):
    """Display results in a formatted table with enhanced information"""
    if not results:
        console.print("[yellow]No results to display[/]")
        return
    
    table = Table(title="VALR Verification Results")
    
    table.add_column("Phone Number", style="cyan", no_wrap=True)
    table.add_column("Status", style="green", justify="center")
    table.add_column("Code", style="yellow", justify="center")
    table.add_column("Account Email", style="blue", no_wrap=True, max_width=25)
    table.add_column("Proxy", style="magenta", no_wrap=True)
    table.add_column("Error Type", style="red", no_wrap=True)
    table.add_column("Error", style="red", max_width=30)
    table.add_column("Time", style="blue", no_wrap=True)
    
    for result in results:
        status = "✅ Success" if result.success else "❌ Failed"
        code = result.code or "-"
        account_email = result.account_email[:22] + "..." if result.account_email and len(result.account_email) > 25 else (result.account_email or "-")
        proxy = result.proxy_used or "-"
        error_type = result.error_type or "-"
        error = result.error[:30] + "..." if result.error and len(result.error) > 30 else (result.error or "-")
        timestamp = result.timestamp.strftime("%H:%M:%S") if result.timestamp else "-"
        
        # Color code based on status
        status_style = "green" if result.success else "red"
        
        table.add_row(
            result.phone_number,
            f"[{status_style}]{status}[/]",
            code,
            account_email,
            proxy,
            error_type,
            error,
            timestamp
        )
    
    console.print(table)
    
    # Display summary
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    success_rate = (successful / total * 100) if total > 0 else 0
    
    summary_panel = Panel(
        f"[bold green]Successful: {successful}[/] | "
        f"[bold red]Failed: {failed}[/] | "
        f"[bold blue]Success Rate: {success_rate:.1f}%[/]",
        title="Summary",
        border_style="blue"
    )
    console.print(summary_panel)


def display_account_creation_table(accounts: List[AccountCreationResult]):
    """Display account creation results in a formatted table"""
    if not accounts:
        console.print("[yellow]No account creation results to display[/]")
        return
    
    table = Table(title="VALR Account Creation Results")
    
    table.add_column("Email", style="cyan", max_width=30)
    table.add_column("Status", style="green", justify="center")
    table.add_column("Proxy", style="magenta", no_wrap=True)
    table.add_column("Error", style="red", max_width=40)
    
    for account in accounts:
        status = "✅ Success" if account.success else "❌ Failed"
        email = account.email[:27] + "..." if len(account.email) > 30 else account.email
        proxy = account.proxy_used or "-"
        error = account.error[:37] + "..." if account.error and len(account.error) > 40 else (account.error or "-")
        
        # Color code based on status
        status_style = "green" if account.success else "red"
        
        table.add_row(
            email,
            f"[{status_style}]{status}[/]",
            proxy,
            error
        )
    
    console.print(table)


def display_error_analysis(results: List[VerificationResult]):
    """Display detailed error analysis"""
    failed_results = [r for r in results if not r.success]
    
    if not failed_results:
        console.print("[green]🎉 No errors to analyze![/]")
        return
    
    # Error type analysis
    error_types = {}
    for result in failed_results:
        error_type = result.error_type or "UNKNOWN_ERROR"
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(result)
    
    # Display error breakdown
    error_table = Table(title="Error Analysis")
    error_table.add_column("Error Type", style="red")
    error_table.add_column("Count", style="blue", justify="center")
    error_table.add_column("Percentage", style="yellow", justify="center")
    error_table.add_column("Example", style="dim", max_width=40)
    
    total_errors = len(failed_results)
    
    for error_type, error_results in error_types.items():
        count = len(error_results)
        percentage = (count / total_errors * 100)
        example = error_results[0].error[:40] + "..." if error_results[0].error and len(error_results[0].error) > 40 else (error_results[0].error or "No details")
        
        error_table.add_row(
            error_type,
            str(count),
            f"{percentage:.1f}%",
            example
        )
    
    console.print(error_table)


def save_results(results: List[VerificationResult], format: str, file_path: str = None):
    """Save results in specified format with enhanced data"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            file_path = file_path or f'valr_verification_results_{timestamp}.json'
            data = []
            for r in results:
                data.append({
                    'phone_number': r.phone_number,
                    'service': r.service.value,
                    'success': r.success,
                    'code': r.code,
                    'proxy_used': r.proxy_used,
                    'account_email': r.account_email,
                    'error': r.error,
                    'error_type': r.error_type,
                    'retry_count': r.retry_count,
                    'timestamp': r.timestamp.isoformat() if r.timestamp else None
                })
            
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
                    'account_email': r.account_email,
                    'error': r.error,
                    'error_type': r.error_type,
                    'retry_count': r.retry_count,
                    'timestamp': r.timestamp.isoformat() if r.timestamp else None
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        
        console.print(f"[green]✅ Results saved to {file_path}[/]")
        return file_path
        
    except Exception as e:
        console.print(f"[red]❌ Failed to save results: {e}[/]")
        logger.error(f"Failed to save results: {e}")
        return None


# Main execution block
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]💥 System crash: {e}[/]")
        sys.exit(1)
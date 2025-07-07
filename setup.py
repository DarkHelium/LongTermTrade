#!/usr/bin/env python3
"""
Long-Term Investment Bot Setup Script

This script helps set up the long-term investment bot environment.
It creates necessary directories, checks dependencies, and validates configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> bool:
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'requests', 'PyYAML', 'python-dotenv',
        'alpaca-trade-api', 'alpha-vantage', 'yfinance', 'vaderSentiment',
        'beautifulsoup4', 'scipy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages


def install_dependencies(missing_packages: List[str]) -> bool:
    """Install missing dependencies."""
    if not missing_packages:
        return True
    
    print(f"\nInstalling {len(missing_packages)} missing packages...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
        ])
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        'logs',
        'data',
        'backtest_results',
        'tests/__pycache__',
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False


def setup_environment_file() -> bool:
    """Set up environment file from template."""
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if not env_template.exists():
        print("❌ .env.template file not found")
        return False
    
    if env_file.exists():
        print("⚠️  .env file already exists, skipping creation")
        return True
    
    try:
        shutil.copy(env_template, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def validate_config() -> bool:
    """Validate configuration file."""
    config_file = Path('config.yml')
    
    if not config_file.exists():
        print("❌ config.yml file not found")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_sections = [
            'allocation', 'core_etfs', 'stock_selection',
            'scoring_weights', 'hard_filters', 'sell_rules'
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required config section: {section}")
                return False
        
        print("✅ Configuration file validated")
        return True
    except Exception as e:
        print(f"❌ Failed to validate config: {e}")
        return False


def check_code_formatting() -> bool:
    """Check if code formatting tools are available."""
    tools = ['black', 'ruff']
    available_tools = []
    
    for tool in tools:
        try:
            subprocess.run([tool, '--version'], 
                         capture_output=True, check=True)
            available_tools.append(tool)
            print(f"✅ {tool} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {tool} not available (optional)")
    
    return len(available_tools) > 0


def run_basic_tests() -> bool:
    """Run basic import tests."""
    print("\nRunning basic import tests...")
    
    modules = [
        'data', 'scoring', 'portfolio', 'broker', 
        'sentiment', 'backtest', 'main'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}.py imports successfully")
        except ImportError as e:
            print(f"❌ {module}.py import failed: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! Next Steps:")
    print("="*60)
    print()
    print("1. 📝 Edit .env file and add your API keys:")
    print("   - Alpha Vantage API key (free at alphavantage.co)")
    print("   - Alpaca paper trading credentials (free at alpaca.markets)")
    print()
    print("2. 🧪 Run a backtest to validate the strategy:")
    print("   python main.py --backtest")
    print()
    print("3. 📊 Test with paper trading:")
    print("   python main.py --paper")
    print()
    print("4. 📈 View portfolio status:")
    print("   python main.py --status")
    print()
    print("5. 📚 Explore the sample backtest notebook:")
    print("   jupyter notebook sample_backtest.ipynb")
    print()
    print("6. 🧪 Run unit tests:")
    print("   python -m pytest tests/")
    print()
    print("⚠️  IMPORTANT REMINDERS:")
    print("   • Always start with paper trading")
    print("   • Monitor the bot's performance regularly")
    print("   • Start with small amounts for live trading")
    print("   • Review logs in the logs/ directory")
    print()
    print("📖 For more information, see README.md")
    print("="*60)


def main():
    """Main setup function."""
    print("🚀 Long-Term Investment Bot Setup")
    print("="*40)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\nFound {len(missing)} missing dependencies.")
        install = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install == 'y':
            if not install_dependencies(missing):
                sys.exit(1)
        else:
            print("❌ Cannot proceed without required dependencies")
            sys.exit(1)
    
    # Create directories
    print("\n3. Creating directories...")
    if not create_directories():
        sys.exit(1)
    
    # Setup environment file
    print("\n4. Setting up environment file...")
    if not setup_environment_file():
        sys.exit(1)
    
    # Validate configuration
    print("\n5. Validating configuration...")
    if not validate_config():
        sys.exit(1)
    
    # Check code formatting tools
    print("\n6. Checking code formatting tools...")
    check_code_formatting()
    
    # Run basic tests
    print("\n7. Running basic import tests...")
    if not run_basic_tests():
        print("⚠️  Some modules failed to import. Check dependencies.")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
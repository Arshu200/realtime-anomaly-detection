#!/bin/bash

# Installation script for Real-Time Anomaly Detection System dependencies
# This script helps install all required dependencies for the system

set -e  # Exit on error

echo "🔧 Installing Real-Time Anomaly Detection System Dependencies"
echo "============================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    else
        DISTRO="unknown"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    DISTRO="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    DISTRO="windows"
else
    OS="unknown"
    DISTRO="unknown"
fi

print_info "Detected OS: $OS ($DISTRO)"

# Check Python version
echo
echo "1. Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_status "Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.8+ required, found Python $PYTHON_VERSION"
        exit 1
    fi
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_status "Python $PYTHON_VERSION found"
        PYTHON_CMD="python"
    else
        print_error "Python 3.8+ required, found Python $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
echo
echo "2. Checking pip..."
if command -v pip3 &> /dev/null; then
    print_status "pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    print_status "pip found"
    PIP_CMD="pip"
else
    print_error "pip not found. Installing pip..."
    if [ "$OS" = "linux" ]; then
        if [ "$DISTRO" = "debian" ]; then
            sudo apt-get update
            sudo apt-get install -y python3-pip
        elif [ "$DISTRO" = "redhat" ]; then
            sudo yum install -y python3-pip
        fi
    elif [ "$OS" = "macos" ]; then
        print_info "Please install pip using: curl https://bootstrap.pypa.io/get-pip.py | python3"
        exit 1
    fi
    PIP_CMD="pip3"
fi

# Install system dependencies
echo
echo "3. Installing system dependencies..."
if [ "$OS" = "linux" ]; then
    if [ "$DISTRO" = "debian" ]; then
        print_info "Installing build dependencies for Ubuntu/Debian..."
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            python3-dev \
            libssl-dev \
            libffi-dev \
            curl \
            wget \
            git
        print_status "System dependencies installed"
    elif [ "$DISTRO" = "redhat" ]; then
        print_info "Installing build dependencies for CentOS/RHEL..."
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            python3-devel \
            openssl-devel \
            libffi-devel \
            curl \
            wget \
            git
        print_status "System dependencies installed"
    fi
elif [ "$OS" = "macos" ]; then
    print_info "Installing Xcode command line tools..."
    if ! xcode-select -p &> /dev/null; then
        xcode-select --install
        print_warning "Please complete Xcode installation and run this script again"
        exit 1
    fi
    print_status "Xcode command line tools available"
fi

# Create virtual environment (optional but recommended)
echo
echo "4. Setting up Python virtual environment..."
read -p "Create virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    VENV_NAME="venv_anomaly_detection"
    
    if [ ! -d "$VENV_NAME" ]; then
        print_info "Creating virtual environment: $VENV_NAME"
        $PYTHON_CMD -m venv $VENV_NAME
    fi
    
    print_info "Activating virtual environment"
    source $VENV_NAME/bin/activate
    
    # Update pip in venv
    pip install --upgrade pip
    print_status "Virtual environment ready"
    
    echo
    print_warning "Remember to activate the virtual environment before using the system:"
    print_warning "source $VENV_NAME/bin/activate"
else
    print_warning "Installing packages globally (not recommended for production)"
fi

# Install Python packages
echo
echo "5. Installing Python packages..."
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    $PIP_CMD install -r requirements.txt
    print_status "Python packages installed"
else
    print_warning "requirements.txt not found, installing core packages..."
    $PIP_CMD install \
        pandas>=1.5.0 \
        numpy>=1.21.0 \
        matplotlib>=3.5.0 \
        seaborn>=0.11.0 \
        scikit-learn>=1.1.0 \
        prophet>=1.1.0 \
        requests>=2.28.0 \
        loguru>=0.6.0 \
        joblib>=1.2.0 \
        statsmodels>=0.13.0 \
        plotly>=5.10.0 \
        influxdb-client>=1.36.0 \
        pyyaml>=6.0
    print_status "Core packages installed"
fi

# Install optional development packages
echo
echo "6. Installing development packages (optional)..."
read -p "Install development packages (pytest, black, flake8, etc.)? [Y/n]: " install_dev
install_dev=${install_dev:-Y}

if [[ $install_dev =~ ^[Yy]$ ]]; then
    $PIP_CMD install \
        pytest>=7.0.0 \
        pytest-cov>=4.0.0 \
        black>=22.0.0 \
        flake8>=5.0.0 \
        mypy>=1.0.0 \
        isort>=5.0.0
    print_status "Development packages installed"
fi

# Test installation
echo
echo "7. Testing installation..."
$PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')

try:
    import pandas as pd
    import numpy as np
    import prophet
    import influxdb_client
    import matplotlib
    import seaborn
    import sklearn
    import loguru
    print('✓ All required packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Create necessary directories
echo
echo "8. Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p outputs
print_status "Directories created"

# Set up configuration
echo
echo "9. Setting up configuration..."
if [ ! -f "config/config.yaml" ]; then
    print_warning "Main configuration file not found"
    print_info "Please copy and customize config/config.yaml from the template"
else
    print_status "Configuration file found"
fi

# Test the application
echo
echo "10. Testing application..."
if [ -f "src/main.py" ]; then
    cd src
    if $PYTHON_CMD main.py --help &> /dev/null; then
        print_status "Application test passed"
    else
        print_warning "Application test failed - check dependencies"
    fi
    cd ..
else
    print_warning "main.py not found - make sure you're in the project root"
fi

# Installation summary
echo
echo "============================================================="
echo -e "${GREEN}✅ Installation completed successfully!${NC}"
echo
echo "Next steps:"
echo "1. Activate virtual environment (if created):"
echo "   source $VENV_NAME/bin/activate"
echo
echo "2. Install and configure InfluxDB:"
echo "   python scripts/setup_influxdb.py"
echo
echo "3. Install and configure Grafana:"
echo "   See docs/installation.md for instructions"
echo
echo "4. Train a model:"
echo "   python src/main.py --mode train --dataset dataset.csv"
echo
echo "5. Start monitoring:"
echo "   python src/main.py --mode monitor"
echo
echo "6. Access Grafana dashboard:"
echo "   http://localhost:3000"
echo
echo "For detailed instructions, see docs/installation.md"
echo "For troubleshooting, see docs/troubleshooting.md"
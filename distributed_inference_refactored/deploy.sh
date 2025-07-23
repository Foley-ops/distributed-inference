#!/bin/bash
# Quick deployment script for refactored distributed inference

echo "Distributed Inference Deployment Helper"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: Must run from distributed_inference_refactored directory"
    exit 1
fi

# Function to check dependencies
check_dependencies() {
    echo "Checking local dependencies..."
    
    if [ ! -d "core" ]; then
        echo "  ✗ Missing core module"
        echo "    Run: cp -r ../core ."
        return 1
    else
        echo "  ✓ core module found"
    fi
    
    if [ ! -d "metrics" ]; then
        echo "  ✗ Missing metrics module"
        echo "    Run: cp -r ../metrics ."
        return 1
    else
        echo "  ✓ metrics module found"
    fi
    
    if [ ! -d "profiling" ]; then
        echo "  ✗ Missing profiling module"
        echo "    Run: cp -r ../profiling ."
        return 1
    else
        echo "  ✓ profiling module found"
    fi
    
    if [ ! -d "pipelining" ]; then
        echo "  ✗ Missing pipelining module"
        echo "    Run: cp -r ../pipelining ."
        return 1
    else
        echo "  ✓ pipelining module found"
    fi
    
    return 0
}

# Function to test locally
test_local() {
    echo -e "\nRunning local tests..."
    python test_refactored.py
}

# Function to show usage examples
show_usage() {
    echo -e "\nUsage Examples:"
    echo "==============="
    echo ""
    echo "1. Quick local test (no Pis):"
    echo "   python main.py --rank 0 --world-size 1 --num-test-samples 16"
    echo ""
    echo "2. Deploy to Pis and run:"
    echo "   python automated_runner.py deploy"
    echo "   python automated_runner.py quick"
    echo ""
    echo "3. Run comparison tests:"
    echo "   python automated_runner.py compare"
    echo ""
    echo "4. Interactive mode:"
    echo "   python automated_runner.py"
}

# Main menu
while true; do
    echo -e "\nOptions:"
    echo "1. Check dependencies"
    echo "2. Copy missing dependencies"
    echo "3. Run local test"
    echo "4. Show usage examples"
    echo "5. Launch automated runner"
    echo "0. Exit"
    
    read -p "Select option: " choice
    
    case $choice in
        1)
            check_dependencies
            ;;
        2)
            echo "Copying dependencies from parent directory..."
            cp -r ../core ../metrics ../profiling ../pipelining . 2>/dev/null
            echo "Done!"
            check_dependencies
            ;;
        3)
            test_local
            ;;
        4)
            show_usage
            ;;
        5)
            python automated_runner.py
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
done
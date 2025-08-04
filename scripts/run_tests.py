#!/usr/bin/env python3
"""Test runner script for the LLM Query Retrieval System."""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run tests for the LLM Query Retrieval System")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (number of workers)")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--test", "-t", help="Run specific test function")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage if requested
    if args.coverage or args.all:
        cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
    
    # Determine which tests to run
    if args.file:
        cmd.append(args.file)
    elif args.test:
        cmd.extend(["-k", args.test])
    elif args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.e2e:
        cmd.extend(["-m", "e2e"])
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.all:
        # Run all tests in sequence
        test_suites = [
            (["python", "-m", "pytest", "-m", "unit", "--tb=short"], "Unit Tests"),
            (["python", "-m", "pytest", "-m", "integration", "--tb=short"], "Integration Tests"),
            (["python", "-m", "pytest", "-m", "e2e", "--tb=short"], "End-to-End Tests"),
            (["python", "-m", "pytest", "-m", "performance", "--tb=short"], "Performance Tests"),
        ]
        
        all_passed = True
        for test_cmd, description in test_suites:
            if not run_command(test_cmd, description):
                all_passed = False
        
        # Generate final coverage report
        if args.coverage:
            coverage_cmd = ["python", "-m", "pytest", "--cov=app", "--cov-report=html", "--cov-report=term"]
            run_command(coverage_cmd, "Coverage Report Generation")
        
        if all_passed:
            print(f"\n🎉 All test suites passed!")
            return 0
        else:
            print(f"\n❌ Some test suites failed!")
            return 1
    else:
        # Default: run unit and integration tests
        cmd.extend(["-m", "unit or integration"])
    
    # Run the tests
    description = "Test Suite"
    if args.unit:
        description = "Unit Tests"
    elif args.integration:
        description = "Integration Tests"
    elif args.e2e:
        description = "End-to-End Tests"
    elif args.performance:
        description = "Performance Tests"
    
    success = run_command(cmd, description)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
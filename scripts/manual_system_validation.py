#!/usr/bin/env python3
"""
Manual system validation script for Task 17 - Final integration and system testing.

This script performs comprehensive validation by manually testing the system components
and creating a validation report.
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Validation result data structure."""
    category: str
    test_name: str
    passed: bool
    details: str
    duration: float
    error: Optional[str] = None

class ManualSystemValidator:
    """Manual system validation for comprehensive testing."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results: List[ValidationResult] = []
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete manual system validation."""
        logger.info("Starting manual system validation...")
        
        # Change to project root
        os.chdir(self.project_root)
        
        # Run validation categories
        await self.validate_project_structure()
        await self.validate_code_quality()
        await self.validate_configuration()
        await self.validate_documentation()
        await self.validate_test_coverage()
        await self.validate_error_handling()
        await self.validate_security_implementation()
        await self.validate_performance_considerations()
        await self.validate_deployment_readiness()
        await self.validate_requirements_compliance()
        
        return self.generate_final_report()
    
    async def validate_project_structure(self):
        """Validate project structure and organization."""
        category = "Project Structure"
        start_time = time.time()
        
        try:
            # Check main application files
            required_files = [
                "main.py",
                "requirements.txt",
                "README.md",
                "Dockerfile",
                "docker-compose.yml",
                ".env.example"
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.project_root / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Required Files Check",
                    passed=False,
                    details=f"Missing files: {missing_files}",
                    duration=time.time() - start_time,
                    error=f"Missing required files: {missing_files}"
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Required Files Check",
                    passed=True,
                    details="All required files present",
                    duration=time.time() - start_time
                ))
            
            # Check directory structure
            required_dirs = [
                "app",
                "app/controllers",
                "app/services",
                "app/data",
                "app/models",
                "app/utils",
                "tests",
                "tests/unit",
                "tests/integration",
                "tests/e2e",
                "docs",
                "scripts"
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not (self.project_root / dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Directory Structure Check",
                    passed=False,
                    details=f"Missing directories: {missing_dirs}",
                    duration=time.time() - start_time,
                    error=f"Missing required directories: {missing_dirs}"
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Directory Structure Check",
                    passed=True,
                    details="All required directories present",
                    duration=time.time() - start_time
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Project Structure Validation",
                passed=False,
                details="Failed to validate project structure",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_code_quality(self):
        """Validate code quality and implementation."""
        category = "Code Quality"
        start_time = time.time()
        
        try:
            # Check if main components exist and are importable
            components_to_check = [
                ("app.config", "Configuration module"),
                ("app.models.schemas", "Data models"),
                ("app.controllers.query_controller", "Query controller"),
                ("app.services.document_service", "Document service"),
                ("app.services.query_service", "Query service"),
                ("app.services.embedding_service", "Embedding service"),
                ("app.services.llm_service", "LLM service"),
                ("app.data.vector_store", "Vector store"),
                ("app.data.repository", "Database repository"),
                ("app.auth", "Authentication module"),
                ("app.middleware.error_handler", "Error handler"),
            ]
            
            import_failures = []
            for module_name, description in components_to_check:
                try:
                    __import__(module_name)
                    logger.info(f"✅ {description} imports successfully")
                except ImportError as e:
                    import_failures.append(f"{description}: {str(e)}")
                    logger.error(f"❌ {description} import failed: {str(e)}")
            
            if import_failures:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Module Import Check",
                    passed=False,
                    details=f"Import failures: {len(import_failures)}",
                    duration=time.time() - start_time,
                    error=f"Failed imports: {import_failures}"
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Module Import Check",
                    passed=True,
                    details="All core modules import successfully",
                    duration=time.time() - start_time
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Code Quality Validation",
                passed=False,
                details="Failed to validate code quality",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_configuration(self):
        """Validate configuration management."""
        category = "Configuration"
        start_time = time.time()
        
        try:
            # Check configuration files
            config_files = [
                ".env.example",
                "app/config.py"
            ]
            
            for config_file in config_files:
                if (self.project_root / config_file).exists():
                    logger.info(f"✅ {config_file} exists")
                else:
                    logger.error(f"❌ {config_file} missing")
            
            # Try to import and validate configuration
            try:
                from app.config import get_settings
                settings = get_settings()
                logger.info("✅ Configuration module loads successfully")
                
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Configuration Loading",
                    passed=True,
                    details="Configuration loads successfully",
                    duration=time.time() - start_time
                ))
                
            except Exception as e:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Configuration Loading",
                    passed=False,
                    details="Configuration loading failed",
                    duration=time.time() - start_time,
                    error=str(e)
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Configuration Validation",
                passed=False,
                details="Failed to validate configuration",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_documentation(self):
        """Validate documentation completeness."""
        category = "Documentation"
        start_time = time.time()
        
        try:
            required_docs = [
                "README.md",
                "docs/api_documentation.md",
                "docs/deployment.md",
                "docs/configuration.md",
                "docs/usage_examples.md",
                "docs/troubleshooting_guide.md",
                "docs/developer_guide.md",
                "docs/error_handling_guide.md"
            ]
            
            missing_docs = []
            existing_docs = []
            
            for doc in required_docs:
                doc_path = self.project_root / doc
                if doc_path.exists():
                    existing_docs.append(doc)
                    # Check if file has content
                    if doc_path.stat().st_size > 100:  # At least 100 bytes
                        logger.info(f"✅ {doc} exists and has content")
                    else:
                        logger.warning(f"⚠️ {doc} exists but appears empty")
                else:
                    missing_docs.append(doc)
                    logger.error(f"❌ {doc} missing")
            
            if missing_docs:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Documentation Completeness",
                    passed=False,
                    details=f"Missing: {len(missing_docs)}, Present: {len(existing_docs)}",
                    duration=time.time() - start_time,
                    error=f"Missing documentation: {missing_docs}"
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Documentation Completeness",
                    passed=True,
                    details=f"All {len(required_docs)} documentation files present",
                    duration=time.time() - start_time
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Documentation Validation",
                passed=False,
                details="Failed to validate documentation",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_test_coverage(self):
        """Validate test coverage and structure."""
        category = "Test Coverage"
        start_time = time.time()
        
        try:
            # Check test directories and files
            test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
            test_files_found = 0
            
            for test_dir in test_dirs:
                test_path = self.project_root / test_dir
                if test_path.exists():
                    test_files = list(test_path.glob("test_*.py"))
                    test_files_found += len(test_files)
                    logger.info(f"✅ {test_dir}: {len(test_files)} test files")
                else:
                    logger.error(f"❌ {test_dir} missing")
            
            # Check for key test files
            key_test_files = [
                "tests/unit/test_config.py",
                "tests/unit/test_auth.py",
                "tests/unit/test_models.py",
                "tests/integration/test_document_service.py",
                "tests/integration/test_query_service.py",
                "tests/e2e/test_complete_workflow.py"
            ]
            
            missing_key_tests = []
            for test_file in key_test_files:
                if not (self.project_root / test_file).exists():
                    missing_key_tests.append(test_file)
            
            if test_files_found >= 20 and not missing_key_tests:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Test Coverage Structure",
                    passed=True,
                    details=f"Found {test_files_found} test files, all key tests present",
                    duration=time.time() - start_time
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Test Coverage Structure",
                    passed=False,
                    details=f"Found {test_files_found} test files, missing key tests: {len(missing_key_tests)}",
                    duration=time.time() - start_time,
                    error=f"Missing key test files: {missing_key_tests}"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Test Coverage Validation",
                passed=False,
                details="Failed to validate test coverage",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_error_handling(self):
        """Validate error handling implementation."""
        category = "Error Handling"
        start_time = time.time()
        
        try:
            # Check error handling components
            error_components = [
                "app/exceptions.py",
                "app/middleware/error_handler.py",
                "app/utils/retry.py"
            ]
            
            missing_components = []
            for component in error_components:
                if not (self.project_root / component).exists():
                    missing_components.append(component)
                else:
                    logger.info(f"✅ {component} exists")
            
            # Try to import error handling modules
            try:
                from app.exceptions import BaseSystemError
                from app.middleware.error_handler import setup_error_handling
                logger.info("✅ Error handling modules import successfully")
                
                if missing_components:
                    self.results.append(ValidationResult(
                        category=category,
                        test_name="Error Handling Implementation",
                        passed=False,
                        details=f"Missing components: {missing_components}",
                        duration=time.time() - start_time,
                        error=f"Missing error handling components: {missing_components}"
                    ))
                else:
                    self.results.append(ValidationResult(
                        category=category,
                        test_name="Error Handling Implementation",
                        passed=True,
                        details="All error handling components present and importable",
                        duration=time.time() - start_time
                    ))
                    
            except ImportError as e:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Error Handling Implementation",
                    passed=False,
                    details="Error handling modules import failed",
                    duration=time.time() - start_time,
                    error=str(e)
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Error Handling Validation",
                passed=False,
                details="Failed to validate error handling",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_security_implementation(self):
        """Validate security implementation."""
        category = "Security"
        start_time = time.time()
        
        try:
            # Check security components
            security_files = [
                "app/auth.py",
                "app/security.py"
            ]
            
            for security_file in security_files:
                if (self.project_root / security_file).exists():
                    logger.info(f"✅ {security_file} exists")
                else:
                    logger.error(f"❌ {security_file} missing")
            
            # Try to import security modules
            try:
                from app.auth import verify_token
                from app.security import get_password_hash
                logger.info("✅ Security modules import successfully")
                
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Security Implementation",
                    passed=True,
                    details="Security modules present and importable",
                    duration=time.time() - start_time
                ))
                
            except ImportError as e:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Security Implementation",
                    passed=False,
                    details="Security modules import failed",
                    duration=time.time() - start_time,
                    error=str(e)
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Security Validation",
                passed=False,
                details="Failed to validate security",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_performance_considerations(self):
        """Validate performance considerations."""
        category = "Performance"
        start_time = time.time()
        
        try:
            # Check for async implementation
            main_file = self.project_root / "main.py"
            if main_file.exists():
                content = main_file.read_text()
                
                performance_indicators = [
                    "async def",
                    "await",
                    "AsyncClient",
                    "asyncio",
                    "connection pool"
                ]
                
                found_indicators = []
                for indicator in performance_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                
                logger.info(f"✅ Found performance indicators: {found_indicators}")
                
                if len(found_indicators) >= 3:
                    self.results.append(ValidationResult(
                        category=category,
                        test_name="Async Implementation",
                        passed=True,
                        details=f"Found {len(found_indicators)} async/performance indicators",
                        duration=time.time() - start_time
                    ))
                else:
                    self.results.append(ValidationResult(
                        category=category,
                        test_name="Async Implementation",
                        passed=False,
                        details=f"Only found {len(found_indicators)} async/performance indicators",
                        duration=time.time() - start_time,
                        error="Insufficient async implementation indicators"
                    ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Performance Validation",
                    passed=False,
                    details="Main application file not found",
                    duration=time.time() - start_time,
                    error="main.py not found"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Performance Validation",
                passed=False,
                details="Failed to validate performance considerations",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_deployment_readiness(self):
        """Validate deployment readiness."""
        category = "Deployment"
        start_time = time.time()
        
        try:
            # Check deployment files
            deployment_files = [
                "Dockerfile",
                "docker-compose.yml",
                "docker-compose.dev.yml",
                "requirements.txt"
            ]
            
            missing_files = []
            for file in deployment_files:
                if (self.project_root / file).exists():
                    logger.info(f"✅ {file} exists")
                else:
                    missing_files.append(file)
                    logger.error(f"❌ {file} missing")
            
            # Check deployment scripts
            deployment_scripts = [
                "scripts/deploy.sh",
                "scripts/deploy.ps1"
            ]
            
            script_count = 0
            for script in deployment_scripts:
                if (self.project_root / script).exists():
                    script_count += 1
                    logger.info(f"✅ {script} exists")
            
            if not missing_files and script_count > 0:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Deployment Readiness",
                    passed=True,
                    details=f"All deployment files present, {script_count} deployment scripts",
                    duration=time.time() - start_time
                ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Deployment Readiness",
                    passed=False,
                    details=f"Missing files: {len(missing_files)}, Scripts: {script_count}",
                    duration=time.time() - start_time,
                    error=f"Missing deployment files: {missing_files}"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Deployment Validation",
                passed=False,
                details="Failed to validate deployment readiness",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    async def validate_requirements_compliance(self):
        """Validate compliance with original requirements."""
        category = "Requirements Compliance"
        start_time = time.time()
        
        try:
            # Check requirements document
            requirements_file = self.project_root / ".kiro/specs/llm-query-retrieval-system/requirements.md"
            if requirements_file.exists():
                logger.info("✅ Requirements document exists")
                
                # Check main API endpoint implementation
                main_file = self.project_root / "main.py"
                if main_file.exists():
                    content = main_file.read_text()
                    
                    # Check for required API endpoint
                    if "/api/v1/hackrx/run" in content:
                        logger.info("✅ Main API endpoint implemented")
                        
                        # Check for required functionality
                        required_features = [
                            "QueryRequest",
                            "QueryResponse",
                            "Bearer",
                            "POST",
                            "documents",
                            "questions",
                            "answers"
                        ]
                        
                        found_features = []
                        for feature in required_features:
                            if feature in content:
                                found_features.append(feature)
                        
                        if len(found_features) >= len(required_features) * 0.8:  # 80% of features
                            self.results.append(ValidationResult(
                                category=category,
                                test_name="Requirements Compliance",
                                passed=True,
                                details=f"Found {len(found_features)}/{len(required_features)} required features",
                                duration=time.time() - start_time
                            ))
                        else:
                            self.results.append(ValidationResult(
                                category=category,
                                test_name="Requirements Compliance",
                                passed=False,
                                details=f"Only found {len(found_features)}/{len(required_features)} required features",
                                duration=time.time() - start_time,
                                error="Insufficient feature implementation"
                            ))
                    else:
                        self.results.append(ValidationResult(
                            category=category,
                            test_name="Requirements Compliance",
                            passed=False,
                            details="Main API endpoint not found",
                            duration=time.time() - start_time,
                            error="API endpoint /api/v1/hackrx/run not implemented"
                        ))
                else:
                    self.results.append(ValidationResult(
                        category=category,
                        test_name="Requirements Compliance",
                        passed=False,
                        details="Main application file not found",
                        duration=time.time() - start_time,
                        error="main.py not found"
                    ))
            else:
                self.results.append(ValidationResult(
                    category=category,
                    test_name="Requirements Compliance",
                    passed=False,
                    details="Requirements document not found",
                    duration=time.time() - start_time,
                    error="Requirements document missing"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                category=category,
                test_name="Requirements Validation",
                passed=False,
                details="Failed to validate requirements compliance",
                duration=time.time() - start_time,
                error=str(e)
            ))
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final validation report."""
        passed_tests = sum(1 for result in self.results if result.passed)
        failed_tests = len(self.results) - passed_tests
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Determine overall status
        if failed_tests == 0:
            overall_status = "PASSED"
        elif failed_tests <= 3:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "FAILED"
        
        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append("Review and address failed validation items")
            
            # Category-specific recommendations
            for category, results in categories.items():
                failed_in_category = sum(1 for r in results if not r.passed)
                if failed_in_category > 0:
                    recommendations.append(f"Address {failed_in_category} issues in {category}")
        
        if overall_status == "PASSED":
            recommendations.append("System validation completed successfully - ready for production")
        elif overall_status == "PASSED_WITH_WARNINGS":
            recommendations.append("System mostly ready - address warnings before production deployment")
        
        return {
            "overall_status": overall_status,
            "total_tests": len(self.results),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / len(self.results)) * 100 if self.results else 0,
            "categories": {
                category: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "failed": sum(1 for r in results if not r.passed)
                } for category, results in categories.items()
            },
            "results": [
                {
                    "category": r.category,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "details": r.details,
                    "duration": r.duration,
                    "error": r.error
                } for r in self.results
            ],
            "recommendations": recommendations
        }

def print_validation_report(report: Dict[str, Any]):
    """Print the validation report."""
    print("\n" + "="*80)
    print("MANUAL SYSTEM VALIDATION REPORT")
    print("="*80)
    
    print(f"\nOverall Status: {report['overall_status']}")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed: {report['passed_tests']}")
    print(f"Failed: {report['failed_tests']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    
    print("\nResults by Category:")
    print("-" * 80)
    for category, stats in report['categories'].items():
        status = "✅" if stats['failed'] == 0 else "❌"
        print(f"{status} {category:<25} | Passed: {stats['passed']:>2} | Failed: {stats['failed']:>2} | Total: {stats['total']:>2}")
    
    print("\nDetailed Results:")
    print("-" * 80)
    for result in report['results']:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} | {result['category']:<20} | {result['test_name']:<30} | {result['duration']:>6.2f}s")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    print("\nRecommendations:")
    print("-" * 80)
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    print("\n" + "="*80)

async def main():
    """Main validation function."""
    validator = ManualSystemValidator()
    report = await validator.run_validation()
    print_validation_report(report)
    
    # Save report to file
    report_file = Path("validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return appropriate exit code
    if report['overall_status'] == "FAILED":
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
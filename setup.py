#!/usr/bin/env python3
"""
Setup script for Brein AI - Local AI System Installer
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BreinInstaller:
    """Installer for Brein AI system."""

    def __init__(self, install_dir: str = None):
        self.install_dir = Path(install_dir) if install_dir else Path.home() / "brein_ai"
        self.venv_dir = self.install_dir / "venv"
        self.config_file = self.install_dir / "config.json"

    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        logger.info("Checking system requirements...")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False

        # Check available disk space (need at least 2GB)
        try:
            stat = os.statvfs(str(self.install_dir.parent))
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if free_space_gb < 2:
                logger.error(f"Insufficient disk space. Need at least 2GB, have {free_space_gb:.1f}GB")
                return False
        except:
            logger.warning("Could not check disk space")

        logger.info("System requirements check passed")
        return True

    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment."""
        logger.info(f"Creating virtual environment at {self.venv_dir}")

        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)

            # Upgrade pip
            pip_exe = self.venv_dir / "bin" / "pip"  # Unix
            if not pip_exe.exists():
                pip_exe = self.venv_dir / "Scripts" / "pip.exe"  # Windows

            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
            logger.info("Virtual environment created successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        logger.info("Installing dependencies...")

        pip_exe = self.get_pip_executable()
        requirements_file = Path(__file__).parent / "requirements.txt"

        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False

        try:
            subprocess.run([
                str(pip_exe), "install", "-r", str(requirements_file)
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def setup_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Setting up directories...")

        directories = [
            self.install_dir / "memory",
            self.install_dir / "models",
            self.install_dir / "logs",
            self.install_dir / "quarantine",
            self.install_dir / "sync",
            self.install_dir / "test_results",
            self.install_dir / "performance_logs"
        ]

        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            logger.info("Directories created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def create_config_file(self) -> bool:
        """Create default configuration file."""
        logger.info("Creating configuration file...")

        config = {
            "version": "1.0.0",
            "database": {
                "path": str(self.install_dir / "memory" / "brein_memory.db")
            },
            "models": {
                "embedding_model": "all-MiniLM-L6-v2",
                "export_dir": str(self.install_dir / "models")
            },
            "web_fetcher": {
                "quarantine_dir": str(self.install_dir / "quarantine"),
                "trusted_domains": [
                    "wikipedia.org",
                    "github.com",
                    "stackoverflow.com",
                    "arxiv.org"
                ]
            },
            "sync": {
                "sync_dir": str(self.install_dir / "sync")
            },
            "logging": {
                "audit_logs": str(self.install_dir / "logs" / "audit"),
                "performance_logs": str(self.install_dir / "logs" / "performance")
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "auto_start": True
            },
            "security": {
                "web_access_default": False,
                "audit_enabled": True
            }
        }

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration file created at {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to create config file: {e}")
            return False

    def create_startup_script(self) -> bool:
        """Create startup script for easy launching."""
        logger.info("Creating startup script...")

        if sys.platform == "win32":
            script_content = f'''@echo off
echo Starting Brein AI...
cd /d "{self.install_dir}"
call "{self.venv_dir}\\Scripts\\activate.bat"
python backend\\main.py
pause
'''
            script_path = self.install_dir / "start_brein.bat"
        else:
            script_content = f'''#!/bin/bash
echo "Starting Brein AI..."
cd "{self.install_dir}"
source "{self.venv_dir}/bin/activate"
python backend/main.py
'''
            script_path = self.install_dir / "start_brein.sh"

        try:
            with open(script_path, 'w') as f:
                f.write(script_content)

            if sys.platform != "win32":
                script_path.chmod(0o755)

            logger.info(f"Startup script created at {script_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create startup script: {e}")
            return False

    def create_desktop_shortcut(self) -> bool:
        """Create desktop shortcut (optional)."""
        if sys.platform != "win32":
            return True  # Skip on non-Windows for now

        try:
            import winshell
            from win32com.client import Dispatch

            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "Brein AI.lnk")

            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = str(self.install_dir / "start_brein.bat")
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.IconLocation = sys.executable
            shortcut.save()

            logger.info(f"Desktop shortcut created at {shortcut_path}")
            return True

        except ImportError:
            logger.warning("pywin32 not available, skipping desktop shortcut")
            return True
        except Exception as e:
            logger.warning(f"Failed to create desktop shortcut: {e}")
            return True

    def get_pip_executable(self) -> Path:
        """Get the pip executable path for the virtual environment."""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"

    def install(self) -> bool:
        """Run the complete installation process."""
        logger.info("Starting Brein AI installation...")

        steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up directories", self.setup_directories),
            ("Creating configuration file", self.create_config_file),
            ("Creating startup script", self.create_startup_script),
            ("Creating desktop shortcut", self.create_desktop_shortcut)
        ]

        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Installation failed at step: {step_name}")
                return False

        logger.info("Installation completed successfully!")
        logger.info(f"Brein AI installed at: {self.install_dir}")
        logger.info(f"To start Brein AI, run: {self.install_dir / 'start_brein.bat' if sys.platform == 'win32' else self.install_dir / 'start_brein.sh'}")

        return True

def main():
    parser = argparse.ArgumentParser(description="Install Brein AI")
    parser.add_argument("--dir", help="Installation directory", default=None)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    installer = BreinInstaller(args.dir)

    print("Brein AI Installer")
    print("==================")
    print(f"Installation directory: {installer.install_dir}")
    print()

    if not args.yes:
        response = input("Continue with installation? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Installation cancelled.")
            return

    if installer.install():
        print("\n🎉 Brein AI installation completed successfully!")
        print(f"📁 Installation directory: {installer.install_dir}")
        print("🚀 To start Brein AI, run the startup script or use:")
        print(f"   cd {installer.install_dir}")
        print("   source venv/bin/activate  # On Linux/Mac"        print("   call venv\\Scripts\\activate.bat  # On Windows"        print("   python backend/main.py")
    else:
        print("\n❌ Installation failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
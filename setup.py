import os
import subprocess
import sys

def run_command(command, venv_python):
    """Executes a command using the virtual environment's Python."""
    try:
        # The most robust way to run pip is as a module of its Python interpreter.
        # This command will be formatted like: "path/to/python.exe" -m pip install ...
        full_command = f'"{venv_python}" {command}'
        print(f"--- Running: {full_command}")
        subprocess.check_call(full_command, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"--- ERROR: Command not found. Is Python/venv installed correctly?")
        return False

def main():
    """Main function to guide user through setup."""
    print("========================================")
    print(" Offline Oracle Interactive Setup")
    print("========================================")
    
    # --- Create Virtual Environment ---
    if not os.path.exists("venv"):
        print("\n[Step 1/4] Creating Python virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("--- Virtual environment created successfully.")
        except subprocess.CalledProcessError:
            print("--- FAILED to create virtual environment. Please check your Python installation.")
            sys.exit(1)
    else:
        print("\n[Step 1/4] Virtual environment already exists. Skipping.")

    # Determine paths to python in venv
    if sys.platform == "win32":
        venv_python = os.path.join("venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join("venv", "bin", "python")

    # --- Upgrade Pip ---
    print("\n[Step 2/4] Upgrading pip...")
    # Use the robust 'python -m pip' method
    if not run_command("-m pip install --upgrade pip", venv_python):
        print("--- FAILED to upgrade pip.")
        sys.exit(1)

    # --- Ask user for hardware setup ---
    print("\n[Step 3/4] Preparing to install dependencies...")
    while True:
        choice = input("--- Do you have an NVIDIA GPU with CUDA installed? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("\n--- Installing GPU-specific packages from requirements-gpu.txt...")
            if not run_command("-m pip install -r requirements-gpu.txt", venv_python):
                print("--- FAILED to install GPU packages. Please check your CUDA version and requirements file.")
                sys.exit(1)
            break
        elif choice in ['n', 'no']:
            print("\n--- Installing CPU-only packages from requirements-cpu.txt...")
            if not run_command("-m pip install -r requirements-cpu.txt", venv_python):
                 print("--- FAILED to install CPU packages.")
                 sys.exit(1)
            break
        else:
            print("--- Invalid input. Please enter 'y' or 'n'.")

    # --- Final Instructions ---
    print("\n[Step 4/4] Finalizing setup...")
    if not os.path.exists("config.ini"):
        try:
            import shutil
            shutil.copy("config.ini.template", "config.ini")
            print("--- 'config.ini.template' copied to 'config.ini'.")
        except Exception as e:
            print(f"--- Could not copy config template: {e}")
            
    print("\n========================================")
    print(" Setup Complete!")
    print("\nNext steps:")
    print(" 1. Edit 'config.ini' to set your ZIM paths and choose an LLM provider.")
    print(" 2. Run '2-Build_Oracle.bat' (or .sh) to build your knowledge base.")
    print(" 3. Run '3-Run_Oracle.bat' (or .sh) to start the server.")
    print("========================================")

if __name__ == "__main__":
    main()
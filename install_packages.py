import subprocess

def is_package_installed(package_name):
    """Check if a package is installed using pip."""
    installed_packages = subprocess.run(['pip', 'list'], stdout=subprocess.PIPE, text=True)
    return package_name in installed_packages.stdout

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.run(['pip', 'install', package_name], check=True)

def main():
    print("Checking and installing necessary Python packages for the Audio Classifier project...")

    packages = ["pygame", "librosa", "scikit-learn"]

    for package in packages:
        if is_package_installed(package):
            print(f"{package} is already installed.")
        else:
            install_package(package)

    print("All necessary packages are checked and installed.")

if __name__ == "__main__":
    main()

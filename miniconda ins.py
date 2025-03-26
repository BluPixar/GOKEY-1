import os
import subprocess
import requests
import time

def install_miniconda():
    upf = os.environ.get('USERPROFILE')
    if os.path.exists(r"C:\miniconda3") == False and os.path.exists(f"{upf}\\miniconda3") == False:
        # os.path.exists(r"C:\miniconda3") == False - checks if miniconda3 is not installed system wide for all users in C drive 
        installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        installer_name = "Miniconda3-latest-Windows-x86_64.exe"
    

    
        print("Downloading the latest Miniconda installer...")
        response = requests.get(installer_url, stream=True)
        with open(installer_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
            print("Download completed.")        
    
        print("Installing Miniconda...")   
        subprocess.run(["start", "/wait", "Miniconda3-latest-Windows-x86_64.exe", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=1", "/S"], shell=True)
        print("Miniconda Installed Successfully")
    else:
        print("Miniconda is already installed.")
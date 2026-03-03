import os
import subprocess
import sys

def main():
    print("="*60)
    print("🦜 Advanced Bird Species Recognition System 🦜")
    print("   Final Year Project - YOLO & EfficientNet Architecture")
    print("="*60)
    print("\nPlease select your compute hardware backend:")
    print("  [1] CPU (Default, highly stable for all laptops/desktops)")
    print("  [2] GPU (NVIDIA CUDA - Extremely fast, requires PyTorch CUDA drivers)")
    print()
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    device = "cpu"
    if choice == '2':
        device = "cuda"
        print("\n> [HARDWARE] GPU Selected. If your system lacks drivers, it will safely fallback to CPU.")
    else:
        print("\n> [HARDWARE] CPU Selected. Maximum stability guaranteed.")
        
    print("> [NETWORK] Booting the Streamlit Benchmarking Application on localhost...")
    print("  (A browser window should open automatically in a few seconds)\n")
    
    # Store the choice as an environment variable so app_advanced.py reads it
    os.environ["STREAMLIT_DEFAULT_DEVICE"] = device
    
    # Call streamlit to launch the app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_advanced.py"])
    except KeyboardInterrupt:
        print("\nApplication gracefully closed by user.")

if __name__ == "__main__":
    main()

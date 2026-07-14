import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    # Project ki root directory find karte hain
    root_dir = Path(__file__).resolve().parent

    # Sabhi paths ko define karte hain
    data_raw_dir = root_dir / "data" / "raw"
    data_processed_dir = root_dir / "data" / "processed"
    reports_dir = root_dir / "reports"
    scripts_dir = root_dir / "scripts"

    # build_dataset.py ka path check karte hain
    script_path = scripts_dir / "build_dataset.py"
    if not script_path.exists():
        # Agar scripts folder me nahi hai, to root me check karte hain
        script_path = root_dir / "build_dataset.py"

    if not script_path.exists():
        print(f"\n❌ [ERROR] 'build_dataset.py' nahi mila! "
              f"Please check karein ki ye script sahi folder me hai.")
        sys.exit(1)

    print("\n🧹 ========================================== 🧹")
    print("      PURANI SABHI FILES KO DELETE KIYA JAA RAHA HAI")
    print("==================================================")

    # 1. data/raw/ ki sabhi purani cache files aur sub-folders ko delete karna
    if data_raw_dir.exists():
        deleted_raw = 0
        # Pehle sabhi files ko delete karte hain (subfolders ke andar ki files ko bhi)
        for path in data_raw_dir.glob("**/*"):
            if path.is_file():
                path.unlink()
                print(f"🗑️ Deleted Raw Cache: raw/{path.relative_to(data_raw_dir)}")
                deleted_raw += 1
        
        # Ab khali subfolders (jaise github_api/) ko remove karte hain
        for path in sorted(data_raw_dir.glob("**/*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        
        if deleted_raw == 0:
            print("ℹ️ 'data/raw' pehle se hi khali hai.")
    else:
        print("ℹ️ 'data/raw' folder nahi mila, build script ise auto-create karegi.")

    # 2. data/processed/ ki sabhi purani files ko delete karna
    if data_processed_dir.exists():
        deleted_processed = 0
        for file in data_processed_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"🗑️ Deleted Processed: processed/{file.name}")
                deleted_processed += 1
        if deleted_processed == 0:
            print("ℹ️ 'data/processed' pehle se hi khali hai.")
    else:
        print("ℹ️ 'data/processed' folder nahi mila.")

    # 3. reports/ ki sabhi purani files ko delete karna
    if reports_dir.exists():
        deleted_reports = 0
        for file in reports_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"🗑️ Deleted Report: reports/{file.name}")
                deleted_reports += 1
        if deleted_reports == 0:
            print("ℹ️ 'reports' folder pehle se hi khali hai.")
    else:
        print("ℹ️ 'reports' folder nahi mila.")

    print("\n🚀 ========================================== 🚀")
    print("      NAYA DATASET GENERATE KIYA JAA RAHA HAI")
    print("==================================================")

    # Command taiyar karte hain build_dataset.py ko run karne ke liye
    cmd = [sys.executable, str(script_path)]

    # Agar user ne is script ke sath extra arguments pass kiye hain, to unhe bhi forward kar dete hain
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print(f"Running: {' '.join(cmd)}\n")

    try:
        # Build script ko run karte hain aur uska output live terminal par show karte hain
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\n✨ ========================================== ✨")
            print(" 🎉 CONGRATS! Purana saara data saaf karke naya dataset ready ho gaya hai! 🎉")
            print("==================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [ERROR] Dataset generation me dikkat aayi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
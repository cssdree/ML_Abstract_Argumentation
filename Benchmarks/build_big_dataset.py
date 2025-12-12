from pathlib import Path
import urllib.request
import subprocess
import zipfile
import shutil
import sys
import os

ICCMA2023 = "https://zenodo.org/record/8348039/files/iccma2023_benchmarks.zip?download=1"
ARCHIVE = "iccma2023_benchmarks.zip"
ROOT = Path("Benchmarks")

AF_TO_APX_SCRIPT = ROOT/'af_to_apx.py'
AF_TO_IAF_SCRIPT = ROOT/'af_to_iaf.py'

source_directory_temp = "iccma2023_benchmarks"
source_directory = "main"
converted_apx_directory = "ICCMA2023-apx"
output_directory = "ICCMA2023-inc"


def RunCommand(command, step_name):
    print(f"\n--- {step_name} ---")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print("SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Step '{step_name}' failed.")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)


def DownloadAndExtract():
    archive_path = ROOT/ARCHIVE
    if not archive_path.exists():
        print(f"Downloading {archive_path}...")
        try:
            urllib.request.urlretrieve(ICCMA2023, archive_path)
            print("Download successful")
        except Exception as e:
            print(f"Download failed : {e}")
            sys.exit(1)
    else:
        print(f"Archive {archive_path} already exists")

    print("Extracting archive...")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT)
        print("Extraction successful")
    except Exception as e:
        print(f"Extraction failed : {e}")
        sys.exit(1)

    source_path_final = ROOT/source_directory
    source_path_temp = ROOT/source_directory_temp
    main_folder_in_zip = source_path_temp/source_directory
    if main_folder_in_zip.exists():
        shutil.move(main_folder_in_zip, source_path_final)
        print(f"Moved {source_directory} to {source_path_final}")
    else:
        print(f"ERROR: Could not find {source_directory} inside extracted archive")
        sys.exit(1)
    if source_path_temp.exists():
        shutil.rmtree(source_path_temp)


def ConvertAfToApx():
    if not AF_TO_APX_SCRIPT.exists():
        print(f"ERROR: Conversion script not found at {AF_TO_APX_SCRIPT}")
        sys.exit(1)

    input_path = ROOT / source_directory
    output_path = ROOT / converted_apx_directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Starting conversion from {input_path}/*.af to {output_path}/*.apx...")
    
    file_count = 0
    for af_file in input_path.glob("*.af"):
        apx_filename = af_file.stem + ".apx"
        apx_filepath = output_path/apx_filename
        with open(apx_filepath, 'w') as outfile:
            subprocess.run(["python3", AF_TO_APX_SCRIPT, af_file], check=True, stdout=outfile, text=True)
        file_count += 1
    print(f"Converted {file_count} .af files to .apx")


def ConvertAfToIaf():
    if not AF_TO_IAF_SCRIPT.exists():
        print(f"ERROR: Generation script not found at {AF_TO_IAF_SCRIPT}")
        sys.exit(1)

    input_path = ROOT / converted_apx_directory
    output_path = ROOT / output_directory
    output_path.mkdir(parents=True, exist_ok=True)

    command = ["python3", AF_TO_IAF_SCRIPT, str(input_path), str(output_path)]
    RunCommand(command, "Generating IAF Instances")


def Cleanup():
    print("\n--- Cleaning up ---")
    if (ROOT/ARCHIVE).exists():
        os.remove(ROOT/ARCHIVE)
    if (ROOT/source_directory).exists():
        shutil.rmtree(ROOT/source_directory)
    if (ROOT/converted_apx_directory).exists():
        shutil.rmtree(ROOT/converted_apx_directory)
    print("Cleanup complete")


if __name__ == "__main__":
    DownloadAndExtract()
    ConvertAfToApx()
    ConvertAfToIaf()
    Cleanup()
    print("\n\nPROCESS COMPLETE")
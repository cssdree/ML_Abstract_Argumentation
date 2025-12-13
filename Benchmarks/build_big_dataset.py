from pathlib import Path
import urllib.request
import numpy as np
import subprocess
import zipfile
import shutil
import sys
import os

ICCMA2023 = "https://zenodo.org/record/8348039/files/iccma2023_benchmarks.zip?download=1"
ARCHIVE = "iccma2023_benchmarks.zip"

ROOT = Path.cwd()
AF_TO_APX_SCRIPT = Path('af_to_apx.py')
AF_TO_IAF_SCRIPT = Path('af_to_iaf.py')

source_directory_temp = "benchmarks"
source_directory = "main"
converted_apx_directory = "ICCMA2023-apx"
output_directory = "ICCMA2023-inc"

seed = 2023
np.random.seed(seed)
probs = [0, 0.05, 0.1, 0.15, 0.2]


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


def SelectingMainFolder():
    source_path_temp = ROOT/source_directory_temp
    source_path = ROOT/source_directory
    main_folder_in_zip = source_path_temp/source_directory

    if main_folder_in_zip.exists():
        shutil.move(main_folder_in_zip, source_path)
        print(f"Moved {main_folder_in_zip} to {source_path}")
    else:
        print(f"ERROR: Could not find {source_directory} inside extracted archive")
        sys.exit(1)


def ConvertAfToApx():
    if not AF_TO_APX_SCRIPT.exists():
        print(f"ERROR: Conversion script not found at {AF_TO_APX_SCRIPT}")
        sys.exit(1)

    source_path = ROOT/source_directory
    output_path = ROOT/converted_apx_directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting conversion of .af files to .apx...")
    count = 0
    for af_file in source_path.glob("*.af"):
        print(af_file)
        apx_filepath = output_path/(af_file.stem+".apx")
        with open(apx_filepath, 'w') as outfile:
            subprocess.run(["python3", AF_TO_APX_SCRIPT, af_file], check=True, stdout=outfile, text=True)
        count += 1
    print(f"Converted {count} .af files to .apx")


def CopyArgFiles():
    source_path = ROOT/source_directory
    output_path = ROOT/converted_apx_directory

    count = 0
    for arg_file in source_path.glob("*.arg"):
        shutil.copy(arg_file, output_path/arg_file.name)
        count += 1
    print(f"Copied {count} .arg files from {source_directory} to {converted_apx_directory}.")


def ConvertAfToIaf():
    source_path = ROOT/converted_apx_directory
    output_path = ROOT/output_directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting generation of IAF instances from AF...")
    af_count = 0
    iaf_count = 0
    for apx_file in source_path.glob("*.apx"):
        print(apx_file)
        af_count += 1
        arg_filepath = source_path/(apx_file.stem+".af.arg")
        q = arg_filepath.read_text().strip()
        query = f"arg({q})."
        lines = apx_file.read_text().splitlines()
        args = [line for line in lines if line.startswith("arg")]
        atts = [line for line in lines if line.startswith("att")]

        for p in probs:
            base_filename = apx_file.stem+"_"+(str(int(100*p)))
            if p == 0:
                out_filepath = output_path/(base_filename+"_inc.apx")
                with open(out_filepath, 'w') as out:
                    for arg in args:
                        out.write(arg+"\n")
                    for att in atts:
                        out.write(att+"\n")
                (out_filepath.with_suffix(".arg")).write_text(q)
                iaf_count += 1
                continue

            def_args = []
            def_atts = []
            inc_args = []
            inc_atts = []
            for arg in args:
                if np.random.uniform() < p and arg != query:
                    inc_args.append(arg)
                else:
                    def_args.append(arg)
            for att in atts:
                if np.random.uniform() < p:
                    inc_atts.append(att)
                else:
                    def_atts.append(att)
            Export(output_path, base_filename, q, "inc", def_args, def_atts, inc_args, inc_atts)
            Export(output_path, base_filename, q, "arg-inc", def_args, atts, inc_args, [])
            Export(output_path, base_filename, q, "att-inc", args, def_atts, [], inc_atts)
            iaf_count += 3
    print(f"Generated {iaf_count} IAF instances from {af_count} AF")


def Export(output_path, base_filename, q, inc_type, def_args, def_atts, inc_args, inc_atts):
    out_filepath = output_path/(base_filename+"_"+inc_type+".apx")
    with open(out_filepath, 'w') as out:
        for arg in def_args:
            out.write(arg+"\n")
        for arg in inc_args:
            out.write("?"+arg+"\n")
        for att in def_atts:
            out.write(att+"\n")
        for att in inc_atts:
            out.write("?"+att+"\n")
    (out_filepath.with_suffix(".arg")).write_text(q)


def Cleanup():
    print("Cleaning up")
    if (ROOT/ARCHIVE).exists():
        os.remove(ROOT/ARCHIVE)
    if (ROOT/source_directory_temp).exists():
        shutil.rmtree(ROOT/source_directory_temp)
    if (ROOT/source_directory).exists():
        shutil.rmtree(ROOT/source_directory)
    if (ROOT/converted_apx_directory).exists():
        shutil.rmtree(ROOT/converted_apx_directory)
    print("Cleanup complete")


if __name__ == "__main__":
    DownloadAndExtract()
    SelectingMainFolder()
    ConvertAfToApx()
    CopyArgFiles()
    ConvertAfToIaf()
    Cleanup()
    print(f"\n\nPROCESS COMPLETE: Big dataset located in '{output_directory}' directory")
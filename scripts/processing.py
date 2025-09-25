import base64
import os
import sys
import shutil
import subprocess
import uuid
from pathlib import Path

from docx import Document
from docxcompose.composer import Composer
import pandas as pd
import numpy as np

# Define project structure constants
SCRIPTS_DIR = Path(__file__).parent
BASE_DIR = SCRIPTS_DIR.parent
REPORTS_DIR = BASE_DIR / "generated_reports"

# Ensure the reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)

def run_command(command, working_dir):
    """
    Executes a command in a subprocess, capturing and logging output.
    Raises an exception with detailed error info if the command fails.
    """
    print(f"Running command: {' '.join(command)} in {working_dir}")
    try:
        # Using capture_output=True to get stdout/stderr
        # Using text=True to decode them as text
        result = subprocess.run(
            command,
            check=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        # Construct a detailed error message
        error_message = (
            f"Command failed with exit code {e.returncode}.\n"
            f"Command: {' '.join(e.cmd)}\n\n"
            f"--- STDOUT ---\n{e.stdout}\n\n"
            f"--- STDERR ---\n{e.stderr}"
        )
        # Raise a new exception with the formatted message
        raise Exception(error_message) from e

def run_analysis_pipeline(file_paths, experiment_name, options):
    """
    Main pipeline function to process a list of CDF files.
    
    Args:
        file_paths (list[pathlib.Path]): A list of Path objects for the input CDF files.
        experiment_name (str): The name for the experiment run.
        options (dict): A dictionary of processing options from the UI.
    """
    # Create a unique directory for this run's reports
    run_id = f"{experiment_name}_{uuid.uuid4().hex[:8]}"
    run_report_dir = REPORTS_DIR / run_id
    run_report_dir.mkdir(exist_ok=True)

    # The cdf13_run.py script expects to be run from the base directory
    # and for input files to be in a 'raw_cdf' subdirectory. We create this temporarily.
    temp_raw_dir = BASE_DIR / "raw_cdf"
    temp_raw_dir.mkdir(exist_ok=True)

    bases = []
    for file_path in file_paths:
        # The 'base' is the filename without the .cdf extension.
        base = file_path.stem
        bases.append(base)
        # Copy the file into the expected 'raw_cdf' directory
        shutil.copy(file_path, temp_raw_dir / file_path.name)

    # --- Build the command for cdf13_run.py ---
    py_executable = sys.executable or "python3"
    command = [py_executable, str(SCRIPTS_DIR / "cdf13_run.py")]
    
    # Add bases
    command.extend(bases)

    # Add options from the UI, filtering out empty ones
    for key, value in options.items():
        if value is not None and value != '':
            command.extend([f"--{key}", str(value)])

    # --- Run the analysis script ---
    run_command(command, working_dir=BASE_DIR)

    # --- Consolidate results from all runs for the UI ---
    individual_results = []
    largest_peak_areas = []

    for base, original_path in zip(bases, file_paths):
        # 1. Find and encode the plot image
        plot_path = temp_raw_dir / f"{base}_with_baseline.png"
        plot_image_src = None
        if plot_path.exists():
            encoded_image = base64.b64encode(plot_path.read_bytes()).decode('ascii')
            plot_image_src = f"data:image/png;base64,{encoded_image}"

        # 2. Read the peaks data table
        peaks_csv_path = temp_raw_dir / f"{base}_peaks.csv"
        table_data = []
        table_columns = []
        if peaks_csv_path.exists():
            df = pd.read_csv(peaks_csv_path)
            
            # For the summary, find the largest peak's area %
            if not df.empty and 'area_percent' in df.columns:
                largest_peak_areas.append(df['area_percent'].max())
            
            # Drop the 'measure_value' column as it's not for display
            if "measure_value" in df.columns:
                df = df.drop(columns=["measure_value"])
            table_data = df.to_dict('records')
            table_columns = [{"name": i, "id": i} for i in df.columns]

        individual_results.append({
            "filename": original_path.name,
            "plot_image_src": plot_image_src,
            "table_data": table_data,
            "table_columns": table_columns,
        })

    # --- Calculate Summary Statistics ---
    summary_stats = {}
    if largest_peak_areas:
        summary_stats = {
            "mean_area_percent": np.mean(largest_peak_areas),
            "std_area_percent": np.std(largest_peak_areas),
            "n_replicates": len(largest_peak_areas)
        }

    # 3. Find all generated DOCX reports, merge them, and provide a download URL.
    docx_download_url = None
    report_paths = [BASE_DIR / f"{base}_report.docx" for base in bases]
    existing_reports = [p for p in report_paths if p.exists()]

    if existing_reports:
        final_report_name = f"{experiment_name}_replicates_report.docx"
        final_report_path = run_report_dir / final_report_name

        if len(existing_reports) > 1:
            # Merge multiple reports into one
            master_document = Document(existing_reports[0])
            composer = Composer(master_document)
            for i in range(1, len(existing_reports)):
                composer.append(Document(existing_reports[i]))
            composer.save(final_report_path)
        else:
            # If only one report, just move it
            shutil.move(str(existing_reports[0]), final_report_path)

        # Clean up the individual report files from the base directory
        for report_file in existing_reports:
            if report_file.exists():
                report_file.unlink()

        docx_download_url = f"/download/docx/{run_id}/{final_report_name}"

    # Clean up the temporary raw_cdf directory
    shutil.rmtree(temp_raw_dir)

    return {
        "individual_results": individual_results,
        "summary_stats": summary_stats,
        "docx_download_url": docx_download_url,
    }
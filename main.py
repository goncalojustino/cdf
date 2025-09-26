import base64
import uuid
from pathlib import Path
from flask import send_from_directory

import dash
from dash import dcc, html
from dash import dash_table, Input, Output, State
from dash.exceptions import PreventUpdate

# --- Import your new script runner ---
from scripts.processing import run_analysis_pipeline

# --- App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Create a directory for temporary uploads
UPLOAD_DIR = Path(__file__).parent / "temp_uploads"
REPORTS_DIR = Path(__file__).parent / "generated_reports"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Read changelog for display ---
CHANGES_PATH = Path(__file__).parent / "CHANGES.txt"
try:
    with open(CHANGES_PATH, "r", encoding="utf-8") as f:
        changes_content = f.read()
except FileNotFoundError:
    changes_content = "Changelog file (CHANGES.txt) not found."

# --- Preset Definitions for UI Display ---
PRESETS = {
    "default": {"noise-mode":"pre","noise-window":"0.3","snr":"2","smooth":"8","min-width-sec":"1","measure":"area","reject-height":"0.01"},
    "lenient": {"noise-mode":"pre","noise-window":"0.5","snr":"1.5","smooth":"5","min-width-sec":"0.5","measure":"area","reject-height":"0"},
    "strict":  {"noise-mode":"pre","noise-window":"0.3","snr":"3","smooth":"16","min-width-sec":"2","measure":"area","reject-height":"0.02"},
}

def create_preset_table_data(presets_dict):
    """Transforms the PRESETS dictionary into a format for Dash DataTable."""
    # Find all unique parameter keys across all presets
    all_params = set()
    for params in presets_dict.values():
        all_params.update(params.keys())
    
    # Sort for consistent order
    sorted_params = sorted(list(all_params))
    
    table_data = []
    for param in sorted_params:
        row = {'Parameter': param}
        row['Default'] = presets_dict['default'].get(param, '-')
        row['Lenient'] = presets_dict['lenient'].get(param, '-')
        row['Strict'] = presets_dict['strict'].get(param, '-')
        table_data.append(row)
        
    return table_data

preset_table_data = create_preset_table_data(PRESETS)

# --- App Layout ---
# We wrap the main content in a new parent Div to control the background color of the whole page.
# The original content is placed inside a white "card" for better readability.
app.layout = html.Div(style={'backgroundColor': '#f0f0f0', 'minHeight': '100vh', 'padding': '2rem 0'}, children=[
    html.Div(style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'sans-serif', 'backgroundColor': 'white', 'padding': '2rem', 'borderRadius': '5px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'}, children=[

    # --- Original Content Starts Here ---
    html.H1("CDF Analysis Pipeline"),

    # --- Changelog Box (expanded by default) ---
    html.Details([
        html.Summary('View Recent Changes'),
        html.Pre(changes_content, style={'backgroundColor': '#fff3e0', 'border': '1px solid #ffe0b2', 'padding': '10px', 'marginTop': '10px', 'maxHeight': '200px', 'overflowY': 'auto', 'whiteSpace': 'pre-wrap', 'fontSize': '0.8em', 'borderRadius': '3px'})
    ], open=True, style={'margin': '20px 0'}),

    html.P("Upload one or more replicate .cdf files for an experiment to generate a combined report."),

    # --- Experiment Name ---
    html.Div([
        html.Label("Experiment Name:"),
        dcc.Input(id='experiment-name', type='text', placeholder="e.g., 'My First Experiment'", style={'width': '100%'}),
    ], style={'marginBottom': '20px'}),
    
    # --- File Upload ---
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Replicate .cdf Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '20px 0'
        },
        multiple=True
    ),

    # Div to show the name of the uploaded file
    html.Div(id='upload-status-div', style={'marginTop': '10px'}),

    html.Div([
        html.P("Or", style={'textAlign': 'center', 'margin': '15px 0 5px 0'}),
        html.Button("Test with Default File", id="test-button", n_clicks=0, style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'backgroundColor': '#f0f0f0', 'border': '1px solid #ccc'})
    ], style={'marginTop': '10px'}),
    
    # --- Options ---
    html.Div([
        html.H3("Processing Options"),
        html.Div([
            html.Label("Preset:"),
            dcc.Dropdown(
                id='option-preset',
                options=[
                    {'label': 'Default', 'value': 'default'},
                    {'label': 'Lenient', 'value': 'lenient'},
                    {'label': 'Strict', 'value': 'strict'},
                ],
                value='default'
            ),
        ], style={'marginBottom': '10px'}),
        
        html.Div([
            html.Label("SNR Threshold: (--snr)"),
            dcc.Input(id='option-snr', type='number', placeholder="e.g., 2 (overrides preset)", style={'width': '100%'}),
        ], style={'marginBottom': '10px'}),
        
        # --- Advanced Options ---
        html.Details([
            html.Summary('Advanced Options'),
            html.Div(style={'padding': '10px', 'marginTop': '10px', 'border': '1px solid #eee', 'borderRadius': '3px'}, children=[
                html.H5("Preset Default Values"),
                html.P("This table shows the default values for each preset. Any options you set below will override these values.", style={'fontSize': '0.9em', 'color': '#6c757d'}),
                dash_table.DataTable(
                    columns=[
                        {'name': 'Parameter', 'id': 'Parameter'},
                        {'name': 'Default', 'id': 'Default'},
                        {'name': 'Lenient', 'id': 'Lenient'},
                        {'name': 'Strict', 'id': 'Strict'},
                    ],
                    data=preset_table_data,
                    style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                    style_data_conditional=[{
                        'if': {'column_id': 'Parameter'},
                        'fontWeight': 'bold'
                    }],
                    style_table={'marginBottom': '20px'}
                ),

                html.Div([
                    html.Label("Smoothing Window (points): (--smooth)"),
                    dcc.Input(id='option-smooth', type='number', placeholder="e.g., 8", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Minimum Peak Width (seconds): (--min-width-sec)"),
                    dcc.Input(id='option-min-width-sec', type='number', placeholder="e.g., 1.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Noise Mode: (--noise-mode)"),
                    dcc.Dropdown(
                        id='option-noise-mode',
                        options=[
                            {'label': 'Pre-peak', 'value': 'pre'},
                            {'label': 'Start of run', 'value': 'start'},
                            {'label': 'Fixed value', 'value': 'fixed'},
                        ],
                        placeholder="Select noise estimation method"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Baseline Method: (--baseline-method)"),
                    dcc.Dropdown(
                        id='option-baseline-method',
                        options=[
                            {'label': 'Valley-to-valley', 'value': 'valley'},
                            {'label': 'Tangent skim', 'value': 'tangent'},
                        ],
                        placeholder="Select baseline drawing method"
                    ),
                ], style={'marginBottom': '10px'}),

                # --- New Controls Start Here ---
                html.H5("Peak Detection", style={'marginTop': '20px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                html.Div([
                    html.Label("Minimum Peak Prominence: (--min-prominence)"),
                    dcc.Input(id='option-min-prominence', type='number', placeholder="e.g., 0.01", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Noise Level (for 'fixed' mode): (--noise)"),
                    dcc.Input(id='option-noise', type='number', placeholder="e.g., 5000", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Noise Window (minutes): (--noise-window)"),
                    dcc.Input(id='option-noise-window', type='number', placeholder="e.g., 0.3", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),

                html.H5("Measurement & Rejection", style={'marginTop': '20px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                html.Div([
                    html.Label("Peak Measurement for Rejection: (--measure)"),
                    dcc.Dropdown(id='option-measure', options=[{'label': 'Area', 'value': 'area'}, {'label': 'Height', 'value': 'height'}, {'label': 'Sqrt(Height)', 'value': 'sqrt_height'}]),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Reject if Measurement Below: (--reject-height)"),
                    dcc.Input(id='option-reject-height', type='number', placeholder="e.g., 1000", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),

                html.H5("Overlaps & Baseline", style={'marginTop': '20px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                html.Div([
                    html.Label("Split Overlapping Peaks: (--split-overlaps)"),
                    dcc.Dropdown(id='option-split-overlaps', options=[{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}]),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Valley Depth Fraction: (--valley-depth-frac)"),
                    dcc.Input(id='option-valley-depth-frac', type='number', placeholder="e.g., 0.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Tangent Height Percent (for tangent baseline): (--tangent-height-pct)"),
                    dcc.Input(id='option-tangent-height-pct', type='number', placeholder="e.g., 10", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Clip Negative Values (before integration): (--clip-negative)"),
                    dcc.Dropdown(id='option-clip-negative', options=[{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}]),
                ], style={'marginBottom': '15px'}),

                html.H5("Blank Subtraction", style={'marginTop': '20px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                html.Div([
                    html.Label("Subtract Blank: (--subtract-blank)"),
                    dcc.Dropdown(id='option-subtract-blank', options=[{'label': 'On', 'value': 'on'}, {'label': 'Off', 'value': 'off'}]),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Blank Scale Factor: (--blank-scale)"),
                    dcc.Input(id='option-blank-scale', type='number', placeholder="e.g., 1.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),

                html.H5("ROI & Reporting", style={'marginTop': '20px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
                html.Div([
                    html.Label("Region of Interest (ROI) Start (min): (--roi-start)"),
                    dcc.Input(id='option-roi-start', type='number', placeholder="e.g., 0.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Region of Interest (ROI) End (min): (--roi-end)"),
                    dcc.Input(id='option-roi-end', type='number', placeholder="e.g., 32.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Integrator: (--integrator)"),
                    dcc.Dropdown(id='option-integrator', options=[{'label': 'Trapezoid', 'value': 'trapz'}, {'label': 'Simpson', 'value': 'simpson'}]),
                ], style={'marginBottom': '15px'}),
                html.Div([
                    html.Label("Denominator for % Area: (--percent-denominator)"),
                    dcc.Dropdown(id='option-percent-denominator', options=[{'label': 'All integrated peaks', 'value': 'all'}, {'label': 'ROI', 'value': 'roi'}, {'label': 'ID-only', 'value': 'idonly'}]),
                ], style={'marginBottom': '15px'}),
            ])
        ])
        
    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # --- Run Button ---
    html.Button("Run Analysis", id="run-button", n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': '18px', 'marginTop': '20px'}),
    
    # --- Output Area ---
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=html.Div(id='output-div', style={'marginTop': '20px'})
    ),
    
])])

def save_uploaded_file(name, content) -> Path:
    """Decodes and saves an uploaded file to a temporary directory."""
    # Sanitize the filename to remove spaces and other problematic characters
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

    data = content.encode("utf8").split(b";base64,")[1]
    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{safe_name}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as fp:
        fp.write(base64.decodebytes(data))
    return file_path

# --- Callback to provide feedback on file upload ---
@app.callback(
    Output('upload-status-div', 'children'),
    Input('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_upload_status(filenames):
    if filenames:
        file_list_items = [html.Li(f) for f in filenames]
        return html.Div([
            f'✅ {len(filenames)} file(s) selected:',
            html.Ul(file_list_items)
        ], style={'padding': '10px', 'border': '1px solid #c3e6cb', 'borderRadius': '5px', 'backgroundColor': '#d4edda', 'color': '#155724', 'marginTop': '10px'})
    return ""

# --- Main Callback to run the pipeline ---
@app.callback(
    Output('output-div', 'children'),
    [Input('run-button', 'n_clicks'),
     Input('test-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('experiment-name', 'value'),
     State('option-preset', 'value'),
     State('option-snr', 'value'),
     State('option-smooth', 'value'),
     State('option-min-width-sec', 'value'),
     State('option-noise-mode', 'value'),
     State('option-baseline-method', 'value'),
     State('option-min-prominence', 'value'),
     State('option-tangent-height-pct', 'value'),
     State('option-noise', 'value'),
     State('option-noise-window', 'value'),
     State('option-measure', 'value'),
     State('option-reject-height', 'value'),
     State('option-split-overlaps', 'value'),
     State('option-valley-depth-frac', 'value'),
     State('option-clip-negative', 'value'),
     State('option-subtract-blank', 'value'),
     State('option-blank-scale', 'value'),
     State('option-roi-start', 'value'),
     State('option-roi-end', 'value'),
     State('option-integrator', 'value'),
     State('option-percent-denominator', 'value'),
    ],
    prevent_initial_call=True
)
def run_analysis_callback(run_clicks, test_clicks, list_of_contents, list_of_filenames, experiment_name, preset, snr, 
                          smooth, min_width_sec, noise_mode, baseline_method,
                          min_prominence, tangent_height_pct,
                          noise, noise_window, measure, reject_height, split_overlaps, valley_depth_frac, clip_negative,
                          subtract_blank, blank_scale, roi_start, roi_end, integrator, percent_denominator) -> tuple[html.Div, str | None]:
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    if not experiment_name:
        # Generate a default experiment name if not provided
        experiment_name = f"Experiment_{uuid.uuid4().hex[:8]}"

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    uploaded_file_paths = []
    if triggered_id == 'test-button':
        # Use the default file for testing as a single replicate
        test_file = Path(__file__).parent / "test_data" / "kh18.cdf"
        if not test_file.exists():
            return html.Div([
                html.H4("Test File Not Found", style={'color': 'red'}),
                html.P(f"The default test file could not be found at the expected container location: {test_file}")
            ])
        uploaded_file_paths.append(test_file)
    elif triggered_id == 'run-button':
        # Use the user-uploaded file
        if not list_of_contents:
            return html.P("Please upload at least one file to run the analysis.", style={'color': 'red'})
        
        for name, content in zip(list_of_filenames, list_of_contents):
            uploaded_file_paths.append(save_uploaded_file(name, content))

    options = {
        'preset': preset, 
        'snr': snr,
        'smooth': smooth,
        'min-width-sec': min_width_sec,
        'noise-mode': noise_mode,
        'baseline-method': baseline_method,
        'min-prominence': min_prominence,
        'tangent-height-pct': tangent_height_pct,
        'noise': noise,
        'noise-window': noise_window,
        'measure': measure,
        'reject-height': reject_height,
        'split-overlaps': split_overlaps,
        'valley-depth-frac': valley_depth_frac,
        'clip-negative': clip_negative,
        'subtract-blank': subtract_blank,
        'blank-scale': blank_scale,
        'roi-start': roi_start,
        'roi-end': roi_end,
        'integrator': integrator,
        'percent-denominator': percent_denominator,
    }
    
    try:
        results = run_analysis_pipeline(uploaded_file_paths, experiment_name, options)
    except Exception as e:
        # The exception from processing.py now contains detailed info.
        # We will display it in a pre-formatted block for readability.
        error_details = str(e)
        return html.Div([
            html.H4("Pipeline Failed", style={'color': 'red'}),
            html.P("An error occurred while running the analysis scripts. Here are the details:"),
            html.Pre(
                error_details,
                style={
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'padding': '1rem',
                    'whiteSpace': 'pre-wrap',
                    'wordBreak': 'break-all'
                }
            )
        ])
        
    # --- Build the results display ---
    output_children = [html.H2("Analysis Complete", style={'color': 'green'})]

    # --- Display Individual Results for Each File ---
    for res in results.get("individual_results", []):
        output_children.append(html.Hr())
        output_children.append(html.H3(f"Results for: {res['filename']}", style={'marginTop': '30px'}))

        # Display the plot for this file
        if res.get("plot_image_src"):
            output_children.extend([
                html.H4("Chromatogram Plot"),
                html.Img(src=res["plot_image_src"], style={'maxWidth': '100%', 'border': '1px solid #ddd'})
            ])

        # Display the data table for this file
        if res.get("table_data"):
            output_children.extend([
                html.H4("Integration Results", style={'marginTop': '20px'}),
                dash_table.DataTable(
                    columns=res["table_columns"],
                    data=res["table_data"],
                    style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif'},
                    style_header={'fontWeight': 'bold'},
                    style_table={'overflowX': 'auto'}
                )
            ])

    # --- Display Summary Statistics Table ---
    summary = results.get("summary_stats")
    if summary:
        summary_data = [
            {'Metric': 'Number of Replicates', 'Value': f"{summary.get('n_replicates', 0)}"},
            {'Metric': 'Mean of Largest Peak (% Area)', 'Value': f"{summary.get('mean_area_percent', 0):.2f}"},
            {'Metric': 'Std. Dev. of Largest Peak (% Area)', 'Value': f"{summary.get('std_area_percent', 0):.2f}"},
        ]
        output_children.extend([
            html.Hr(),
            html.H3("Summary Statistics", style={'marginTop': '40px'}),
            dash_table.DataTable(
                columns=[
                    {"name": "Metric", "id": "Metric"},
                    {"name": "Value", "id": "Value"},
                ],
                data=summary_data,
                style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif', 'padding': '10px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_data_conditional=[{
                    'if': {'column_id': 'Metric'},
                    'fontWeight': 'bold'
                }]
            )
        ])

    # --- Display the NEW final summary table ---
    summary_table = results.get("summary_table")
    if summary_table and summary_table.get('data'):
        stats = summary_table['stats_row']
        # Create the final summary row for the table display
        summary_row_for_table = {'File': 'Summary'}
        # Format the pct_area column with mean and std dev
        summary_row_for_table['pct_area'] = f"{stats['mean']:.2f} (± {stats['std']:.2f})"

        table_data = summary_table['data'] + [summary_row_for_table]

        output_children.extend([
            html.Hr(),
            html.H3("Peak Summary Table", style={'marginTop': '40px'}),
            html.P("The table below shows the peak with the largest '% area' from each replicate."),
            dash_table.DataTable(
                columns=summary_table['columns'],
                data=table_data,
                style_cell={'textAlign': 'left', 'fontFamily': 'sans-serif'},
                style_header={'fontWeight': 'bold'},
                style_table={'overflowX': 'auto', 'marginTop': '10px'},
                style_data_conditional=[
                    {
                        'if': {'row_index': len(table_data) - 1}, # Target the last 'Summary' row
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    }
                ]
            )
        ])


    # Add the download button for the DOCX report
    if results.get("docx_download_url"):
        output_children.extend([
            html.P(
                "Note: The full report contains individual results for all uploaded files.",
                style={'fontSize': '0.9em', 'color': '#6c757d', 'textAlign': 'center', 'marginTop': '20px', 'fontStyle': 'italic'}
            ),
            html.A(
                html.Button("Download Full Report (DOCX)", style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'marginTop': '30px'}),
                href=results["docx_download_url"],
                download=True
            )
        ])

    return html.Div(output_children)


# --- Flask route to serve the generated DOCX files ---
@server.route('/download/docx/<run_id>/<filename>')
def download_docx_file(run_id, filename):
    directory = REPORTS_DIR / run_id
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)

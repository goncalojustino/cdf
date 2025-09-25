import base64
import uuid
from pathlib import Path
from flask import send_from_directory

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
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
                placeholder="Select a preset (overridden by specific options below)"
            ),
        ], style={'marginBottom': '10px'}),
        
        html.Div([
            html.Label("SNR Threshold:"),
            dcc.Input(id='option-snr', type='number', placeholder="e.g., 2 (overrides preset)", style={'width': '100%'}),
        ], style={'marginBottom': '10px'}),
        
        # --- Advanced Options ---
        html.Details([
            html.Summary('Advanced Options'),
            html.Div(style={'padding': '10px', 'marginTop': '10px', 'border': '1px solid #eee', 'borderRadius': '3px'}, children=[
                html.Div([
                    html.Label("Smoothing Window (points):"),
                    dcc.Input(id='option-smooth', type='number', placeholder="e.g., 8", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Minimum Peak Width (seconds):"),
                    dcc.Input(id='option-min-width-sec', type='number', placeholder="e.g., 1.0", style={'width': '100%'}),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Noise Mode:"),
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
                    html.Label("Baseline Method:"),
                    dcc.Dropdown(
                        id='option-baseline-method',
                        options=[
                            {'label': 'Valley-to-valley', 'value': 'valley'},
                            {'label': 'Tangent skim', 'value': 'tangent'},
                        ],
                        placeholder="Select baseline drawing method"
                    ),
                ], style={'marginBottom': '10px'}),
                # You can continue adding more controls here following the same pattern.
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
    
    # Hidden store to trigger the clientside callback for opening a new tab
    dcc.Store(id='new-report-url-store')
])])

def save_uploaded_file(name, content) -> Path:
    """Decodes and saves an uploaded file to a temporary directory."""
    data = content.encode("utf8").split(b";base64,")[1]
    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{name}"
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
    Output('new-report-url-store', 'data'),
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
    ],
    prevent_initial_call=True
)
def run_analysis_callback(run_clicks, test_clicks, list_of_contents, list_of_filenames, experiment_name, preset, snr, 
                          smooth, min_width_sec, noise_mode, baseline_method):
    
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
        test_file = Path('/opt/cdf_gunicorn/oldfiles/kh18.cdf')
        if not test_file.exists():
            return html.Div([
                html.H4("Test File Not Found", style={'color': 'red'}),
                html.P(f"The default test file could not be found at the expected location: {test_file}")
            ]), None
        uploaded_file_paths.append(test_file)
    elif triggered_id == 'run-button':
        # Use the user-uploaded file
        if not list_of_contents:
            return html.P("Please upload at least one file to run the analysis.", style={'color': 'red'}), None
        
        for name, content in zip(list_of_filenames, list_of_contents):
            uploaded_file_paths.append(save_uploaded_file(name, content))

    options = {
        'preset': preset, 
        'snr': snr,
        'smooth': smooth,
        'min-width-sec': min_width_sec,
        'noise-mode': noise_mode,
        'baseline-method': baseline_method,
    }
    
    try:
        report_paths = run_analysis_pipeline(uploaded_file_paths, experiment_name, options)
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
        ]), None
        
    output_elements = html.Div([
        html.H4("Analysis Complete!", style={'color': 'green'}),
        html.P("Your report is opening in a new tab."),
        html.A("Click here if it doesn't open automatically.", 
               href=report_paths['html_report_url'], 
               target="_blank")
    ])
    
    # This will trigger the clientside callback to open the new tab
    return output_elements, report_paths['html_report_url']
    
# --- Clientside callback to open a new tab ---
app.clientside_callback(
    """
    function(url) {
        if (url) {
            window.open(url);
        }
        return '';
    }
    """,
    Output('new-report-url-store', 'data', allow_duplicate=True),
    Input('new-report-url-store', 'data'),
    prevent_initial_call=True
)

# --- Flask route to serve the generated DOCX files ---
@server.route('/download/docx/<run_id>/<filename>')
def download_docx_file(run_id, filename):
    directory = REPORTS_DIR / run_id
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)

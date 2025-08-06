import os
import subprocess
import sys
import socket
import time
import markdown
from users_db import create_user, authenticate_user, get_non_admin_users
from flask import Flask, render_template, jsonify, request, redirect, session
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
# --- Plotly imports for dashboard visualization ---
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Place admin-login route after app initialization
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    error = None
    issue_points_graph_html = None  # New graph for V, I, T with issue highlights
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        if user and user.get('is_admin', False):
            session['username'] = user['username']
            session['is_admin'] = True
            return redirect('/admin')
        else:
            error = 'Invalid admin credentials.'
    return render_template('admin_login.html', error=error)

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# --- Dashboard route for time-series analytics and Plotly visualization ---

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    import io
    import base64
    from statsmodels.tsa.seasonal import seasonal_decompose
    plot_html = None
    summary_stats = {}
    alerts = []
    latest_data = None
    metrics = ['voltage', 'current', 'temperature']
    selected_metrics = request.form.getlist('metrics') if request.method == 'POST' else metrics
    export_type = request.form.get('export') if request.method == 'POST' else None
    df = None
    separate_graphs = {}  # Ensure always defined
    insights = {}  # Ensure always defined
    alert_graph_html = None  # Ensure always defined
    issue_points_graph_html = None  # Ensure always defined
    from users_db import log_user_activity
    log_user_activity(session.get('username'), 'view_dashboard')
    if request.method == 'POST':
        # Handle clear button
        if request.form.get('clear') == '1':
            log_user_activity(session.get('username'), 'clear_dashboard')
            return render_template('dashboard.html', plot_html=None, summary_stats=None, alerts=None, latest_data=None, metrics=metrics, selected_metrics=metrics)
        file = request.files.get('csv_file')
        if file and file.filename.endswith('.csv'):
            log_user_activity(session.get('username'), 'upload_file', {'filename': file.filename})
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            # Time-series analysis: rolling mean (window=10)
            for m in metrics:
                df[f'{m}_ma'] = df[m].rolling(window=10, min_periods=1).mean()

            # Add 7-day and 30-day moving averages for each metric
            for m in metrics:
                df[f'{m}_ma7'] = df[m].rolling(window=7, min_periods=1).mean()
                df[f'{m}_ma30'] = df[m].rolling(window=30, min_periods=1).mean()


            # Alerts (reuse existing logic)
            alerts += df.apply(detect_overheating_row, axis=1).explode().dropna().tolist()
            alerts += df.apply(detect_voltage_drop_row, axis=1).explode().dropna().tolist()
            alerts += [a for i, row in df.iterrows() for a in detect_current_spike(df.iloc[i-1] if i > 0 else None, row)]
            alerts += df.apply(estimate_efficiency_row, axis=1).explode().dropna().tolist()
            avg_p, peak_p = compute_power_stats(df)
            alerts.append(f"Average power: {avg_p:.1f} W, Peak power: {peak_p:.1f} W")

            # Plotly line chart for all metrics and their moving averages (solid lines only)
            fig = go.Figure()
            for m in metrics:
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[m], mode='lines', name=f'{m.title()}'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f'{m}_ma7'], mode='lines', name=f'{m.title()} 7-day MA'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f'{m}_ma30'], mode='lines', name=f'{m.title()} 30-day MA'))
            fig.update_layout(title='Time-Series Analysis', xaxis_title='Timestamp', yaxis_title='Value', template='plotly_white')
            plotly_config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['toggleFullscreen']
            }
            plot_html = pio.to_html(fig, full_html=False, config=plotly_config)

            # Alerts graph: show alerts as red circle points on timeline
            alert_graph_html = None
            issue_points_graph_html = None
            if alerts:
                import re
                alert_times = []
                alert_texts = []
                for alert in alerts:
                    match = re.match(r'\[(.*?)\]', alert)
                    if match:
                        alert_times.append(match.group(1))
                        alert_texts.append(alert)
                if alert_times:
                    alert_fig = go.Figure()
                    alert_fig.add_trace(go.Scatter(
                        x=alert_times,
                        y=[1]*len(alert_times),
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='circle'),
                        text=alert_texts,
                        name='Alerts',
                        hoverinfo='text'
                    ))
                    alert_fig.update_layout(
                        title='Alerts Timeline',
                        xaxis_title='Timestamp',
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        template='plotly_white',
                        height=250
                    )
                    alert_graph_html = pio.to_html(alert_fig, full_html=False, config=plotly_config)

                    # --- New: Plot V, I, T with issue points highlighted ---
                    # For each metric, plot the time series and overlay red circles at alert timestamps
                    metrics_to_plot = ['voltage', 'current', 'temperature']
                    issue_fig = go.Figure()
                    colors = {'voltage': 'blue', 'current': 'green', 'temperature': 'orange'}
                    for m in metrics_to_plot:
                        issue_fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df[m],
                            mode='lines',
                            name=m.title(),
                            line=dict(color=colors[m])
                        ))
                    # Find issue points for each metric
                    for m in metrics_to_plot:
                        # Find alert timestamps for this metric
                        m_alert_times = []
                        m_alert_values = []
                        for alert in alerts:
                            match = re.match(r'\[(.*?)\]', alert)
                            if match and m in alert.lower():
                                ts = match.group(1)
                                # Find value at this timestamp
                                val = None
                                try:
                                    val = df.loc[df['timestamp'].astype(str) == ts, m].values[0]
                                except Exception:
                                    continue
                                m_alert_times.append(ts)
                                m_alert_values.append(val)
                        if m_alert_times:
                            issue_fig.add_trace(go.Scatter(
                                x=m_alert_times,
                                y=m_alert_values,
                                mode='markers',
                                marker=dict(color='red', size=7, symbol='diamond'),
                                name=f'{m.title()} Issue',
                                text=[f'{m.title()} Issue' for _ in m_alert_times],
                                hoverinfo='text'
                            ))
                    issue_fig.update_layout(
                        title='V, I, T Data with Issue Points Highlighted',
                        xaxis_title='Timestamp',
                        yaxis_title='Value',
                        template='plotly_white',
                        height=400
                    )
                    issue_points_graph_html = pio.to_html(issue_fig, full_html=False, config=plotly_config)
                    print('[DEBUG] issue_points_graph_html:', issue_points_graph_html[:200])

            # Individual graphs for each metric: raw, 7-day MA, 30-day MA (solid lines only)
            separate_graphs = {}
            for m in metrics:
                fig_sep = go.Figure()
                # Raw data
                fig_sep.add_trace(go.Scatter(x=df['timestamp'], y=df[m], mode='lines', name=f'{m.title()}'))
                # Decomposition: trend, seasonality, residuals
                try:
                    result = seasonal_decompose(df[m], model='additive', period=10, extrapolate_trend='freq')
                    fig_sep.add_trace(go.Scatter(x=df['timestamp'], y=result.trend, mode='lines', name=f'{m.title()} Trend'))
                    fig_sep.add_trace(go.Scatter(x=df['timestamp'], y=result.seasonal, mode='lines', name=f'{m.title()} Seasonality'))
                    fig_sep.add_trace(go.Scatter(x=df['timestamp'], y=result.resid, mode='lines', name=f'{m.title()} Residuals'))
                except Exception:
                    pass
                fig_sep.update_layout(title=f'{m.title()} Decomposition', xaxis_title='Timestamp', yaxis_title=m.title(), template='plotly_white')
                separate_graphs[m] = pio.to_html(fig_sep, full_html=False, config=plotly_config)

            # Advanced analytics: trend, seasonality, forecast
            decomposition_results = {}
            for m in selected_metrics:
                try:
                    result = seasonal_decompose(df[m], model='additive', period=10, extrapolate_trend='freq')
                    decomposition_results[m] = {
                        'trend': result.trend.tolist(),
                        'seasonal': result.seasonal.tolist(),
                        'resid': result.resid.tolist()
                    }
                except Exception:
                    decomposition_results[m] = None

            # Simple forecast: next value = last moving average
            forecast = {m: df[f'{m}_ma'].iloc[-1] if f'{m}_ma' in df else None for m in selected_metrics}

            # Dynamic insights for each metric
            insights = {}
            for m in metrics:
                vals = df[m].dropna()
                ma7 = df[f'{m}_ma7'].dropna()
                ma30 = df[f'{m}_ma30'].dropna()
                latest = vals.iloc[-1] if not vals.empty else None
                mean = vals.mean() if not vals.empty else None
                maxv = vals.max() if not vals.empty else None
                minv = vals.min() if not vals.empty else None
                trend = 'increasing' if len(ma30) > 1 and ma30.iloc[-1] > ma30.iloc[0] else 'decreasing' if len(ma30) > 1 and ma30.iloc[-1] < ma30.iloc[0] else 'stable'
                insight = f"<b>{m.title()}:</b> "
                if latest is not None:
                    insight += f"Latest value is <b>{latest:.2f}</b>. "
                if mean is not None:
                    insight += f"Average is <b>{mean:.2f}</b>. "
                if maxv is not None and minv is not None:
                    insight += f"Range: <b>{minv:.2f}</b> to <b>{maxv:.2f}</b>. "
                insight += f"Trend over time is <b>{trend}</b>. "
                if trend == 'increasing' and latest > mean:
                    insight += "Monitor for possible spikes. "
                elif trend == 'decreasing' and latest < mean:
                    insight += "Possible improvement or drop detected. "
                insights[m] = insight

            # Summary stats (always show for voltage, current, temperature)
            summary_stats = {}
            for m in ['voltage', 'current', 'temperature']:
                try:
                    summary_stats[f'{m}_mean'] = float(df[m].mean())
                    summary_stats[f'{m}_median'] = float(df[m].median())
                    summary_stats[f'{m}_min'] = float(df[m].min())
                    summary_stats[f'{m}_max'] = float(df[m].max())
                    summary_stats[f'{m}_std'] = float(df[m].std())
                    summary_stats[f'{m}_count'] = int(df[m].count())
                except Exception:
                    summary_stats[f'{m}_mean'] = None
                    summary_stats[f'{m}_median'] = None
                    summary_stats[f'{m}_min'] = None
                    summary_stats[f'{m}_max'] = None
                    summary_stats[f'{m}_std'] = None
                    summary_stats[f'{m}_count'] = None
            summary_stats['forecast'] = forecast

            # Latest data
            latest_data = df.iloc[-1].to_dict()
            latest_data['energy'] = df['energy'].sum() if 'energy' in df else 0

            # Export logic
            if export_type and df is not None:
                if export_type == 'csv':
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    return (output.getvalue(), 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=analytics.csv'})
                elif export_type == 'excel':
                    output = io.BytesIO()
                    df.to_excel(output, index=False)
                    output.seek(0)
                    return (output.read(), 200, {'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'Content-Disposition': 'attachment; filename=analytics.xlsx'})
                elif export_type == 'pdf':
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_pdf import PdfPages
                    pdf_output = io.BytesIO()
                    with PdfPages(pdf_output) as pdf:
                        for m in selected_metrics:
                            plt.figure(figsize=(8,4))
                            plt.plot(df['timestamp'], df[m], label=m)
                            plt.plot(df['timestamp'], df[f'{m}_ma'], label=f'{m}_ma')
                            plt.title(f'{m.title()} Time-Series')
                            plt.legend()
                            pdf.savefig()
                            plt.close()
                    pdf_output.seek(0)
                    return (pdf_output.read(), 200, {'Content-Type': 'application/pdf', 'Content-Disposition': 'attachment; filename=analytics.pdf'})

    return render_template('dashboard.html', plot_html=plot_html, alert_graph_html=alert_graph_html, issue_points_graph_html=issue_points_graph_html, summary_stats=summary_stats, alerts=alerts, latest_data=latest_data, metrics=metrics, separate_graphs=separate_graphs, insights=insights)

# --- Global flag to track app restart for context reset ---
app_restart_flag = {'cleared': False}

@app.before_request
def clear_context_on_restart():
    # Only clear once per app restart
    if not app_restart_flag['cleared']:
        session.pop('latest_alerts', None)
        session.pop('latest_data', None)
        app_restart_flag['cleared'] = True

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        if user:
            session['username'] = user['username']
            session['is_admin'] = user.get('is_admin', False)
            from users_db import log_user_activity
            log_user_activity(username, 'login')
            if session['is_admin']:
                return redirect('/admin')
            else:
                return redirect('/')
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        is_admin = 'is_admin' in request.form
        success, msg = create_user(username, password, is_admin)
        if success:
            session['username'] = username
            session['is_admin'] = is_admin
            if is_admin:
                return redirect('/admin')
            else:
                return redirect('/')
        else:
            error = msg
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    from users_db import log_user_activity
    log_user_activity(session.get('username'), 'logout')
    session.clear()
    return redirect('/login')

@app.route('/admin')
@login_required
def admin():
    if not session.get('is_admin'):
        return redirect('/')
    from users_db import get_all_users, get_all_user_activities
    users = get_all_users()
    activities = get_all_user_activities()
    return render_template('admin.html', users=users, activities=activities)
# Diagnostic functions
TEMP_THRESHOLD = 70.0
VOLTAGE_MIN = 20.0
CURRENT_SPIKE_DELTA = 2.0
K_T = 0.1
EFFICIENCY_ESTIMATE = 0.9

def detect_overheating_row(row):
    alerts = []
    if row['temperature'] > TEMP_THRESHOLD:
        alerts.append(
            f"[{row['timestamp']}] Overheating: temperature {row['temperature']}\u00b0C exceeds {TEMP_THRESHOLD}\u00b0C"
        )
    return alerts

def detect_voltage_drop_row(row):
    alerts = []
    if row['voltage'] < VOLTAGE_MIN:
        alerts.append(
            f"[{row['timestamp']}] Voltage drop: V={row['voltage']}V below {VOLTAGE_MIN}V"
        )
    return alerts

def detect_current_spike(prev_row, row):
    alerts = []
    if prev_row is not None:
        delta = abs(row['current'] - prev_row['current'])
        if delta > CURRENT_SPIKE_DELTA:
            alerts.append(
                f"[{row['timestamp']}] Current spike: ΔI={delta:.2f}A (I={row['current']:.2f}A)"
            )
    return alerts

def estimate_efficiency_row(row):
    alerts = []
    torque = row.get('torque', np.nan)
    if not np.isnan(torque):
        expected_current = torque / (K_T * EFFICIENCY_ESTIMATE)
        efficiency = min(expected_current / row['current'], 1.0)
        if efficiency < 0.9:
            alerts.append(
                f"[{row['timestamp']}] Efficiency low: {efficiency*100:.1f}%"
            )
    return alerts

def compute_power_stats(df):
    df['power'] = df['voltage'] * df['current']
    return df['power'].mean(), df['power'].max()

@app.route('/', methods=['GET', 'POST'])
@login_required
def diagnostics():
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            # Load and preprocess the CSV
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df['power'] = df['voltage'] * df['current']

            # Generate raw alerts (issues only, no solutions)
            alerts = []
            alerts += df.apply(detect_overheating_row, axis=1).explode().dropna().tolist()
            alerts += df.apply(detect_voltage_drop_row, axis=1).explode().dropna().tolist()
            alerts += [a for i, row in df.iterrows() for a in detect_current_spike(df.iloc[i-1] if i > 0 else None, row)]
            alerts += df.apply(estimate_efficiency_row, axis=1).explode().dropna().tolist()
            avg_p, peak_p = compute_power_stats(df)
            alerts.append(f"Average power: {avg_p:.1f} W, Peak power: {peak_p:.1f} W")

            # Store alerts and data in session for use in recommender and dashboard
            session['latest_alerts'] = alerts
            session['latest_data'] = latest_data = df.iloc[-1].to_dict()
            session['latest_data']['energy'] = df['energy'].sum() if 'energy' in df else 0

            return render_template('index.html', initial_data=[latest_data], alerts=alerts, summary=[], suggestions=[])
    # GET: Use session context if available
    latest_data = session.get('latest_data')
    alerts = session.get('latest_alerts', [])
    if latest_data:
        return render_template('index.html', initial_data=[latest_data], alerts=alerts, summary=[], suggestions=[])
    else:
        return render_template('index.html', initial_data=[], alerts=[], summary=[], suggestions=[])

@app.route('/results')
def results():
    # Remove/disable this route or redirect to dashboard
    return redirect('/')

# --- Energy Schemes AI Assistant (RAG) Integration ---
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import zipfile
import tempfile
import shutil

# Configuration for RAG
class SchemesConfig:
    GROQ_API_KEY = "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"
    VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), 'Schemes_DB')

app.config.from_object(SchemesConfig)

vectorstore = None
retriever = None
qa_chain = None
llm = None
embedding = None

def initialize_rag_system():
    global llm, qa_chain, embedding, prompt_template
    try:
        os.environ["GROQ_API_KEY"] = app.config['GROQ_API_KEY']
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = ChatGroq(api_key=app.config['GROQ_API_KEY'], model_name="llama3-8b-8192")
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "You are an AI-powered virtual assistant dedicated to providing support and guidance about "
                "energy-related government schemes, subsidies, and incentives specifically for Indian MSMEs "
                "in the industrial sector.\n\n"
                "You are strictly limited to answering questions ONLY about:\n"
                "- Government schemes, subsidies, or incentives for industrial MSMEs\n"
                "- Energy efficiency programs for industrial MSMEs\n"
                "- Renewable energy initiatives and adoption for industrial MSMEs\n"
                "- Regulatory frameworks or policies affecting industrial MSME energy use\n"
                "- Energy-saving recommendations for industrial MSMEs\n\n"
                "Do NOT answer questions about non-industrial, household, agricultural, or commercial energy topics, "
                "or about MSMEs outside the industrial/manufacturing sector. If the question is outside this scope, "
                "simply respond with: 'Sorry, I can only answer questions related to energy schemes, incentives, or recommendations "
                "for industrial MSMEs in India.'\n\n"
                "Use accurate and concise responses based only on the provided context.\n"
                "Language should be clear and professional for plant operators, engineers, and MSME managers.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Answer:"
            )
        )
        qa_chain = prompt_template | llm
        return True
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

def load_schemes_vectorstore():
    global vectorstore, retriever
    try:
        vector_db_path = app.config['VECTOR_DB_PATH']
        print(f"[DEBUG] Attempting to load vector DB from: {vector_db_path}")
        if not os.path.exists(vector_db_path):
            print(f"❌ Vector database directory not found: {vector_db_path}")
            return False
        # No extraction needed, load Chroma directly from directory
        global embedding
        if embedding is None:
            print("[DEBUG] Embedding not initialized, initializing now...")
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        try:
            print(f"[DEBUG] Attempting to load Chroma with embedding function...")
            vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding)
            doc_count = vectorstore._collection.count()
            print(f"VectorDB loaded with {doc_count} docs (embedding)")
            if doc_count > 0:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                return True
        except Exception as e:
            print(f"Chroma load with embedding failed: {e}")
            print(f"[DEBUG] Attempting to load Chroma with default embedding...")
            vectorstore = Chroma(persist_directory=vector_db_path)
            doc_count = vectorstore._collection.count()
            print(f"VectorDB loaded with {doc_count} docs (default)")
            if doc_count > 0:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                return True
        return False
    except Exception as e:
        print(f"VectorDB load outer error: {e}")
        return False

def create_fallback_knowledge_base():
    schemes_data = [
        {
            "title": "UDAY Scheme (Ujwal DISCOM Assurance Yojana)",
            "content": """UDAY (Ujwal Discom Assurance Yojana) is a comprehensive scheme for operational and financial turnaround of State Electricity Boards (SEBs).\n\nObjectives:\n- Reduce AT&C (Aggregate Technical & Commercial) losses to 15%\n- Reduce gap between cost of supply and average revenue realized\n- Improve financial health of DISCOMs\n\nKey Incentives under UDAY:\n- <b>5% Incentive:</b> For DISCOMs achieving a capacity addition of up to 12% of their installed base capacity.\n- <b>10% Incentive:</b> For DISCOMs achieving a capacity addition of more than 12% but less than 30% of their installed base capacity.\n- <b>Incentive Cap:</b> The total incentives are limited to the available financial outlay, even if a DISCOM achieves a higher capacity addition.\n- <b>Eligibility:</b> These incentives apply only to the addition of the initial 18,000 MW of Renewable Thermal System (RTS) capacity after the launch of UDAY.\n\nFinancial Benefits:\n- Lower interest costs through state government bonds\n- Extended repayment periods\n- Performance-linked financial support\n- Central government grants for efficiency improvements\n\nOperational Reforms:\n- Mandatory energy audits\n- Smart metering for consumers above 200 units\n- Feeder separation for agriculture\n- LED distribution programs\n\n<b>Objective:</b> These incentives are designed to encourage DISCOMs to expand their capacity and improve efficiency, supporting the overall goal of a financially healthy and operationally efficient power distribution sector in India."""
        },
        {
            "title": "PAT Scheme (Perform, Achieve and Trade)",
            "content": """PAT (Perform, Achieve and Trade) is a market-based mechanism to enhance energy efficiency in energy-intensive industries.\n\nCoverage:\n- Thermal power plants\n- Iron & steel industry\n- Cement industry\n- Fertilizer industry\n- Aluminum industry\n- Textile industry\n- Paper & pulp industry\n- Chlor-alkali industry\n\nMechanism:\n- Mandatory energy reduction targets for designated consumers\n- Energy efficiency certificates (ESCerts) for excess reductions\n- Trading platform for certificates\n- Financial penalties for non-compliance\n\nBenefits:\n- Reduced energy costs\n- Lower greenhouse gas emissions\n- Revenue generation through certificate trading\n- Technology upgradation incentives\n\nEligibility:\n- Energy-intensive industries with consumption above threshold limits\n- Designated consumers under Energy Conservation Act\n\nImplementation:\n- Three-year cycles with specific reduction targets\n- Baseline energy consumption assessment\n- Annual monitoring and verification\n- Certificate issuance for overachievement"""
        },
        {
            "title": "MSME Energy Efficiency Schemes",
            "content": """Multiple schemes available for Micro, Small & Medium Enterprises (MSMEs) to improve energy efficiency:\n\n1. Credit Linked Capital Subsidy for Technology Upgradation (CLCSS):\n- 15% capital subsidy for technology upgradation\n- Maximum subsidy: ₹15 lakhs\n- Focus on energy-efficient technologies\n\n2. Energy Efficiency Financing Platform (EEFP):\n- Partial risk guarantee for energy efficiency projects\n- Lower interest rates for MSME borrowers\n- Technical assistance for project development\n\n3. Technology Upgradation Fund Scheme (TUFS):\n- Financial assistance for modern energy-efficient machinery\n- Reduced interest rates\n- Tax benefits\n\nBest Practices for MSMEs:\n- Energy audits and monitoring\n- LED lighting adoption\n- Variable frequency drives for motors\n- Power factor correction\n- Waste heat recovery systems\n- Solar water heating systems\n\nFinancial Support:\n- Subsidies ranging from 10-25% of project cost\n- Concessional loans through SIDBI\n- Technical assistance grants\n- Performance-based incentives"""
        },
        {
            "title": "Energy Conservation Building Code (ECBC)",
            "content": """ECBC provides minimum energy performance standards for commercial buildings.\n\nCoverage:\n- Commercial buildings with connected load ≥ 100 kW\n- New constructions and major renovations\n- Voluntary compliance with incentives\n\nKey Components:\n- Building envelope requirements\n- HVAC system efficiency standards\n- Lighting power density limits\n- Solar water heating mandates\n\nCompliance Levels:\n- ECBC: Basic compliance level\n- ECBC+: 25% more efficient than ECBC\n- Super ECBC: 50% more efficient than ECBC\n\nBenefits:\n- 25-40% energy savings\n- Reduced operating costs\n- Green building certification eligibility\n- Enhanced property value\n\nIncentives:\n- Fast-track approvals\n- Reduced property taxes in some states\n- Priority in government projects\n- Access to green financing"""
        }
    ]
    return schemes_data


# --- Formatting Utility ---
def clean_and_format_response(response):
    import re
    # Remove asterisks
    response = re.sub(r'\*+', '', response)
    # Only add a newline before numbered points at the start of a line or after a blank line, not in the middle of a sentence
    # This will match numbered points that are at the start of a line or after a newline and a space (not after a digit or letter)
    response = re.sub(r'(^|\n)(\s*)(\d+)\.\s+', r'\1\2\3. ', response)
    # Remove multiple newlines or spaces
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'[ \t]+\n', '\n', response)
    response = response.strip()
    # Wrap in a <div> with justified text for HTML rendering
    return f'<div style="text-align:justify;white-space:pre-line;">{response}</div>'

def answer_question_with_rag(question):
    if retriever and qa_chain:
        try:
            retrieved_docs = retriever.get_relevant_documents(question)
            print(f"[DEBUG] RAG: Retrieved {len(retrieved_docs) if retrieved_docs else 0} docs from vectorstore for question: '{question}'")
            if not retrieved_docs:
                print("[DEBUG] RAG: No docs found, trying similarity_search...")
                retrieved_docs = vectorstore.similarity_search(question, k=5)
            if retrieved_docs:
                context = " ".join([doc.page_content for doc in retrieved_docs])
                response = qa_chain.invoke({"question": question, "context": context})
                print("[DEBUG] RAG: Answer generated from vectorstore context. Returning used_vector_db=True.")
                return clean_and_format_response(response), True
            else:
                print("[DEBUG] RAG: No relevant docs found, using fallback. Returning used_vector_db=False.")
        except Exception as e:
            print(f"RAG error: {e}")
    print("[DEBUG] RAG: Using fallback knowledge base. Returning used_vector_db=False.")
    from flask import current_app
    return clean_and_format_response(current_app.view_functions['fallback_answer_question'](question)), False

@app.route('/schemes')
def schemes_index():
    return render_template('schemes.html')

@app.route('/tutorials')
def tutorials():
    return render_template('tutorials.html')

@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route('/best_practices')
def best_practices():
    return render_template('best_practices.html')

@app.route('/guides')
def guides():
    return render_template('guides.html')

@app.route('/toggle_admin', methods=['POST'])
def toggle_admin():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'No username provided'}), 400
    from users_db import get_all_users, update_user_admin_status
    users = get_all_users()
    user = next((u for u in users if u['username'] == username), None)
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    new_status = not user.get('is_admin', False)
    success = update_user_admin_status(username, new_status)
    if success:
        return jsonify({'success': True, 'is_admin': new_status})
    else:
        return jsonify({'success': False, 'error': 'Failed to update status'}), 500

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'No username provided'}), 400
    from users_db import delete_user
    success = delete_user(username)
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to delete user'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    try:
        answer, used_vector_db = answer_question_with_rag(question)
        print(f"[DEBUG] /ask: used_vector_db={used_vector_db}")
        return jsonify({'question': question, 'answer': answer, 'used_vector_db': used_vector_db})
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

# --- RS775 GenAI Integration (Vector DB + LLM) ---
from langchain_community.vectorstores import Chroma as ChromaCommunity
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddingsCommunity
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🧠 Prompt Template for Generating Motor Issue Recommendations
rs775_recommend_prompt_template = PromptTemplate(
    input_variables=["issue", "context"],
    template=(
        "You are an AI assistant with expertise in electrical engineering and motor systems. "
        "Based on the following datasheet context for the RS775-4538 DC motor and the issue described, "
        "provide a clear, practical recommendation for resolving or mitigating the issue. "
        "Base your reasoning on the datasheet when possible. If not covered, use standard best practices. "
        "Do NOT invent technical specs that are not in the datasheet.\n\n"
        "Datasheet Context:\n{context}\n\n"
        "Observed Issue:\n{issue}\n\n"
        "Recommended Action:"
    )
)

rs775_vectorstore = None
rs775_retriever = None
rs775_qa_chain = None
rs775_llm = None
rs775_embedding = None
rs775_initialized = False

RS775_VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), 'Schemes_DB', 'RS775_VectorDB')
RS775_GROQ_API_KEY = app.config.get('GROQ_API_KEY', None)

# Q&A Prompt for RS775
rs775_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an AI assistant that helps engineers understand technical information from motor datasheets. "
        "Use the provided context to answer the question accurately. Stick to the content of the RS775-4538 motor datasheet. "
        "If the answer is not available in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

def initialize_rs775_rag():
    global rs775_llm, rs775_qa_chain, rs775_embedding, rs775_vectorstore, rs775_retriever, rs775_initialized
    try:
        print("[DEBUG] (RS775) Initializing RS775 GenAI RAG system...")
        os.environ["GROQ_API_KEY"] = RS775_GROQ_API_KEY
        rs775_embedding = HFEmbeddingsCommunity(model_name="sentence-transformers/all-MiniLM-L6-v2")
        rs775_llm = ChatGroq(api_key=RS775_GROQ_API_KEY, model_name="llama3-8b-8192", temperature=0.2)
        rs775_vectorstore = ChromaCommunity(persist_directory=RS775_VECTOR_DB_PATH, embedding_function=rs775_embedding)
        rs775_retriever = rs775_vectorstore.as_retriever()
        rs775_qa_chain = LLMChain(llm=rs775_llm, prompt=rs775_prompt_template)
        rs775_initialized = True
        print("[DEBUG] (RS775) RS775 GenAI RAG system initialized successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] (RS775) Failed to initialize RS775 GenAI: {e}")
        rs775_initialized = False
        return False

# --- RS775 Recommendation Chain and Endpoint ---
rs775_recommend_chain = None

def initialize_rs775_recommend_chain():
    global rs775_recommend_chain, rs775_llm
    if rs775_llm is None:
        # Ensure LLM is initialized
        return False
    rs775_recommend_chain = LLMChain(llm=rs775_llm, prompt=rs775_recommend_prompt_template)
    return True

def answer_rs775_recommendation(issue):
    global rs775_initialized, rs775_retriever, rs775_recommend_chain
    if not rs775_initialized or not rs775_retriever or not rs775_recommend_chain:
        return "RS775 GenAI system is not initialized. Please try again later."
    try:
        # Retrieve context from vectorstore
        retrieved_docs = rs775_retriever.get_relevant_documents(issue)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        response = rs775_recommend_chain.run(issue=issue, context=context)
        return clean_and_format_response(response)
    except Exception as e:
        print(f"[ERROR] (RS775) Recommendation error: {e}")
        return clean_and_format_response("Sorry, there was an error processing your recommendation. Please try again later.")

def answer_rs775_question(question):
    global rs775_initialized, rs775_retriever, rs775_qa_chain
    if not rs775_initialized or not rs775_retriever or not rs775_qa_chain:
        return "RS775 GenAI system is not initialized. Please try again later."
    try:
        # Retrieve context from vectorstore
        retrieved_docs = rs775_retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        response = rs775_qa_chain.run(question=question, context=context)
        return clean_and_format_response(response)
    except Exception as e:
        print(f"[ERROR] (RS775) Q&A error: {e}")
        return clean_and_format_response("Sorry, there was an error processing your question. Please try again later.")

@app.route('/genai')
def genai_chat():
    return render_template('genai.html')

@app.route('/genai-recommend')
def genai_recommend():
    alerts = session.get('latest_alerts', [])
    return render_template('genai_recommend.html', alerts=alerts)

@app.route('/sample-questions')
def sample_questions():
    questions = [
        "Tell me schemes to increase energy efficiency in my MSME?",
        "What is the PAT scheme and how does it work?",
        "Explain the bemefits of installing rooftop solar panels in MSMEs.",
        "What is the capital subsidy offered under the TEQUP scheme for MSMEs?",
        "Recommend me a scheme for msme and give its incentives",
        "What is UDAY and explain the incentives given by it?",
        "Who is eligible for the Credit Linked Capital Subsidy Scheme (CLCSS) in the context of energy efficiency?"
    ]
    return jsonify({'questions': questions})

@app.route('/health')
def health_check():
    # Report vectorstore status for frontend
    return jsonify({'status': 'ok', 'vectorstore_loaded': vectorstore is not None}), 200

@app.route('/ask-genai', methods=['POST'])
def ask_genai():
    data = request.get_json()
    if not data or 'question' not in data:
        print('[DEBUG] /ask-genai: No question provided in request')
        return jsonify({'error': 'No question provided'}), 400
    question = data['question'].strip()
    if not question:
        print('[DEBUG] /ask-genai: Empty question received')
        return jsonify({'error': 'Question cannot be empty'}), 400
    try:
        answer = answer_rs775_question(question)
        if not answer or not isinstance(answer, str) or not answer.strip():
            print('[DEBUG] /ask-genai: No answer returned, using fallback message')
            answer = 'Sorry, I could not find an answer. Please try rephrasing your question.'
        print(f'[DEBUG] /ask-genai: Answer returned: {answer[:100]}...')
        return jsonify({'question': question, 'answer': answer})
    except Exception as e:
        print(f'[DEBUG] /ask-genai: Exception: {e}')
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500

@app.route('/recommend-genai', methods=['POST'])
def recommend_genai():
    data = request.get_json()
    if not data or 'issue' not in data:
        return jsonify({'error': 'No issue provided'}), 400
    issue = data['issue'].strip()
    if not issue:
        return jsonify({'error': 'Issue cannot be empty'}), 400
    try:
        answer = answer_rs775_recommendation(issue)
        if not answer or not isinstance(answer, str) or not answer.strip():
            answer = 'Sorry, I could not generate a recommendation. Please try rephrasing your issue.'
        return jsonify({'issue': issue, 'recommendation': answer})
    except Exception as e:
        return jsonify({'error': f'Error processing recommendation: {str(e)}'}), 500

@app.route('/get-alerts')
def get_alerts():
    alerts = session.get('latest_alerts', [])
    # Remove duplicates while preserving order
    seen = set()
    unique_alerts = []
    for alert in alerts:
        if alert not in seen:
            unique_alerts.append(alert)
            seen.add(alert)
    return jsonify({'alerts': unique_alerts})

@app.template_filter('markdown_to_html')
def markdown_to_html_filter(text):
    if not text:
        return ''
    # Use markdown with nl2br and sane_lists for better formatting
    return markdown.markdown(text, extensions=['nl2br', 'sane_lists'])

if __name__ == '__main__':
    print("[DEBUG] Starting unified Dashboard + Schemes app...")
    rag_initialized = initialize_rag_system()
    print(f"[DEBUG] RAG system initialized: {rag_initialized}")
    if rag_initialized:
        db_loaded = load_schemes_vectorstore()
        print(f"[DEBUG] VectorDB loaded: {db_loaded}")
        if db_loaded:
            print("🎉 Full RAG system ready! Starting Flask server...")
        else:
            print("⚠️ Vector database not loaded. Using fallback knowledge base.")
        # --- Initialize RS775 GenAI ---
        rs775_initialized = initialize_rs775_rag()
        print(f"[DEBUG] RS775 GenAI initialized: {rs775_initialized}")
        if rs775_initialized:
            initialize_rs775_recommend_chain()
        else:
            print("⚠️ RS775 GenAI initialization failed. RS775 endpoints will not work.")
    else:
        print("⚠️ RAG system initialization failed. Using fallback knowledge base only.")
    os.makedirs('uploads', exist_ok=True)
    print("[DEBUG] Flask app is about to run on http://localhost:5000 ...")
    app.run(debug=True)
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_all_metrics
from src.instance import (
    list_instances,
    load_instance,
    load_wikipedia_as_instance,
    save_instance,
)
from src.llm.client import get_default_client
from src.problem import Segmentation, SegmentationProblem
from src.solver import baseline as solver_baseline
from src.solver import two_phase as solver_two_phase

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Segmentación Óptima de Contenido",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para estética premium
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
    }
    .segment-container {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border-left: 6px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .paragraph-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        font-size: 15px;
        line-height: 1.5;
    }
    .paragraph-meta {
        font-size: 11px;
        color: #888;
        margin-bottom: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título Principal
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>🧩 Segmentación Óptima de Contenido</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 16px; margin-bottom: 30px;'>Optimización híbrida de coherencia textual usando Programación Dinámica y LLMs</p>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar - Controles e Importación
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("### 📥 Instancia de Datos")

# Cargar lista de instancias JSON disponibles
instances_dir = "data/instances"
Path(instances_dir).mkdir(parents=True, exist_ok=True)
instance_paths = list_instances(instances_dir)
instance_names = [Path(p).name for p in instance_paths]

selected_instance_name = st.sidebar.selectbox(
    "Selecciona una instancia:",
    options=instance_names,
    index=0 if instance_names else None,
    help="Elige una instancia JSON del repositorio local",
)

# Sección de descarga de Wikipedia
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Importar desde Wikipedia")
wiki_title = st.sidebar.text_input("Título del artículo:", placeholder="e.g., Inteligencia artificial")
wiki_lang = st.sidebar.selectbox("Idioma:", options=["es", "en", "fr", "de", "it"], index=0)

if st.sidebar.button("Descargar e Importar", use_container_width=True):
    if wiki_title.strip():
        with st.sidebar.status("Descargando de Wikipedia...", expanded=True) as status:
            try:
                prob = load_wikipedia_as_instance(page_title=wiki_title.strip(), language=wiki_lang)
                output_file = Path(instances_dir) / f"{prob.name}.json"
                save_instance(prob, str(output_file))
                status.update(label="¡Artículo importado con éxito!", state="complete", expanded=False)
                # Recargar la página para actualizar el selector de instancias
                st.rerun()
            except Exception as e:
                status.update(label=f"Error: {e}", state="error", expanded=True)
    else:
        st.sidebar.warning("Introduce un título válido.")

# Sección de descarga de YouTube
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎥 Importar desde YouTube")
yt_url = st.sidebar.text_input("Enlace o ID del video:", placeholder="https://www.youtube.com/watch?v=...")
yt_lang = st.sidebar.selectbox("Idioma principal (YT):", options=["es", "en"], index=0)
yt_words = st.sidebar.number_input(
    "Palabras promedio por párrafo (YT):",
    min_value=30,
    max_value=500,
    value=120,
    step=10,
    key="yt_words"
)

if st.sidebar.button("Descargar y Normalizar (YT)", use_container_width=True):
    if yt_url.strip():
        with st.sidebar.status("Descargando de YouTube...", expanded=True) as status:
            try:
                from src.utils.youtube_downloader import download_and_convert_youtube
                prob = download_and_convert_youtube(
                    url=yt_url.strip(),
                    target_word_count=int(yt_words),
                    languages=[yt_lang, "en"],
                    lambda_pen=lambda_pen,
                    min_seg=min_seg,
                )
                output_file = Path(instances_dir) / f"{prob.name}.json"
                save_instance(prob, str(output_file))
                status.update(label="¡Transcripción importada con éxito!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label=f"Error: {e}", state="error", expanded=True)
    else:
        st.sidebar.warning("Introduce un enlace o ID de video válido.")

# Sección de carga de archivo local
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Importar Transcripción Local")
uploaded_file = st.sidebar.file_uploader("Subir archivo de transcripción (.txt):", type=["txt"])
local_words = st.sidebar.number_input(
    "Palabras promedio por párrafo (Local):",
    min_value=30,
    max_value=500,
    value=120,
    step=10,
    key="local_words"
)

if uploaded_file is not None:
    if st.sidebar.button("Normalizar e Importar (Local)", use_container_width=True):
        with st.sidebar.status("Procesando archivo local...", expanded=True) as status:
            try:
                from src.utils.transcript_parser import convert_transcript_to_problem
                # Leer el contenido del archivo
                text_content = uploaded_file.read().decode("utf-8", errors="ignore")
                
                # Nombre de la instancia
                file_name = Path(uploaded_file.name).stem
                file_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in file_name)
                file_name = file_name.strip('_')
                
                prob = convert_transcript_to_problem(
                    text=text_content,
                    name=file_name,
                    target_word_count=int(local_words),
                    lambda_pen=lambda_pen,
                    min_seg=min_seg,
                )
                output_file = Path(instances_dir) / f"{prob.name}.json"
                save_instance(prob, str(output_file))
                status.update(label="¡Transcripción local importada con éxito!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label=f"Error: {e}", state="error", expanded=True)


# Sección de Configuración de Hiperparámetros
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configuración del Algoritmo")

solver_type = st.sidebar.selectbox(
    "Tipo de Solver:",
    options=["baseline_cosine", "two_phase"],
    format_func=lambda x: "Baseline Coseno" if x == "baseline_cosine" else "Two-Phase (Híbrido LLM)",
)

lambda_pen = st.sidebar.slider(
    "Penalización de cortes (λ):",
    min_value=0.01,
    max_value=5.00,
    value=0.50,
    step=0.01,
    help="Valores altos favorecen menos segmentos largos; valores bajos favorecen segmentos cortos y numerosos. Si λ <= 1.0, tiende a separar casi cada párrafo.",
)

min_seg = st.sidebar.number_input(
    "Longitud mínima de segmento:",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    help="Fuerza a que cada segmento tenga al menos este número de párrafos.",
)

k_max = st.sidebar.number_input(
    "Número máximo de cortes (k_max):",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
    help="Límite estricto de cortes. 0 significa sin límite.",
)

# Parámetros específicos de two_phase
if solver_type == "two_phase":
    ambiguity_threshold = st.sidebar.slider(
        "Umbral de ambigüedad coseno:",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Cortes con similitud coseno por encima de este umbral serán evaluados por el LLM.",
    )
    window_size = st.sidebar.slider(
        "Ventana de revisión local (w):",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Cantidad de elementos a evaluar a cada lado del corte ambiguo.",
    )
    refinement_method = st.sidebar.selectbox(
        "Método de Refinamiento:",
        options=["batch", "pairwise"],
        format_func=lambda x: "Lote (Rápido - 1 llamada/corte)" if x == "batch" else "Par-a-Par (Lento - w llamadas/corte)",
        help="El método 'Lote' envía la ventana completa al LLM de una vez, reduciendo drásticamente las llamadas y evitando límites de cuota.",
    )
    st.sidebar.markdown(
        """
        💡 **Tip de Velocidad Gratuito:**
        Para procesar a costo $0 y velocidad extrema:
        1. Cambia el modelo en tu `.env` a **Groq** (`groq/llama-3.3-70b-versatile`).
        2. O corre **Ollama** localmente (`ollama/llama3`).
        """
    )
else:
    ambiguity_threshold = 0.60
    window_size = 3
    refinement_method = "batch"

# Sección de Simulación Monte Carlo
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏳ Simulación Monte Carlo")
sim_replicas = st.sidebar.number_input(
    "Réplicas Monte Carlo:",
    min_value=5,
    max_value=200,
    value=50,
    step=5,
    help="Número de simulaciones independientes para estimar los intervalos de confianza.",
)
sim_rpm = st.sidebar.number_input(
    "Límite RPM de API:",
    min_value=1,
    max_value=1000,
    value=15,
    step=1,
    help="Peticiones por minuto máximas admitidas por el servidor (Gemini Free = 15).",
)
sim_mu = st.sidebar.slider(
    "Tiempo medio servicio LLM (s):",
    min_value=0.1,
    max_value=5.0,
    value=1.2,
    step=0.1,
    help="Tiempo promedio que tarda el LLM en responder una petición individual.",
)
sim_sigma = st.sidebar.slider(
    "Desviación estándar LLM (s):",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Dispersión estocástica del tiempo de servicio.",
)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de la instancia seleccionada
# ──────────────────────────────────────────────────────────────────────────────

if selected_instance_name:
    selected_path = Path(instances_dir) / selected_instance_name
    problem = load_instance(str(selected_path))
    # Sobrescribir los parámetros configurados en la interfaz
    problem.lambda_pen = lambda_pen
    problem.min_seg = min_seg
    problem.k_max = k_max if k_max > 0 else None
    
    st.info(
        f"**Instancia activa:** `{problem.name}` | "
        f"**Elementos (párrafos):** {problem.n} | "
        f"**Ground Truth:** {'Disponible' if problem.true_cuts is not None else 'No disponible'}"
    )
    
    # Calcular métricas básicas de colas / ambigüedad para simulación
    try:
        embeddings_sim = solver_baseline.compute_embeddings(problem)
        coh_matrix_sim = solver_baseline.build_coherence_matrix_cosine(embeddings_sim)
        seg0_sim = solver_baseline.dp_segmentation(problem, coh_matrix_sim)
        from src.solver.two_phase import _find_ambiguous_cuts
        ambiguous_cuts_sim = _find_ambiguous_cuts(list(seg0_sim.cuts), embeddings_sim, ambiguity_threshold)
        num_ambiguous_sim = len(ambiguous_cuts_sim)
        total_initial_cuts = len(seg0_sim.cuts)
    except Exception:
        num_ambiguous_sim = 0
        total_initial_cuts = 0
else:
    st.warning("No hay instancias JSON disponibles. Por favor, descarga un artículo de Wikipedia en la barra lateral o ejecuta `python src/main.py generate-instances` desde la terminal.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución del solver
# ──────────────────────────────────────────────────────────────────────────────

run_button = st.button("🚀 Ejecutar Segmentación", type="primary", use_container_width=True)

# Lógica del Solver
if run_button:
    with st.spinner("Ejecutando algoritmos de Programación Dinámica..."):
        t_start = time.perf_counter()
        
        # Cliente LLM
        client = get_default_client(verbose=False)
        
        try:
            if solver_type == "baseline_cosine":
                embeddings = solver_baseline.compute_embeddings(problem)
                coherence_matrix = solver_baseline.build_coherence_matrix_cosine(embeddings)
                seg, stats = solver_baseline.solve(problem)
            else:
                # Verificar API Key si se usa LLM
                api_key_set = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
                if not api_key_set:
                    st.warning("⚠️ No se ha detectado ninguna API Key en las variables de entorno. Las llamadas al LLM fallarán si no están en el caché. Asegúrate de configurar tu archivo `.env`.")
                
                embeddings = solver_baseline.compute_embeddings(problem)
                coherence_matrix = solver_baseline.build_coherence_matrix_cosine(embeddings)
                seg, stats = solver_two_phase.solve(
                    problem,
                    client=client,
                    ambiguity_threshold=ambiguity_threshold,
                    window_size=window_size,
                    refinement_method=refinement_method,
                    show_progress=False
                )
                
            metrics = compute_all_metrics(seg, problem, coherence_matrix, embeddings, stats)
            
            # Guardar resultados en session_state para persistencia al cambiar de pestaña
            st.session_state["results"] = {
                "seg": seg,
                "stats": stats,
                "metrics": metrics,
                "problem": problem,
                "coherence_matrix": coherence_matrix,
                "embeddings": embeddings,
            }
            
            st.success(f"¡Segmentación completada en {stats['time_total_s']:.4f} segundos!")
            
        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar el solver: {e}")
            st.exception(e)


# ──────────────────────────────────────────────────────────────────────────────
# Visualización de Resultados
# ──────────────────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    res = st.session_state["results"]
    seg: Segmentation = res["seg"]
    stats: dict = res["stats"]
    metrics: dict = res["metrics"]
    prob: SegmentationProblem = res["problem"]
    coherence_matrix = res["coherence_matrix"]
    embeddings = res["embeddings"]
    
    # 1. Dashboard de Métricas Clave
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{seg.num_segments()}</div>
                <div class="metric-label">Segmentos Creados</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{seg.score:.4f}</div>
                <div class="metric-label">Valor de Función Objetivo f(C)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cols[2]:
        f1_value = f"{metrics['boundary_f1']:.4f}" if metrics['boundary_f1'] is not None else "N/A"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{f1_value}</div>
                <div class="metric-label">Boundary F1 (vs Ground Truth)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{stats.get('llm_calls', 0)}</div>
                <div class="metric-label">Llamadas reales al LLM</div>
            </div>""",
            unsafe_allow_html=True,
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pestañas de Visualización
    tab_text, tab_plots, tab_sim = st.tabs([
        "📝 Texto Segmentado", 
        "📊 Gráficas e Insight Técnico", 
        "⏳ Simulación de Sistemas / Colas"
    ])
    
    # --- PESTAÑA 1: Texto Segmentado ---
    with tab_text:
        st.markdown("### Estructura de Segmentos")
        st.markdown("Cada bloque a continuación representa un segmento temático óptimo calculado por el algoritmo:")
        
        # Paleta de colores suaves (HSL) para diferenciar los segmentos
        segments = seg.segments()
        num_segs = len(segments)
        
        # Generar HSLs armoniosos distribuidos uniformemente en el círculo cromático
        colors = []
        for s_idx in range(num_segs):
            hue = int((s_idx * (360 / max(1, num_segs))) % 360)
            bg_color = f"hsl({hue}, 70%, 95%)"
            border_color = f"hsl({hue}, 75%, 65%)"
            text_color = f"hsl({hue}, 80%, 25%)"
            colors.append((bg_color, border_color, text_color))
            
        for s_idx, (start, end) in enumerate(segments):
            bg, border, text_c = colors[s_idx]
            
            st.markdown(
                f"""
                <div class="segment-container" style="background-color: {bg}; border-color: {border};">
                    <h4 style="margin-top: 0; color: {text_c}; font-weight: 700;">
                        Segmento {s_idx + 1} (Párrafos {start} - {end})
                    </h4>
                """,
                unsafe_allow_html=True,
            )
            
            # Renderizar los párrafos contenidos en el segmento
            for p_idx in range(start, end + 1):
                p_text = prob.elements[p_idx].text
                st.markdown(
                    f"""
                    <div class="paragraph-card">
                        <div class="paragraph-meta">PÁRRAFO {p_idx}</div>
                        {p_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            st.markdown("</div>", unsafe_allow_html=True)

    # --- PESTAÑA 2: Gráficas y Matrices ---
    with tab_plots:
        st.markdown("### Diagnóstico Técnico")
        
        plot_cols = st.columns(2)
        
        with plot_cols[0]:
            st.markdown("#### Matriz de Similitud Coseno (Fase 1)")
            st.markdown("Esta matriz representa la similitud semántica superficial de pares de párrafos (embeddings). Las zonas rojas/claras marcan alta similitud temática local.")
            
            # Calcular la matriz coseno completa para graficar
            n_elements = prob.n
            cos_matrix = np.zeros((n_elements, n_elements))
            for i in range(n_elements):
                for j in range(i, n_elements):
                    cos_matrix[i, j] = coherence_matrix[i][j]
                    cos_matrix[j, i] = coherence_matrix[i][j]  # hacerla simétrica
                    
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cos_matrix,
                cmap="coolwarm",
                vmin=0.2,
                vmax=1.0,
                square=True,
                xticklabels=max(1, n_elements // 10),
                yticklabels=max(1, n_elements // 10),
                ax=ax,
            )
            
            # Añadir líneas de los cortes predichos por la DP
            for cut in seg.cuts:
                ax.axvline(x=cut + 0.5, color="black", linestyle="--", linewidth=1.5)
                ax.axhline(y=cut + 0.5, color="black", linestyle="--", linewidth=1.5)
                
            plt.title("Matriz de Coherencia Coseno y Cortes Óptimos (DP)")
            st.pyplot(fig)
            
        with plot_cols[1]:
            st.markdown("#### Distribución de Longitud de Segmentos")
            st.markdown("Cantidad de párrafos agrupados por cada segmento temático detectado.")
            
            lengths = [end - start + 1 for start, end in segments]
            labels = [f"Seg {i+1}" for i in range(len(lengths))]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(labels, lengths, color="#1a73e8", edgecolor="black", alpha=0.8)
            ax.set_ylabel("Cantidad de Párrafos")
            ax.set_xlabel("Segmentos Detectados")
            ax.set_title("Longitud de los Segmentos")
            
            # Mostrar los valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
                
            st.pyplot(fig)
            
        # Tabla comparativa detallada
        st.markdown("---")
        st.markdown("#### Tabla Completa de Métricas")
        
        metrics_df = {
            "Métrica": [
                "Número de Segmentos",
                "Coherencia Interna Promedio (Intra-coh)",
                "Separación entre Fronteras (Inter-sep)",
                "Boundary F1 Score",
                "Tiempo Total de Ejecución",
                "Llamadas al LLM",
                "Caché Hits del LLM",
            ],
            "Valor": [
                str(seg.num_segments()),
                f"{metrics['intra_coherence']:.6f}",
                f"{metrics['inter_separation']:.6f}",
                str(f1_value),
                f"{stats['time_total_s']:.4f} s",
                str(stats.get("llm_calls", 0)),
                str(stats.get("cache_hits", 0)),
            ]
        }
        st.table(metrics_df)

    # --- PESTAÑA 3: Simulación de Sistemas / Colas ---
    with tab_sim:
        st.markdown("### 📊 Análisis de Simulación Estocástica y Colas")
        st.markdown(
            f"""
            Este análisis modela estocásticamente el resolvedor **Two-Phase** utilizando un simulador de eventos discretos. 
            Permite evaluar el rendimiento teórico del sistema bajo las cuotas gratuitas de la API sin realizar llamadas de red reales.
            
            **Métricas del Escenario Actual:**
            *   **Cortes iniciales detectados (Fase 1):** `{total_initial_cuts}`
            *   **Cortes ambiguos (Fase 2):** `{num_ambiguous_sim}` (con umbral coseno de `{ambiguity_threshold}`)
            *   **Ventana de revisión local ($w$):** `{window_size}`
            """
        )
        
        if num_ambiguous_sim == 0:
            st.warning("⚠️ No se detectaron cortes ambiguos con el umbral actual. Incrementa el umbral de ambigüedad en la barra lateral para simular la Fase 2.")
        else:
            # Ejecutar Simulación Monte Carlo
            from src.evaluation.simulation import simulate_comparison
            
            with st.spinner("Ejecutando réplicas de simulación Monte Carlo..."):
                sim_data = simulate_comparison(
                    num_ambiguous_cuts=num_ambiguous_sim,
                    window_size=window_size,
                    num_replicas=int(sim_replicas),
                    rpm_limit=int(sim_rpm),
                    service_mu=sim_mu,
                    service_sigma=sim_sigma,
                )
                
            # Extraer resultados
            batch_time, batch_time_lo, batch_time_hi = sim_data["batch"]["total_time"]
            pair_time, pair_time_lo, pair_time_hi = sim_data["pairwise"]["total_time"]
            
            batch_blocks, _, _ = sim_data["batch"]["blocks_triggered"]
            pair_blocks, _, _ = sim_data["pairwise"]["blocks_triggered"]
            
            batch_delay, _, _ = sim_data["batch"]["queue_delay"]
            pair_delay, _, _ = sim_data["pairwise"]["queue_delay"]
            
            # Comparativa rápida en tarjetas
            sim_cols = st.columns(3)
            with sim_cols[0]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value" style="color: #34a853;">{batch_time:.2f} s</div>
                        <div class="metric-label">Tiempo Batch Simulado<br>(IC 95%: [{batch_time_lo:.2f} - {batch_time_hi:.2f}])</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with sim_cols[1]:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value" style="color: #ea4335;">{pair_time:.2f} s</div>
                        <div class="metric-label">Tiempo Pairwise Simulado<br>(IC 95%: [{pair_time_lo:.2f} - {pair_time_hi:.2f}])</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with sim_cols[2]:
                ratio = pair_time / max(0.1, batch_time)
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value" style="color: #1a73e8;">{ratio:.1f}x</div>
                        <div class="metric-label">Factor de Aceleración<br>(Batch vs Pairwise)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gráficas
            g_cols = st.columns(2)
            
            with g_cols[0]:
                st.markdown("#### Tiempo de Ejecución Simulado e Intervalos de Confianza (95%)")
                fig_time, ax_time = plt.subplots(figsize=(6, 4.5))
                
                methods = ["Batch (Lote)", "Pairwise (Par-a-Par)"]
                means = [batch_time, pair_time]
                errors = [batch_time_hi - batch_time, pair_time_hi - pair_time]
                
                bars = ax_time.bar(methods, means, yerr=errors, capsize=8, color=["#34a853", "#ea4335"], edgecolor="black", alpha=0.8)
                ax_time.set_ylabel("Tiempo Total (segundos)")
                ax_time.set_title("Comparación de Tiempos Totales Simulación")
                
                # Anotar medias
                for bar in bars:
                    height = bar.get_height()
                    ax_time.annotate(
                        f"{height:.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold"
                    )
                st.pyplot(fig_time)
                
            with g_cols[1]:
                st.markdown("#### Histograma de Tiempos de Simulación Monte Carlo")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
                sns.histplot(sim_data["batch"]["raw_times"][0], kde=True, color="#34a853", label="Batch", ax=ax_hist, alpha=0.5)
                sns.histplot(sim_data["pairwise"]["raw_times"][0], kde=True, color="#ea4335", label="Pairwise", ax=ax_hist, alpha=0.5)
                ax_hist.set_xlabel("Tiempo Total (segundos)")
                ax_hist.set_ylabel("Frecuencia")
                ax_hist.set_title("Distribución Estocástica de Tiempos")
                ax_hist.legend()
                st.pyplot(fig_hist)
                
            # Mostrar tabla detallada
            st.markdown("---")
            st.markdown("#### Tabla Comparativa de Parámetros de Simulación")
            
            sim_df = {
                "Esquema de Simulación": ["Batch (Lote)", "Pairwise (Par-a-Par)"],
                "Total Peticiones a Servidor": [sim_data["params"]["requests_batch"], sim_data["params"]["requests_pairwise"]],
                "Tiempo Promedio Simulado": [f"{batch_time:.3f} s", f"{pair_time:.3f} s"],
                "Límite de Confianza 95%": [f"[{batch_time_lo:.3f} s, {batch_time_hi:.3f} s]", f"[{pair_time_lo:.3f} s, {pair_time_hi:.3f} s]"],
                "Retardo de Cola / Espera Promedio (429)": [f"{batch_delay:.2f} s", f"{pair_delay:.2f} s"],
                "Bloqueos 429 Promedio": [f"{batch_blocks:.2f}", f"{pair_blocks:.2f}"]
            }
            st.table(sim_df)
            
            # Texto explicativo de Teoría de Colas
            st.markdown(
                """
                ### 📚 Análisis del Sistema desde la Teoría de Colas (Sistemas de Línea de Espera)
                En el ámbito de la simulación y la teoría de colas, el resolvedor Two-Phase actúa como un **sistema de colas con servicio secuencial**:
                
                *   **Tasa de Arribo ($\lambda_a$):** Está gobernada por el método de refinamiento. En el método **Pairwise**, la tasa de arribo es elevada, ya que genera $(2w+1) \times 2$ peticiones por corte. En el método **Batch**, la tasa se reduce a 1 petición por corte.
                *   **Tasa de Servicio ($\mu_s$):** Es el tiempo de respuesta del LLM externo, modelado estocásticamente mediante una **distribución Normal** ($\mathcal{N}(\mu, \sigma)$).
                *   **Capacidad de Servicio Limitada (API Throttling):** La API gratuita impone una restricción rígida de $15$ RPM. Esto actúa como un **servidor con disciplina de cola especial**: si la ventana de 60 segundos se satura, el sistema se bloquea y se introduce un retardo forzado (tiempo de *backoff* de 15 segundos o lo necesario para limpiar la ventana deslizante).
                
                **Conclusión del Modelo:**
                El método **Batch** optimiza el sistema reduciendo la tasa de arribos de peticiones ($\lambda_a$), impidiendo de forma de terminista que se alcance la saturación del servidor externo. Esto elimina los retardos exponenciales introducidos por los bloqueos 429 y optimiza el tiempo de respuesta total.
                """
            )

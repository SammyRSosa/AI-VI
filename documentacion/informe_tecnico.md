# Informe Técnico — Segmentación Óptima de Contenido
## Proyecto Final: Inteligencia Artificial 2025-2026

> **Autor:** [Tu nombre]  
> **Tema:** 6 — Segmentación óptima de contenido  
> **Algoritmo principal:** Programación Dinámica (cortes óptimos)  
> **Rol del LLM:** Evaluador de coherencia semántica (Patrón #2)

---

## 1. Descripción del problema

### 1.1 Motivación

La segmentación de texto es una tarea fundamental en procesamiento de lenguaje natural: dado un documento largo o una secuencia de fragmentos, ¿dónde están los límites temáticos naturales? Aplicaciones típicas incluyen la estructuración automática de transcripts de podcast, la división de libros en capítulos temáticos, o la partición de videos educativos en módulos coherentes.

El reto central es que la "coherencia temática" es un concepto semántico que los algoritmos clásicos no pueden medir directamente. Aquí es donde el LLM aporta valor genuino: actúa como un evaluador de coherencia que va más allá de la similitud léxica superficial.

### 1.2 Definición informal

Dado un texto dividido en fragmentos E = [e₁, e₂, ..., eₙ], queremos encontrar las "divisiones naturales" que agrupan fragmentos temáticamente relacionados en segmentos contiguos.

---

## 2. Modelado formal

### 2.1 Definición del problema

**Entrada:**
- Secuencia E = [e₁, ..., eₙ] de elementos de texto.
- Función coherencia: Seg → [0, 1] que mide la unidad temática de un segmento.
- Hiperparámetro λ ≥ 0 (penalización por corte adicional).
- k_max ∈ ℕ ∪ {∞}: número máximo de cortes.

**Variables de decisión:**
- C = {c₁ < c₂ < ... < cₖ} ⊆ {0, 1, ..., n-2}: conjunto de cortes.
  - cⱼ es el índice del **último elemento** del segmento j.

**Segmentación inducida:**
- Segmento 0: E[0 .. c₁]
- Segmento j: E[cⱼ₋₁+1 .. cⱼ]  para j = 1, ..., k
- Segmento k: E[cₖ+1 .. n-1]

**Restricciones:**
1. Los segmentos son contiguos y no vacíos.
2. |C| ≤ k_max.
3. Cada segmento tiene ≥ min_seg elementos.

**Función objetivo (maximizar):**

```
f(C) = Σⱼ coherencia(E[cⱼ₋₁+1 .. cⱼ]) − λ · |C|
```

donde λ penaliza la fragmentación excesiva.

### 2.2 Complejidad

- **Espacio de soluciones:** 2^(n-1) posibles conjuntos de cortes.
- **Con DP:** O(n²) estados × O(n) transiciones → O(n²) con precomputación de la tabla de coherencias.
- **Espacio:** O(n²) para la matriz de coherencias + O(n) para los arrays dp y parent.

---

## 3. Descripción del dataset

### 3.1 Instancias sintéticas (generadas)

**Proceso de generación:**
1. Se definen K+1 temas a partir de un banco de 5 temas (astronomía, gastronomía, historia, informática, biología).
2. Cada tema tiene 7 párrafos de referencia escritos manualmente.
3. Se distribuyen n elementos entre K+1 segmentos, asignando a cada elemento un párrafo del tema correspondiente.
4. **Modelado de Ruido y Deriva Temática (Drift):** Se implementan dos variantes de ruido paramétrico con un `noise_ratio` ajustable (por defecto, 10%):
   - **Topic Swap (Intercambio de tema):** Reemplazo aleatorio de un párrafo por otro de un tema totalmente distinto.
   - **Filler Injection (Inyección de frases de transición):** Inserción de frases transitorias realistas (e.g., *"Cambiando de tema..."*, *"Ahora pasemos a..."*) al principio o al final de los párrafos en los límites del corte, con el fin de evaluar la robustez semántica de los solucionadores.
5. Los cortes verdaderos (ground truth) se conocen por construcción.

**Ventajas:**
- Verdad de campo conocida → permite calcular Boundary F1.
- Reproducible por semilla aleatoria.
- Control sobre dificultad (n, k, ruido).

**Tres tamaños de instancia:**

| Tamaño | n elementos | k cortes | Instancias |
|--------|-------------|---------|-----------|
| Small  | 15          | 3       | 5         |
| Medium | 60          | 7       | 5         |
| Large  | 250         | 12      | 5         |

**Total: 15 instancias, 45 corridas experimentales.**

### 3.2 Instancias reales (Wikipedia)

Para evaluar el sistema en escenarios del mundo real con límites temáticos determinados por humanos, el proyecto incorpora un módulo para descargar e importar artículos de Wikipedia en cualquier idioma (por defecto, español).

- **Proceso de importación:** El módulo realiza una consulta a la API de Wikipedia, parsea el HTML y extrae la secuencia de párrafos (elementos) ignorando secciones de ruido (como infoboxes, tablas de referencias y enlaces externos).
- **Fronteras de referencia:** Los encabezados del artículo (`h2`, `h3`), tanto si están directos como envueltos en divs contenedores de sección (`mw-heading`), se interpretan automáticamente como límites naturales de sección (`true_cuts`), permitiendo calcular de forma directa la métrica *Boundary F1* sobre datos reales.
- **Ejemplo disponible:** Se incluye la instancia `wikipedia_inteligencia_artificial.json` (98 párrafos, 28 cortes reales) descargada directamente desde la Wikipedia en español.

### 3.3 Por qué el dataset es válido

Las instancias sintéticas y reales cumplen con las condiciones del problema real:
1. Los párrafos dentro de cada segmento tratan el mismo tema (alta coherencia interna).
2. Los párrafos de segmentos distintos tratan temas distintos (baja coherencia inter-segmento).
3. El ruido o frases de transición inyectadas simulan la variabilidad real de los textos (digresiones, transiciones).
4. El ground truth permite evaluar correctitud objetivamente.

---

## 4. Diseño del algoritmo

### 4.1 Algoritmo principal: Programación Dinámica

**Pseudocódigo:**

```
FUNCIÓN DP-Segmentación(E, coherencia, λ, k_max):
    n ← |E|
    
    // Precomputar la matriz de coherencias
    PARA i desde 0 hasta n-1:
        PARA j desde i hasta n-1:
            coh[i][j] ← coherencia(E[i..j])
    
    // DP
    dp[i] ← -∞  para todo i
    parent[i] ← -1
    ncuts[i] ← 0
    
    PARA i desde 0 hasta n-1:
        PARA j desde 0 hasta i:
            // El segmento actual es E[j..i]
            cortes_si_j ← 0 si j=0, si no ncuts[j-1]+1
            
            SI cortes_si_j > k_max: CONTINUAR
            
            prev ← 0 si j=0, si no dp[j-1]
            penalización ← λ si j>0, si no 0
            val ← prev + coh[j][i] − penalización
            
            SI val > dp[i]:
                dp[i] ← val
                parent[i] ← j
                ncuts[i] ← cortes_si_j
    
    // Reconstruir cortes por backtracking
    cortes ← []
    pos ← n-1
    MIENTRAS VERDAD:
        j ← parent[pos]
        SI j = 0: ROMPER
        cortes.AGREGAR(j-1)
        pos ← j-1
    
    DEVOLVER INVERTIR(cortes), dp[n-1]
```

**Análisis de complejidad:**
- Precomputación de coherencias: O(n² · C) donde C = costo de evaluar coherencia.
  - Con coseno de embeddings: C = O(d), total O(n² · d).
  - Con LLM: C = O(1 llamada), amortizado con caché.
- DP: O(n²) operaciones aritméticas.
- Backtracking: O(k) ≤ O(n).
- **Total: O(n²)** con coherencias precomputadas.

**Correctitud:**
La DP satisface el principio de optimalidad de Bellman: la segmentación óptima de E[0..i] que termina con el segmento E[j..i] depende solo de la solución óptima de E[0..j-1]. No hay ciclos ni dependencias cruzadas. Por tanto, la DP es exacta.

### 4.2 Tres variantes implementadas

**Configuración A — baseline_cosine (control):**
- Coherencia = promedio de similitudes coseno entre pares de embeddings.
- Sin LLM. Baseline puro.

**Configuración B — dp_llm_full:**
- Coherencia = score LLM para TODOS los O(n²) rangos.
- Máxima calidad semántica pero máximo costo.

**Configuración C — two_phase (recomendada):**
- Fase 1: Coseno (igual que A) → segmentación S₀.
- Fase 2: LLM solo en cortes "borderline" (similitud coseno > 0.60).
- O(k · w²) llamadas al LLM, donde k ≤ n y w es la ventana (default 3).

---

## 5. Rol del LLM en el sistema

### 5.1 Patrón de integración: Evaluador (Patrón #2)

El LLM actúa como un **oráculo de coherencia semántica**. Dado un segmento de texto, el LLM devuelve un score [0.0, 1.0] que cuantifica su unidad temática. Este valor es el que el algoritmo DP no puede calcular por sí solo.

**¿Por qué el LLM aporta valor real aquí?**
La similitud coseno de embeddings captura proximidad semántica superficial, pero no detecta:
- Cambios de tema sutiles dentro de un párrafo.
- Coherencia narrativa o argumentativa (no solo léxica).
- Contexto pragmático (el texto es coherente aunque use palabras distintas).

### 5.2 Prompt literal

```
Eres un evaluador experto en análisis de coherencia textual.
Tu tarea es evaluar qué tan coherente y temáticamente unificado es el
fragmento de texto que se te proporcionará.

Criterios de evaluación:
- Un texto es COHERENTE si todos sus párrafos o frases tratan sobre
  el mismo tema o contribuyen a un único hilo narrativo.
- Un texto es INCOHERENTE si mezcla temas distintos y no relacionados.

<SEGMENT>
{segment_text}
</SEGMENT>

Responde ÚNICAMENTE con un objeto JSON con exactamente esta estructura:
{
  "score": <número decimal entre 0.0 y 1.0>,
  "razon": "<una frase concisa que justifique el score>"
}

Escala de referencia:
  1.0 → El segmento trata un único tema de forma completamente unificada.
  0.7 → Tema principal claro con pequeñas digresiones.
  0.5 → Mezcla dos temas parcialmente relacionados.
  0.3 → Temas distintos con hilo conductor débil.
  0.0 → Temas completamente distintos y no relacionados.
```

**Decisiones de diseño del prompt:**
1. **Temperatura = 0.0**: evaluación determinística, sin varianza aleatoria.
2. **Salida JSON forzada**: parseo determinístico, sin ambigüedad.
3. **Escala con anclas semánticas**: reduce la varianza inter-instancias del LLM.
4. **Delimitadores `<SEGMENT>`**: separa claramente el texto a evaluar del prompt.
5. **Truncamiento a 1500 palabras**: evita exceder el contexto del modelo.

### 5.3 Interacción con el algoritmo

```
Flujo en two_phase (Config C):

Embeddings → Coseno → DP → S₀ → Detección de cortes ambiguos
                                        ↓
                               Para cada corte ambiguo c:
                                   LLM(E[c-w..c]) → score_izq
                                   LLM(E[c+1..c+1+w]) → score_der
                                   Elegir c* que maximiza score_izq + score_der
                                        ↓
                                     S₁ (refinada)
```

### 5.4 Mecanismo de caché

- Clave: SHA-256(modelo || prompt_completo).
- Valor: {"score": float, "razon": str, "raw": str}.
- Almacenado en `data/llm_cache/*.json`.
- Beneficio: las re-ejecuciones de experimentos son O(1) de disco en vez de llamadas a la API.
- Reproducibilidad: los resultados del informe son reproducibles sin conexión.

---

## 6. Metodología experimental

### 6.1 Diseño

| Factor | Niveles |
|--------|---------|
| Configuración (algoritmo) | A, B, C |
| Tamaño de instancia | small, medium, large |
| Instancias por tamaño | 5 |

**Total: 3 × 15 = 45 corridas** (30 si se omite Config B en large).

### 6.2 Métricas registradas

| Métrica | Descripción | ↑ o ↓ |
|---------|-------------|-------|
| `intra_coherence` | Coherencia interna media de los segmentos | ↑ mejor |
| `inter_separation` | Distancia coseno entre centroides de segmentos adyacentes | ↑ mejor |
| `boundary_f1` | F1 de detección de fronteras vs. ground truth (±1) | ↑ mejor |
| `time_total_s` | Tiempo de ejecución total (segundos) | ↓ mejor |
| `llm_calls` | Número de llamadas reales al LLM | ↓ más eficiente |
| `objective_score` | Valor de f(C) producido por el solver | ↑ mejor |

### 6.3 Condiciones de reproducibilidad

- Semilla aleatoria fija (`RANDOM_SEED=42`) para generación de instancias.
- Temperatura LLM = 0.0 para determinismo.
- Caché de respuestas LLM versionado en `data/llm_cache/`.

---

## 7. Resultados y análisis

> **[TODO]** Esta sección se completa después de ejecutar `python src/main.py run-experiments`.
> Los resultados se generan automáticamente desde `data/results/results_all.csv`
> mediante `notebooks/analysis.ipynb`.

### 7.1 Tabla de resultados (placeholder)

| Config | Tamaño | Intra-coh. | Inter-sep. | Boundary F1 | Tiempo (s) | LLM calls |
|--------|--------|-----------|-----------|-------------|-----------|-----------|
| A — baseline | small | — | — | — | — | 0 |
| A — baseline | medium | — | — | — | — | 0 |
| B — dp_llm | small | — | — | — | — | — |
| C — two_phase | small | — | — | — | — | — |
| C — two_phase | medium | — | — | — | — | — |

### 7.2 Gráficas (placeholder)

Las siguientes gráficas se generarán en `notebooks/analysis.ipynb`:
1. **Barplot**: Intra-coherencia por config × tamaño.
2. **Barplot**: Boundary F1 por config × tamaño.
3. **Scatter**: Calidad (intra-coh) vs. Tiempo por config.
4. **Lineplot**: LLM calls vs. n por config.

---

## 8. Limitaciones y posibles mejoras

### 8.1 Limitaciones actuales

1. **LLM como oráculo ruidoso**: la evaluación de coherencia no es 100% consistente entre llamadas (aunque temperatura=0 mitiga esto). Un mismo segmento puede recibir scores ligeramente distintos en modelos diferentes.

2. **Complejidad O(n²)**: para n grande (>500 elementos), la precomputación de la matriz de coherencias es costosa, incluso con coseno.

3. **Coherencia como proxy de segmentación correcta**: un segmento muy largo puede tener coherencia aparente aunque combine subtemas.

4. **Calibración de λ**: la penalización por corte es un hiperparámetro que requiere ajuste por dominio.

### 8.2 Mejoras futuras

1. **TextTiling adaptativo**: usar tasas de cambio de similitud en lugar de umbrales fijos para detectar fronteras.

2. **DP con restricciones semánticas adicionales**: además de coherencia interna, penalizar segmentos que no tengan un "tema central identificable".

3. **Few-shot prompting**: incluir ejemplos de segmentos coherentes/incoherentes en el prompt para mejorar la calibración del LLM.

4. **Ensemble de evaluadores**: promediar scores de múltiples LLMs para reducir varianza.

### 8.3 Mejoras implementadas

1. **Evaluación con Datos Reales (Wikipedia):** Incorporación del módulo de scraping y parsing de Wikipedia, el cual extrae automáticamente la estructura de secciones (`h2`, `h3`) como `true_cuts` para posibilitar la evaluación objetiva de F1 en textos reales.
2. **Robustez ante Deriva Temática (Transiciones de Ruido):** Ampliación del generador sintético para inyectar frases de transición realistas en las fronteras de corte, lo que permite evaluar el comportamiento de los solucionadores ante sutiles cambios conversacionales.

---

## Referencias

- [1] Hearst, M. A. (1997). TextTiling: Segmenting text into multi-paragraph subtopic passages. *Computational Linguistics*, 23(1), 33-64.
- [2] Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- [3] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *EMNLP 2019*.
- [4] Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS 2020*.

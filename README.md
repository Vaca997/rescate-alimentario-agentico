# Sistema Agéntico de Reducción de Desperdicio Alimentario

MVP alineado con el **ODS 2 (Hambre Cero)** y la **Meta 12.3** de reducción
del desperdicio alimentario. El sistema administra lotes próximos a caducar,
propone acciones de rescate y registra las autorizaciones para mantener una
traza auditable.

## Arquitectura modular

El proyecto separa responsabilidades siguiendo principios SOLID, en particular
el Principio de Responsabilidad Única:

- **`models.py`** — entidades del dominio e invariantes: lotes de alimentos,
  trazas ReAct, bancos de alimentos y combos de rescate.
- **`database.py`** — persistencia local en DuckDB, carga de semilla, historial
  de decisiones y operaciones de autorización/reversión.
- **`agents.py`** — cliente resiliente para Ollama, agentes ReAct, orquestación
  de análisis, asignación logística y cálculo de impacto ODS.
- **`app.py`** — interfaz interactiva del MVP construida con Streamlit.

La semilla inicial se encuentra en `seed_inventory.csv`. El directorio `docs/`
está reservado para el Documento de Visión y Alcances y los diagramas UML del
proyecto.

## Funcionalidades del MVP

- Contador en tiempo real hasta el final del día de caducidad y simulador para
  adelantar el reloj.
- Semáforo de cuatro colores: lotes con tiempo suficiente, próximos a caducar,
  urgentes y caducados/descartados.
- Asignación multicriterio a bancos de alimentos, comprobando capacidad,
  distancia y cadena de frío para carnes y lácteos.
- Combos de rescate con ingredientes cruzados, recetas breves y descuento
  social.
- Tres modos de análisis: rápido global, exhaustivo por lote e instantáneo
  determinista. Ante fallos de Ollama, el agente usa reglas de respaldo.
- Registro auditable de trazas, autorización de acciones y reversión de
  autorizaciones.
- Métricas de impacto estimado para kg rescatados, CO₂e evitado y raciones
  equivalentes.

## Requisitos

- Python 3.10 o posterior.
- Ollama instalado y disponible localmente para usar inferencia LLM. El modo
  determinista funciona sin conexión con Ollama.

## Instalación y ejecución

Abre una terminal en la raíz del proyecto y ejecuta:

1. Crea un entorno virtual:

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Descarga el modelo de Ollama utilizado por defecto:

   ```bash
   ollama pull llama3.2:1b
   ```

   Asegúrate de que Ollama esté ejecutándose antes de solicitar análisis con
   LLM. Si no está disponible, selecciona el modo determinista en la aplicación.

4. Inicia el dashboard:

   ```bash
   streamlit run app.py
   ```

La aplicación crea `inventario_rescate.db` localmente y carga la semilla si el
inventario está vacío. El archivo de base de datos se excluye de Git mediante
`.gitignore`. Opcionalmente, coloca `background.png` o `background.jpg` en la
raíz para personalizar el fondo; `logo.png` se muestra en la barra lateral si
está presente.

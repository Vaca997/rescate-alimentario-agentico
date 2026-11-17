import base64
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from agents import (
    AgenteDeteccion,
    CalculadoraImpactoODS,
    ClienteOllamaResiliente,
    OrquestadorRescate,
    obtener_bancos_alimentos,
)
from database import InventarioDB
from models import LoteAlimento, TrazaDecisionReAct


st.set_page_config(
    page_title="Rescate Alimentario Agéntico - MVP",
    layout="wide",
)

SEED_PATH = Path(__file__).with_name("seed_inventory.csv")
PROJECT_DIR = Path(__file__).parent


def aplicar_estilos_globales() -> None:
    imagen_fondo = next(
        (
            PROJECT_DIR / nombre
            for nombre in ("background.png", "background.jpg")
            if (PROJECT_DIR / nombre).is_file()
        ),
        None,
    )
    if imagen_fondo is None:
        fondo = (
            "radial-gradient(circle at top right, #123524, #091218 70%)"
        )
    else:
        mime_type = (
            "image/png" if imagen_fondo.suffix.lower() == ".png" else "image/jpeg"
        )
        contenido_base64 = base64.b64encode(
            imagen_fondo.read_bytes()
        ).decode("ascii")
        fondo = (
            "linear-gradient(rgba(8, 15, 12, 0.84), "
            f"rgba(10, 18, 24, 0.90)), url(\"data:{mime_type};base64,"
            f"{contenido_base64}\")"
        )

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {fondo};
                background-size: cover;
                background-attachment: fixed;
            }}
            .stAppDeployButton,
            [data-testid="stAppDeployButton"] {{
                display: none !important;
            }}
            [data-testid="stMetric"],
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stExpander"] {{
                background: rgba(17, 25, 33, 0.78);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 12px;
            }}
            [data-testid="stDataFrame"] {{
                background: rgba(17, 25, 33, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 12px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


aplicar_estilos_globales()

db = InventarioDB()
db.inicializar_esquema()

if not db.obtener_todos_los_lotes():
    db.cargar_semilla_csv(str(SEED_PATH))

st.title("Rescate Alimentario Agéntico")
st.caption("MVP alineado con el ODS 2 y la Meta 12.3")

if "mensaje_analisis" in st.session_state:
    st.success(st.session_state.pop("mensaje_analisis"))
if "mensaje_limpieza" in st.session_state:
    st.success(st.session_state.pop("mensaje_limpieza"))

lotes = db.obtener_todos_los_lotes()
impacto = CalculadoraImpactoODS().calcular_resumen_impacto(lotes)
columnas_metricas = st.columns(5)
columnas_metricas[0].metric("Kg Rescatados", f"{impacto['kg_rescatados']:.2f}")
columnas_metricas[1].metric(
    "CO₂e Evitado (kg)", f"{impacto['co2_evitado_kg']:.2f}"
)
columnas_metricas[2].metric(
    "Raciones Equivalentes", impacto["raciones_equivalentes"]
)
columnas_metricas[3].metric("Lotes Donados", impacto["lotes_donados"])
columnas_metricas[4].metric(
    "Lotes en Descuento Social", impacto["lotes_descuento"]
)

with st.expander(
    "Ver Directorio de Bancos de Alimentos y Restricciones Logísticas (FE-03)",
    expanded=False,
):
    bancos_df = pd.DataFrame(
        [
            {
                "Nombre": banco.nombre,
                "Distancia (km)": banco.distancia_km,
                "Capacidad (kg)": banco.capacidad_disponible_kg,
                "¿Tiene cadena de frío?": (
                    "Sí" if banco.tiene_cadena_frio else "No"
                ),
                "Prioridad nutricional": banco.prioridad_nutricional,
            }
            for banco in obtener_bancos_alimentos()
        ]
    )
    st.dataframe(bancos_df, width="stretch", hide_index=True)

with st.sidebar:
    logo_path = PROJECT_DIR / "logo.png"
    if logo_path.is_file():
        st.image(str(logo_path), width=110)
    st.header("Control de inventario")
    if st.button("Reiniciar Inventario Semilla (CSV)", use_container_width=True):
        cantidad = db.cargar_semilla_csv(str(SEED_PATH))
        st.session_state.pop("combos_rescate", None)
        st.success(f"Inventario reiniciado: {cantidad} lotes cargados.")
        st.rerun()

    if st.button(
        "Limpiar Historial y Contexto del Agente",
        use_container_width=True,
    ):
        db.limpiar_historial_trazas()
        contexto_liberado = (
            ClienteOllamaResiliente().liberar_contexto_memoria()
        )
        st.session_state.clear()
        estado_contexto = (
            "El modelo Ollama se descargó de memoria."
            if contexto_liberado
            else "Ollama no está disponible; no se pudo liberar su modelo."
        )
        st.session_state["mensaje_limpieza"] = (
            "Historial de trazas limpiado. " + estado_contexto
        )
        st.rerun()

    st.caption("Simulador de paso del tiempo (+ días)")
    adelantar_dias = st.slider("Adelantar reloj (días)", 0, 10, 0)

    with st.expander("Agregar Lote Manual (Prueba de Invariantes)"):
        with st.form("formulario_lote_manual"):
            id_lote = st.text_input("ID del lote")
            producto = st.text_input("Producto")
            categoria = st.selectbox(
                "Categoría",
                ["Lacteos", "Panaderia", "Carnes", "Frutas_Verduras"],
            )
            unidades = st.number_input("Unidades", value=1, step=1)
            peso_unitario_kg = st.number_input(
                "Peso unitario (kg)", value=1.0, step=0.1, format="%.2f"
            )
            precio_base = st.number_input(
                "Precio base", value=1.0, step=0.1, format="%.2f"
            )
            fecha_caducidad = st.date_input(
                "Fecha de caducidad real",
                value=date.today() + timedelta(days=5),
            )
            agregar_lote = st.form_submit_button("Agregar lote")

        if agregar_lote:
            try:
                lote_manual = LoteAlimento(
                    id_lote=id_lote,
                    producto=producto,
                    categoria=categoria,
                    unidades=int(unidades),
                    peso_unitario_kg=float(peso_unitario_kg),
                    precio_base=float(precio_base),
                    fecha_caducidad=fecha_caducidad,
                )
                db.insertar_lote(lote_manual)
            except ValueError as error:
                st.error(f"No se pudo agregar el lote: {error}")
            else:
                st.success(f"Lote {lote_manual.id_lote} agregado.")
                st.rerun()

try:
    fragment_inventario = st.fragment(run_every="1s")
except (AttributeError, TypeError):
    def fragment_inventario(funcion):
        return funcion


@fragment_inventario
def mostrar_inventario() -> None:
    st.subheader("Inventario")
    ahora_sistema = datetime.now()
    ahora_simulado = ahora_sistema + timedelta(days=adelantar_dias)
    st.caption(
        f"Reloj del sistema: {ahora_sistema.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if adelantar_dias:
        st.caption(
            "Reloj simulado: "
            f"{ahora_simulado.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(+{adelantar_dias} días)"
        )

    lotes_actuales = db.obtener_todos_los_lotes()
    if not lotes_actuales:
        st.info("No hay lotes registrados.")
        return

    iconos_categoria = {
        "Lacteos": "🥛 Lacteos",
        "Panaderia": "🥖 Panaderia",
        "Carnes": "🥩 Carnes",
        "Frutas_Verduras": "🍓 Frutas_Verduras",
    }
    filas = []
    for lote in lotes_actuales:
        dias_restantes = lote.dias_restantes_Respecto_a(
            ahora_simulado.date()
        )
        filas.append(
            {
                "id_lote": lote.id_lote,
                "producto": lote.producto,
                "categoria": iconos_categoria.get(
                    lote.categoria, lote.categoria
                ),
                "unidades": lote.unidades,
                "peso_total_kg": lote.peso_total_kg,
                "precio_base": lote.precio_base,
                "fecha_caducidad": lote.fecha_caducidad,
                "dias_para_caducar": dias_restantes,
                "cuenta_regresiva": lote.tiempo_restante_formateado(
                    ahora_simulado
                ),
                "estado": lote.estado,
            }
        )
    inventario_df = pd.DataFrame(filas)

    def estilo_fila(fila: pd.Series) -> list[str]:
        if fila["dias_para_caducar"] < 0 or fila["estado"] == "DESCARTADO":
            fondo, texto = "#2E1A36", "#E9D5FF"
        elif fila["dias_para_caducar"] <= 2:
            fondo, texto = "#4C141B", "#FCA5A5"
        elif fila["dias_para_caducar"] <= 5:
            fondo, texto = "#422D0B", "#FCD34D"
        else:
            fondo, texto = "#113826", "#6EE7B7"
        return [f"background-color: {fondo}; color: {texto};"] * len(fila)

    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 12px;">
          <span style="background:#113826;color:#6EE7B7;padding:6px 10px;border-radius:8px;">🟢 Verde: 6+ días</span>
          <span style="background:#422D0B;color:#FCD34D;padding:6px 10px;border-radius:8px;">🟡 Amarillo: 3–5 días</span>
          <span style="background:#4C141B;color:#FCA5A5;padding:6px 10px;border-radius:8px;">🔴 Rojo: 0–2 días</span>
          <span style="background:#2E1A36;color:#E9D5FF;padding:6px 10px;border-radius:8px;">🟣 Morado: caducado o descartado</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        inventario_df.style.apply(estilo_fila, axis=1),
        width="stretch",
        hide_index=True,
        key="tabla_inventario_tiempo_real",
    )


mostrar_inventario()

st.subheader("Análisis agéntico ReAct")
contexto_base = st.selectbox(
    "Contexto operativo del día (Adaptación dinámica)",
    [
        "Operación normal",
        "Alerta: Falla en cámara de refrigeración (Urgencia en Lácteos y Carnes)",
        "Alerta: Lluvia intensa y tráfico pesado (Priorizar rutas cortas y mayor descuento)",
        "Solicitud urgente: Comedor Infantil requiere pan, fruta y lácteos",
    ],
)
nota_adicional = st.text_input("Nota adicional del gerente (opcional)")
modo_ui = st.radio(
    "Modo de análisis agéntico",
    [
        "Análisis Rápido Global (1 sola llamada a Ollama - Recomendado)",
        "Análisis Exhaustivo Lote por Lote (Múltiples llamadas a Ollama)",
        "Modo Instantáneo Determinista (Fallback sin uso de CPU)",
    ],
)
modo_por_etiqueta = {
    "Análisis Rápido Global (1 sola llamada a Ollama - Recomendado)": (
        "rapido_global"
    ),
    "Análisis Exhaustivo Lote por Lote (Múltiples llamadas a Ollama)": (
        "exhaustivo_lote"
    ),
    "Modo Instantáneo Determinista (Fallback sin uso de CPU)": (
        "fallback_determinista"
    ),
}
max_lotes = st.slider(
    "Cantidad de lotes urgentes a analizar",
    1,
    10,
    6,
)

if st.button(
    "Ejecutar Orquestación Agéntica",
    type="primary",
    width="stretch",
):
    contexto_operativo = contexto_base
    if nota_adicional.strip():
        contexto_operativo += f". Nota del gerente: {nota_adicional.strip()}"
    orquestador = OrquestadorRescate(
        db,
        [
            AgenteDeteccion(
                ClienteOllamaResiliente(),
                contexto_operativo=contexto_operativo,
                bancos=obtener_bancos_alimentos(),
            )
        ],
    )
    try:
        with st.spinner("Analizando los lotes más urgentes..."):
            trazas_generadas, combos_generados = (
                orquestador.procesar_inventario_en_riesgo(
                    dias_umbral=5,
                    max_lotes=max_lotes,
                    modo_ejecucion=modo_por_etiqueta[modo_ui],
                    contexto_operativo=contexto_operativo,
                )
            )
    except Exception as error:
        st.error(f"No se pudo completar el análisis: {error}")
    else:
        st.session_state["combos_rescate"] = combos_generados
        st.session_state["mensaje_analisis"] = (
            f"Análisis completado: {len(trazas_generadas)} trazas y "
            f"{len(combos_generados)} combos generados."
        )
        st.rerun()

st.subheader("Combos y Recetas de Rescate Generados (Agente Promotor - FE-04)")
combos_rescate = st.session_state.get("combos_rescate", [])
if not combos_rescate:
    st.info("Ejecuta la orquestación para generar combos y recetas de rescate.")

lotes_por_id = {lote.id_lote: lote for lote in lotes}
for combo in combos_rescate:
    with st.container(border=True):
        st.markdown(f"### {combo.nombre_combo}")
        st.markdown(f"**Productos:** {combo.productos_descripcion}")
        st.markdown(
            f"**Lotes cruzados:** {', '.join(combo.lotes_incluidos)} · "
            f"**Origen:** {combo.origen}"
        )
        st.markdown(
            f"**Precio original:** ${combo.precio_original_total:.2f} · "
            f"**Oferta social ({combo.porcentaje_descuento}%):** "
            f"${combo.precio_combo_descuento:.2f}"
        )
        st.markdown(f"**Receta rápida:** {combo.receta_sugerida}")
        lotes_combo = [
            lotes_por_id[id_lote]
            for id_lote in combo.lotes_incluidos
            if id_lote in lotes_por_id
        ]
        if len(lotes_combo) != len(combo.lotes_incluidos):
            st.warning(
                "El inventario cambió desde que se generó este combo. "
                "Vuelve a ejecutar el análisis."
            )
        elif all(lote.estado == "ACTIVO" for lote in lotes_combo):
            if st.button(
                "Autorizar Combo Completo",
                key=f"autorizar_combo_{combo.id_combo}",
                type="primary",
            ):
                trazas_combo = [
                    TrazaDecisionReAct(
                        id_traza=uuid4().hex[:8],
                        id_lote=lote.id_lote,
                        nombre_agente="AgentePromotor_FE04",
                        pensamiento=(
                            f"Autorización del combo {combo.nombre_combo}: "
                            "se aplicará el precio social del combo."
                        ),
                        accion_sugerida="DESCUENTO",
                        detalle_accion=(
                            f"Combo {combo.nombre_combo} con "
                            f"{combo.porcentaje_descuento}% de descuento."
                        ),
                        fue_fallback=(
                            combo.origen != "Inferencia LLM (Ollama)"
                        ),
                    )
                    for lote in lotes_combo
                ]
                try:
                    db.autorizar_combo(trazas_combo)
                except ValueError as error:
                    st.error(f"No se pudo autorizar el combo: {error}")
                else:
                    st.rerun()
        else:
            st.info("Uno o más lotes de este combo ya no están activos.")

st.subheader("Auditoría de trazas ReAct y autorización")
historial = db.obtener_historial_trazas()
if not historial:
    st.info("Aún no hay trazas registradas.")

for traza in historial:
    with st.container(border=True):
        origen = (
            "Fallback Determinista (Reglas)"
            if traza.fue_fallback
            else "Inferencia LLM (Ollama)"
        )
        st.markdown(
            f"**Lote:** {traza.id_lote} · "
            f"**Agente:** {traza.nombre_agente} · "
            f"**Origen:** {origen}"
        )
        st.markdown(f"**Acción sugerida:** `{traza.accion_sugerida}`")
        st.markdown(f"**Pensamiento:** {traza.pensamiento}")
        st.markdown(f"**Detalle de acción:** {traza.detalle_accion}")

        if traza.autorizada:
            columna_estado, columna_reversion = st.columns([1, 2])
            with columna_estado:
                st.success("Autorizada y Aplicada")
            with columna_reversion:
                if st.button(
                    "Revertir Autorización (Deshacer)",
                    key=f"revertir_{traza.id_traza}",
                ):
                    try:
                        db.revertir_autorizacion(
                            traza.id_traza,
                            traza.id_lote,
                        )
                    except ValueError as error:
                        st.error(f"No se pudo revertir la autorización: {error}")
                    else:
                        st.rerun()
        elif st.button(
            "Autorizar Acción",
            key=f"autorizar_{traza.id_traza}",
        ):
            try:
                db.autorizar_accion(
                    traza.id_traza,
                    traza.id_lote,
                    traza.accion_sugerida,
                )
            except ValueError as error:
                st.error(f"No se pudo autorizar la acción: {error}")
            else:
                st.rerun()

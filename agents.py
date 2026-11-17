import json
import logging
import unicodedata
from abc import ABC, abstractmethod
from uuid import uuid4

import httpx
import ollama

from database import InventarioDB
from models import (
    BancoAlimentos,
    ComboRescate,
    LoteAlimento,
    TrazaDecisionReAct,
)

logger = logging.getLogger(__name__)


def _normalizar_texto(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(
        caracter
        for caracter in normalizado
        if not unicodedata.combining(caracter)
    )


def obtener_bancos_alimentos() -> list[BancoAlimentos]:
    return [
        BancoAlimentos(
            id_banco="BAMX-01",
            nombre="Banco de Alimentos Central BAMX",
            distancia_km=4.2,
            capacidad_disponible_kg=45.0,
            tiene_cadena_frio=True,
            prioridad_nutricional="Proteínas y lácteos para familias",
        ),
        BancoAlimentos(
            id_banco="COM-02",
            nombre="Comedor Infantil Esperanza",
            distancia_km=1.5,
            capacidad_disponible_kg=20.0,
            tiene_cadena_frio=False,
            prioridad_nutricional="Panadería, fruta y energía inmediata",
        ),
        BancoAlimentos(
            id_banco="CHOG-03",
            nombre="Casa Hogar San José",
            distancia_km=2.8,
            capacidad_disponible_kg=25.0,
            tiene_cadena_frio=True,
            prioridad_nutricional="Lácteos, carnes y verduras",
        ),
    ]


class ClienteOllamaResiliente:
    def __init__(
        self,
        modelo: str = "llama3.2:1b",
        timeout_segundos: float = 15.0,
        max_reintentos: int = 2,
    ) -> None:
        self.modelo = modelo
        self.timeout_segundos = timeout_segundos
        self.max_reintentos = max_reintentos
        self.cliente = ollama.Client(timeout=timeout_segundos)

    def liberar_contexto_memoria(self) -> bool:
        try:
            self.cliente.generate(
                model=self.modelo,
                prompt="",
                keep_alive=0,
            )
            return True
        except Exception:
            return False

    def invocar_json(self, system_prompt: str, user_prompt: str) -> dict:
        return self._invocar_json(
            system_prompt,
            user_prompt,
            validar_decision=True,
            max_tokens=120,
        )

    def invocar_json_estructurado(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 768,
        max_intentos: int | None = None,
    ) -> dict:
        return self._invocar_json(
            system_prompt,
            user_prompt,
            validar_decision=False,
            max_tokens=max_tokens,
            max_intentos=max_intentos,
        )

    def _invocar_json(
        self,
        system_prompt: str,
        user_prompt: str,
        validar_decision: bool,
        max_tokens: int,
        max_intentos: int | None = None,
    ) -> dict:
        prompt_usuario = user_prompt

        intentos = (
            self.max_reintentos if max_intentos is None else max_intentos
        )
        for _ in range(intentos):
            try:
                respuesta = self.cliente.chat(
                    model=self.modelo,
                    format="json",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_usuario},
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": max_tokens,
                        "num_ctx": 1024,
                    },
                )
                contenido = respuesta.message.content
                resultado = json.loads(contenido)
                if not isinstance(resultado, dict):
                    raise ValueError("La respuesta JSON debe ser un objeto.")
                if validar_decision:
                    self._validar_decision(resultado)
                return resultado
            except (ConnectionError, httpx.NetworkError) as error:
                raise RuntimeError(
                    "No se pudo conectar con Ollama; se usará el fallback determinista."
                ) from error
            except Exception as error:
                prompt_usuario = (
                    f"{user_prompt}\n\nLa respuesta anterior no fue válida "
                    f"({type(error).__name__}: {error}). Devuelve exclusivamente "
                    "un objeto JSON con las llaves solicitadas y accion_sugerida "
                    "igual a DONADO o DESCUENTO."
                )

        raise RuntimeError("Se agotaron los reintentos de inferencia con Ollama.")

    @staticmethod
    def _validar_decision(resultado: dict) -> None:
        pensamiento = resultado["pensamiento"]
        accion_original = resultado["accion_sugerida"]
        detalle = resultado["detalle_accion"]

        if not isinstance(accion_original, str):
            raise ValueError("accion_sugerida debe ser una cadena.")
        accion_normalizada = accion_original.upper()
        if "DON" in accion_normalizada:
            accion = "DONADO"
        elif "DESC" in accion_normalizada or "OFER" in accion_normalizada:
            accion = "DESCUENTO"
        else:
            accion = accion_normalizada

        if accion not in {"DONADO", "DESCUENTO"}:
            raise ValueError(
                "accion_sugerida debe ser 'DONADO' o 'DESCUENTO'."
            )
        if not all(
            isinstance(valor, str)
            for valor in (pensamiento, accion_original, detalle)
        ):
            raise ValueError(
                "pensamiento, accion_sugerida y detalle_accion "
                "deben ser cadenas."
            )
        resultado["accion_sugerida"] = accion


class AgenteReActBase(ABC):
    def __init__(self, nombre: str, cliente_llm: ClienteOllamaResiliente) -> None:
        self.nombre = nombre
        self.cliente_llm = cliente_llm

    def evaluar_lote(self, lote: LoteAlimento) -> TrazaDecisionReAct:
        if lote.esta_caducado:
            return TrazaDecisionReAct(
                id_traza=uuid4().hex[:8],
                id_lote=lote.id_lote,
                nombre_agente=self.nombre,
                pensamiento=(
                    "El lote está caducado; por normativa alimentaria no se "
                    "puede donar ni vender."
                ),
                accion_sugerida="DESCARTADO",
                detalle_accion="Retirar el lote del inventario y descartarlo "
                "según el procedimiento sanitario.",
                fue_fallback=True,
            )

        try:
            resultado = self.cliente_llm.invocar_json(
                self.construir_system_prompt(),
                self.construir_user_prompt(lote),
            )
            return self.postprocesar_traza(
                lote,
                TrazaDecisionReAct(
                    id_traza=uuid4().hex[:8],
                    id_lote=lote.id_lote,
                    nombre_agente=self.nombre,
                    pensamiento=resultado["pensamiento"],
                    accion_sugerida=resultado["accion_sugerida"],
                    detalle_accion=resultado["detalle_accion"],
                    fue_fallback=False,
                ),
            )
        except Exception:
            logger.warning(
                "La evaluación individual falló; se ejecutan reglas de respaldo.",
                exc_info=True,
            )
            return self.ejecutar_fallback_reglas(lote)

    def postprocesar_traza(
        self, lote: LoteAlimento, traza: TrazaDecisionReAct
    ) -> TrazaDecisionReAct:
        if lote.id_lote != traza.id_lote:
            raise ValueError("La traza no corresponde al lote evaluado.")
        return traza

    @abstractmethod
    def construir_system_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def construir_user_prompt(self, lote: LoteAlimento) -> str:
        raise NotImplementedError

    @abstractmethod
    def ejecutar_fallback_reglas(
        self, lote: LoteAlimento
    ) -> TrazaDecisionReAct:
        raise NotImplementedError


class AgenteDeteccion(AgenteReActBase):
    def __init__(
        self,
        cliente_llm: ClienteOllamaResiliente,
        contexto_operativo: str = "Operación normal",
        bancos: list[BancoAlimentos] | None = None,
    ) -> None:
        super().__init__("AgenteDeteccion_ReAct", cliente_llm)
        self.contexto_operativo = contexto_operativo
        self.bancos = list(bancos) if bancos is not None else obtener_bancos_alimentos()

    def construir_system_prompt(self) -> str:
        return (
            "Actúa como un sistema experto en rescate alimentario, alineado "
            "con el ODS 2 y la Meta 12.3, siguiendo el patrón ReAct. Analiza "
            "los días para caducar, el peso total en kg y la categoría. Sugiere "
            "DONADO si faltan 1 o 2 días para caducar o si el peso total es "
            "alto para un banco de alimentos; sugiere DESCUENTO si faltan de "
            "3 a 5 días, ideal para venta social. Responde EXCLUSIVAMENTE con "
            "JSON válido y las llaves pensamiento, accion_sugerida y "
            "detalle_accion. accion_sugerida debe ser exactamente DONADO o "
            "DESCUENTO. detalle_accion debe ser una recomendación concreta, "
            "como asignar recolección urgente a un banco de alimentos o "
            "publicar una oferta social con 40% de descuento. Si sugieres "
            "DONADO, asigna un banco que tenga capacidad suficiente y cadena "
            "de frío para carnes y lácteos, y prioriza la menor distancia."
        )

    def construir_user_prompt(self, lote: LoteAlimento) -> str:
        bancos = "; ".join(
            f"{banco.nombre}: {banco.distancia_km:.1f} km, capacidad "
            f"{banco.capacidad_disponible_kg:.1f} kg, cadena de frío "
            f"{'sí' if banco.tiene_cadena_frio else 'no'}, prioridad "
            f"{banco.prioridad_nutricional}"
            for banco in self.bancos
        )
        return (
            f"Evalúa el lote {lote.id_lote}: producto={lote.producto}, "
            f"categoría={lote.categoria}, días para caducar="
            f"{lote.dias_para_caducar}, peso total={lote.peso_total_kg:.2f} kg, "
            f"precio base={lote.precio_base:.2f}. Contexto operativo: "
            f"{self.contexto_operativo}. Bancos disponibles: {bancos}. "
            "Adapta la acción al contexto e incluye la asignación logística "
            "si la acción es DONADO."
        )

    def construir_prompt_global(
        self,
        lotes: list[LoteAlimento],
        lotes_para_combos: list[LoteAlimento],
        contexto_operativo: str,
    ) -> tuple[str, str]:
        bancos = [
            {
                "id_banco": banco.id_banco,
                "nombre": banco.nombre,
                "distancia_km": banco.distancia_km,
                "capacidad_disponible_kg": banco.capacidad_disponible_kg,
                "tiene_cadena_frio": banco.tiene_cadena_frio,
                "prioridad_nutricional": banco.prioridad_nutricional,
            }
            for banco in self.bancos
        ]
        lotes_json = [
            {
                "id_lote": lote.id_lote,
                "producto": lote.producto,
                "categoria": lote.categoria,
                "dias_para_caducar": lote.dias_para_caducar,
                "peso_total_kg": lote.peso_total_kg,
                "precio_base": lote.precio_base,
            }
            for lote in lotes
        ]
        ingredientes_json = [
            {
                "id_lote": lote.id_lote,
                "producto": lote.producto,
                "categoria": lote.categoria,
                "dias_para_caducar": lote.dias_para_caducar,
                "precio_base": lote.precio_base,
            }
            for lote in lotes_para_combos
        ]
        system_prompt = (
            "Eres un agente experto de rescate alimentario ODS 2 y Meta 12.3. "
            "Analiza el conjunto completo en una sola respuesta. Aplica la "
            "cadena de frío estrictamente a carnes y lácteos, no superes la "
            "capacidad de cada banco y prioriza la menor distancia. Adapta las "
            "decisiones al contexto operativo. Sugiere DONADO o DESCUENTO para "
            "cada lote evaluado. Crea uno o dos combos gastronómicos con "
            "ingredientes compatibles de la lista de candidatos. Responde "
            "exclusivamente un objeto JSON con las llaves 'decisiones' y "
            "'combos'. Cada decisión incluye id_lote, pensamiento, "
            "accion_sugerida y detalle_accion. Cada combo incluye nombre_combo, "
            "lotes_incluidos, productos_descripcion, receta_sugerida y "
            "porcentaje_descuento entre 40 y 55. La receta debe ser breve y "
            "paso a paso. No inventes IDs."
        )
        user_prompt = (
            f"Contexto operativo: {contexto_operativo}\n"
            f"Lotes a evaluar: {json.dumps(lotes_json, ensure_ascii=False)}\n"
            f"Bancos de alimentos: {json.dumps(bancos, ensure_ascii=False)}\n"
            "Candidatos para combos con vigencia de 2 a 5 días: "
            f"{json.dumps(ingredientes_json, ensure_ascii=False)}"
        )
        return system_prompt, user_prompt

    def construir_prompt_combos(
        self, lotes: list[LoteAlimento], contexto_operativo: str
    ) -> tuple[str, str]:
        disponibles = [
            {
                "id_lote": lote.id_lote,
                "producto": lote.producto,
                "categoria": lote.categoria,
                "dias_para_caducar": lote.dias_para_caducar,
                "precio_base": lote.precio_base,
            }
            for lote in lotes
        ]
        return (
            "Eres un agente promotor de rescate alimentario. Cruza ingredientes "
            "compatibles de los productos disponibles y crea uno o dos combos "
            "con receta breve paso a paso. Devuelve exclusivamente JSON con "
            "la llave 'combos', una lista de objetos con nombre_combo, "
            "lotes_incluidos (IDs existentes), productos_descripcion, "
            "receta_sugerida y porcentaje_descuento entre 40 y 55. No inventes "
            "ingredientes ni IDs.",
            f"Contexto operativo: {contexto_operativo}\nProductos disponibles: "
            f"{json.dumps(disponibles, ensure_ascii=False)}",
        )

    def postprocesar_traza(
        self, lote: LoteAlimento, traza: TrazaDecisionReAct
    ) -> TrazaDecisionReAct:
        contexto = _normalizar_texto(self.contexto_operativo)
        categoria = lote.categoria
        perecedero = categoria in {"Lacteos", "Carnes"}
        contexto_frio = "refrigeracion" in contexto or "frio" in contexto
        contexto_infantil = "infantil" in contexto or "desayuno" in contexto

        if (
            contexto_frio
            and perecedero
            and lote.dias_para_caducar <= 4
        ):
            return self.ejecutar_fallback_reglas(lote)
        if (
            contexto_infantil
            and categoria in {"Panaderia", "Frutas_Verduras", "Lacteos"}
            and self._seleccionar_banco(lote, contexto) is not None
        ):
            return self.ejecutar_fallback_reglas(lote)

        if traza.accion_sugerida == "DONADO":
            banco = self._seleccionar_banco(lote, contexto)
            if banco is None:
                return self.ejecutar_fallback_reglas(lote)
            detalle = (
                f"{traza.detalle_accion} Banco asignado: {banco.nombre}, "
                f"a {banco.distancia_km:.1f} km; "
                f"{'con' if banco.tiene_cadena_frio else 'sin'} "
                "cadena de frío. "
                f"Prioridad: {banco.prioridad_nutricional}."
            )
            return TrazaDecisionReAct(
                id_traza=traza.id_traza,
                id_lote=traza.id_lote,
                nombre_agente=traza.nombre_agente,
                pensamiento=traza.pensamiento,
                accion_sugerida=traza.accion_sugerida,
                detalle_accion=detalle,
                fue_fallback=traza.fue_fallback,
                autorizada=traza.autorizada,
                fecha_registro=traza.fecha_registro,
            )

        if ("lluvia" in contexto or "trafico" in contexto) and "50%" not in traza.detalle_accion:
            traza.detalle_accion = (
                f"{traza.detalle_accion} Contexto de lluvia/tráfico: "
                "ofrecer 50% de descuento para atraer compradores locales."
            )
        return traza

    def _seleccionar_banco(
        self, lote: LoteAlimento, contexto: str | None = None
    ) -> BancoAlimentos | None:
        contexto = contexto or _normalizar_texto(self.contexto_operativo)
        elegibles = [
            banco for banco in self.bancos if banco.puede_recibir(lote)
        ]
        if (
            "refrigeracion" in contexto or "frio" in contexto
        ) and lote.categoria in {"Lacteos", "Carnes"}:
            elegibles = [
                banco for banco in elegibles if banco.tiene_cadena_frio
            ]
        if (
            ("infantil" in contexto or "desayuno" in contexto)
            and lote.categoria in {"Panaderia", "Frutas_Verduras", "Lacteos"}
        ):
            elegibles.sort(
                key=lambda banco: (
                    "comedor infantil" not in _normalizar_texto(banco.nombre),
                    banco.distancia_km,
                )
            )
        else:
            elegibles.sort(key=lambda banco: banco.distancia_km)
        return elegibles[0] if elegibles else None

    def crear_traza_desde_decision(
        self,
        lote: LoteAlimento,
        pensamiento: str,
        accion_sugerida: str,
        detalle_accion: str,
        fue_fallback: bool,
    ) -> TrazaDecisionReAct:
        return self.postprocesar_traza(
            lote,
            TrazaDecisionReAct(
                id_traza=uuid4().hex[:8],
                id_lote=lote.id_lote,
                nombre_agente=self.nombre,
                pensamiento=pensamiento,
                accion_sugerida=accion_sugerida,
                detalle_accion=detalle_accion,
                fue_fallback=fue_fallback,
            ),
        )

    def ejecutar_fallback_reglas(
        self, lote: LoteAlimento
    ) -> TrazaDecisionReAct:
        contexto = _normalizar_texto(self.contexto_operativo)
        if lote.esta_caducado:
            return TrazaDecisionReAct(
                id_traza=uuid4().hex[:8],
                id_lote=lote.id_lote,
                nombre_agente=self.nombre,
                pensamiento=(
                    "El lote está caducado y, por normativa sanitaria, no se "
                    "puede donar ni vender."
                ),
                accion_sugerida="DESCARTADO",
                detalle_accion="Retirar el lote y descartarlo según el "
                "procedimiento sanitario.",
                fue_fallback=True,
            )

        categoria_frio = lote.categoria in {"Lacteos", "Carnes"}
        alerta_frio = (
            ("refrigeracion" in contexto or "frio" in contexto)
            and categoria_frio
            and lote.dias_para_caducar <= 4
        )
        alerta_lluvia = "lluvia" in contexto or "trafico" in contexto
        alerta_infantil = "infantil" in contexto or "desayuno" in contexto
        banco = self._seleccionar_banco(lote, contexto)

        if alerta_frio:
            if banco is not None and banco.tiene_cadena_frio:
                accion = "DONADO"
                pensamiento = (
                    "La alerta de refrigeración y la ventana de caducidad "
                    "requieren donación urgente a un receptor con cadena de frío."
                )
            else:
                accion = "DESCUENTO"
                pensamiento = (
                    "La alerta de refrigeración exige salida urgente; no hay "
                    "un banco compatible con capacidad y cadena de frío, por "
                    "lo que se activa un descuento de emergencia del 55%."
                )
        elif (
            alerta_infantil
            and lote.categoria
            in {"Panaderia", "Frutas_Verduras", "Lacteos"}
            and banco is not None
        ):
            accion = "DONADO"
            pensamiento = (
                "La solicitud del comedor infantil prioriza este producto "
                "compatible para desayuno y nutrición inmediata."
            )
        elif lote.dias_para_caducar <= 2:
            accion = "DONADO"
            pensamiento = (
                f"Respaldo determinista: quedan {lote.dias_para_caducar} días "
                "para caducar; se prioriza la donación inmediata."
            )
        else:
            accion = "DESCUENTO"
            pensamiento = (
                f"Respaldo determinista: quedan {lote.dias_para_caducar} días "
                "para caducar; se propone venta social."
            )

        if accion == "DONADO" and banco is None:
            accion = "DESCUENTO"
            pensamiento += (
                " No hay un banco que cumpla capacidad y requisitos de frío; "
                "se cambia a venta social."
            )

        if accion == "DONADO":
            detalle = self._detalle_banco(banco)
        else:
            if alerta_frio:
                descuento = 55
            elif alerta_lluvia:
                descuento = 50
            else:
                descuento = 40
            precio_rebajado = lote.precio_base * (1 - descuento / 100)
            detalle = (
                f"Publicar oferta social con {descuento}% de descuento; "
                f"precio rebajado: {precio_rebajado:.2f}."
            )
            if alerta_lluvia:
                detalle += " Promocionar a compradores locales por lluvia/tráfico."

        return TrazaDecisionReAct(
            id_traza=uuid4().hex[:8],
            id_lote=lote.id_lote,
            nombre_agente=self.nombre,
            pensamiento=pensamiento,
            accion_sugerida=accion,
            detalle_accion=detalle,
            fue_fallback=True,
        )

    @staticmethod
    def _detalle_banco(banco: BancoAlimentos) -> str:
        return (
            f"Asignar recolección a {banco.nombre}, a "
            f"{banco.distancia_km:.1f} km; ruta "
            f"{'con' if banco.tiene_cadena_frio else 'sin'} cadena de frío. "
            f"Prioridad: {banco.prioridad_nutricional}."
        )

class OrquestadorRescate:
    def __init__(
        self,
        db: InventarioDB,
        agentes: list[AgenteReActBase] | None = None,
    ) -> None:
        self.db = db
        self.agentes = list(agentes) if agentes is not None else []

    def registrar_agente(self, agente: AgenteReActBase) -> None:
        self.agentes.append(agente)

    def procesar_inventario_en_riesgo(
        self,
        dias_umbral: int = 5,
        max_lotes: int = 6,
        modo_ejecucion: str = "rapido_global",
        contexto_operativo: str = "Operación normal",
    ) -> tuple[list[TrazaDecisionReAct], list[ComboRescate]]:
        if max_lotes < 0:
            raise ValueError("max_lotes no puede ser negativo.")
        modos_validos = {
            "rapido_global",
            "exhaustivo_lote",
            "fallback_determinista",
        }
        if modo_ejecucion not in modos_validos:
            raise ValueError(
                f"Modo de ejecución no válido: {modo_ejecucion!r}. "
                f"Debe ser uno de {sorted(modos_validos)}."
            )

        lotes_activos = self.db.obtener_lotes_activos()
        lotes_en_riesgo = sorted(
            (
                lote
                for lote in lotes_activos
                if lote.dias_para_caducar <= dias_umbral
            ),
            key=lambda lote: lote.fecha_caducidad,
        )
        lotes_a_evaluar = lotes_en_riesgo[:max_lotes]
        lotes_para_combos = [
            lote
            for lote in lotes_en_riesgo
            if 2 <= lote.dias_para_caducar <= 5
        ]
        agente_deteccion = self._obtener_agente_deteccion()
        agente_deteccion.contexto_operativo = contexto_operativo

        if modo_ejecucion == "fallback_determinista":
            trazas = self._evaluar_fallback(lotes_a_evaluar, agente_deteccion)
            combos = self._crear_combos_deterministas(lotes_para_combos)
            return trazas, combos

        if modo_ejecucion == "exhaustivo_lote":
            trazas: list[TrazaDecisionReAct] = []
            agentes = self.agentes or [agente_deteccion]
            for lote in lotes_a_evaluar:
                for agente in agentes:
                    if isinstance(agente, AgenteDeteccion):
                        agente.contexto_operativo = contexto_operativo
                    traza = agente.evaluar_lote(lote)
                    trazas.append(traza)
            self.db.guardar_trazas(trazas)
            combos = self.generar_combos_rescate(
                lotes_para_combos,
                usar_llm=True,
                contexto_operativo=contexto_operativo,
            )
            return trazas, combos

        if not lotes_a_evaluar:
            return [], self._crear_combos_deterministas(lotes_para_combos)

        return self._procesar_rapido_global(
            lotes_a_evaluar,
            lotes_para_combos,
            agente_deteccion,
            contexto_operativo,
        )

    def _procesar_rapido_global(
        self,
        lotes: list[LoteAlimento],
        lotes_para_combos: list[LoteAlimento],
        agente: AgenteDeteccion,
        contexto_operativo: str,
    ) -> tuple[list[TrazaDecisionReAct], list[ComboRescate]]:
        system_prompt, user_prompt = agente.construir_prompt_global(
            lotes,
            lotes_para_combos,
            contexto_operativo,
        )
        try:
            respuesta = agente.cliente_llm.invocar_json_estructurado(
                system_prompt,
                user_prompt,
                max_tokens=768,
                max_intentos=1,
            )
            decisiones = respuesta["decisiones"]
            if not isinstance(decisiones, list):
                raise ValueError("'decisiones' debe ser una lista.")
            decisiones_por_lote = {}
            ids_evaluados = {lote.id_lote for lote in lotes}
            for decision in decisiones:
                if (
                    not isinstance(decision, dict)
                    or decision.get("id_lote") not in ids_evaluados
                ):
                    continue
                decision_validada = dict(decision)
                ClienteOllamaResiliente._validar_decision(decision_validada)
                decisiones_por_lote[decision_validada["id_lote"]] = (
                    decision_validada
                )
        except Exception:
            logger.warning(
                "Falló la respuesta del análisis global; se usará el motor "
                "determinista.",
                exc_info=True,
            )
            trazas = self._evaluar_fallback(lotes, agente)
            combos = self._crear_combos_deterministas(lotes_para_combos)
            return trazas, combos

        trazas = []
        for lote in lotes:
            decision = decisiones_por_lote.get(lote.id_lote)
            if decision is None:
                traza = agente.ejecutar_fallback_reglas(lote)
            else:
                traza = agente.crear_traza_desde_decision(
                    lote,
                    pensamiento=decision["pensamiento"],
                    accion_sugerida=decision["accion_sugerida"],
                    detalle_accion=decision["detalle_accion"],
                    fue_fallback=False,
                )
            trazas.append(traza)

        combos = self._combos_desde_respuesta(
            respuesta.get("combos", []), lotes_para_combos
        )
        if not combos:
            combos = self._crear_combos_deterministas(lotes_para_combos)
        self.db.guardar_trazas(trazas)
        return trazas, combos

    def _evaluar_fallback(
        self,
        lotes: list[LoteAlimento],
        agente: AgenteDeteccion,
    ) -> list[TrazaDecisionReAct]:
        trazas = []
        for lote in lotes:
            traza = agente.ejecutar_fallback_reglas(lote)
            trazas.append(traza)
        self.db.guardar_trazas(trazas)
        return trazas

    def _obtener_agente_deteccion(self) -> AgenteDeteccion:
        for agente in self.agentes:
            if isinstance(agente, AgenteDeteccion):
                return agente
        agente = AgenteDeteccion(ClienteOllamaResiliente())
        self.agentes.append(agente)
        return agente

    def generar_combos_rescate(
        self,
        lotes_en_riesgo: list[LoteAlimento],
        usar_llm: bool = True,
        contexto_operativo: str = "Operación normal",
    ) -> list[ComboRescate]:
        candidatos = [
            lote
            for lote in lotes_en_riesgo
            if lote.estado == "ACTIVO" and 2 <= lote.dias_para_caducar <= 5
        ]
        if not candidatos:
            return []

        if usar_llm:
            agente = self._obtener_agente_deteccion()
            agente.contexto_operativo = contexto_operativo
            system_prompt, user_prompt = agente.construir_prompt_combos(
                candidatos, contexto_operativo
            )
            try:
                respuesta = agente.cliente_llm.invocar_json_estructurado(
                    system_prompt,
                    user_prompt,
                    max_tokens=500,
                )
                combos = self._combos_desde_respuesta(
                    respuesta.get("combos", []), candidatos
                )
                if combos:
                    return combos
            except Exception:
                logger.warning(
                    "No se pudieron generar combos con Ollama; se usarán "
                    "las reglas gastronómicas locales.",
                    exc_info=True,
                )

        return self._crear_combos_deterministas(candidatos)

    @staticmethod
    def _combos_desde_respuesta(
        resultados: object, candidatos: list[LoteAlimento]
    ) -> list[ComboRescate]:
        if not isinstance(resultados, list):
            return []
        lotes_por_id = {lote.id_lote: lote for lote in candidatos}
        combos: list[ComboRescate] = []
        for resultado in resultados[:2]:
            if not isinstance(resultado, dict):
                continue
            ids = resultado.get("lotes_incluidos")
            if (
                not isinstance(ids, list)
                or len(ids) < 2
                or any(not isinstance(id_lote, str) for id_lote in ids)
                or len(set(ids)) != len(ids)
                or any(id_lote not in lotes_por_id for id_lote in ids)
            ):
                continue
            nombre = resultado.get("nombre_combo")
            descripcion = resultado.get("productos_descripcion")
            receta = resultado.get("receta_sugerida")
            try:
                descuento = int(resultado["porcentaje_descuento"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not all(isinstance(valor, str) and valor.strip() for valor in (
                    nombre,
                    descripcion,
                    receta,
                ))
                or not 40 <= descuento <= 55
            ):
                continue

            lotes_combo = [lotes_por_id[id_lote] for id_lote in ids]
            precio_original = round(
                sum(lote.precio_base for lote in lotes_combo), 2
            )
            combos.append(
                ComboRescate(
                    id_combo=uuid4().hex[:8],
                    nombre_combo=nombre,
                    lotes_incluidos=ids,
                    productos_descripcion=descripcion,
                    receta_sugerida=receta,
                    precio_original_total=precio_original,
                    precio_combo_descuento=round(
                        precio_original * (1 - descuento / 100), 2
                    ),
                    porcentaje_descuento=descuento,
                    origen="Inferencia LLM (Ollama)",
                )
            )
        return combos

    @staticmethod
    def _crear_combos_deterministas(
        candidatos: list[LoteAlimento],
    ) -> list[ComboRescate]:
        elegibles = [
            lote
            for lote in candidatos
            if lote.estado == "ACTIVO" and 2 <= lote.dias_para_caducar <= 5
        ]

        def buscar(*terminos: str, excluir: set[str] | None = None):
            excluir = excluir or set()
            terminos_normalizados = tuple(
                _normalizar_texto(termino) for termino in terminos
            )
            return next(
                (
                    lote
                    for lote in elegibles
                    if lote.id_lote not in excluir
                    and any(
                        termino in _normalizar_texto(lote.producto)
                        for termino in terminos_normalizados
                    )
                ),
                None,
            )

        combos: list[ComboRescate] = []
        usados: set[str] = set()
        carne = buscar("carne molida", "molida", "carne de res")
        tomate = buscar("tomate", excluir={carne.id_lote} if carne else set())
        espinaca = buscar(
            "espinaca",
            excluir={
                lote.id_lote for lote in (carne, tomate) if lote is not None
            },
        )
        if carne and tomate and espinaca:
            ingredientes = [carne, tomate, espinaca]
            pan = buscar(
                "baguette",
                "pan integral",
                excluir={lote.id_lote for lote in ingredientes},
            )
            if pan:
                ingredientes.append(pan)
            combos.append(
                OrquestadorRescate._crear_combo_determinista(
                    "Pack Guisado Casero Anti-Merma",
                    ingredientes,
                    "Sofríe la carne molida; agrega tomate picado y cocina. "
                    "Incorpora espinaca al final y sirve con baguette.",
                    45,
                )
            )
            usados.update(lote.id_lote for lote in ingredientes)

        yogur = buscar("yogur", "yogurt", excluir=usados)
        fruta = buscar("fresa", "fruta", excluir=usados)
        pan_desayuno = buscar(
            "croissant", "pan integral", "baguette", excluir=usados
        )
        if yogur and fruta and pan_desayuno:
            combos.append(
                OrquestadorRescate._crear_combo_determinista(
                    "Pack Desayuno Express",
                    [yogur, pan_desayuno, fruta],
                    "Sirve el yogur frío con fresas lavadas y rebanadas. "
                    "Acompaña con croissant o pan integral.",
                    40,
                )
            )
        return combos[:2]

    @staticmethod
    def _crear_combo_determinista(
        nombre: str,
        lotes: list[LoteAlimento],
        receta: str,
        descuento: int,
    ) -> ComboRescate:
        precio_original = round(sum(lote.precio_base for lote in lotes), 2)
        return ComboRescate(
            id_combo=uuid4().hex[:8],
            nombre_combo=nombre,
            lotes_incluidos=[lote.id_lote for lote in lotes],
            productos_descripcion=", ".join(
                lote.producto for lote in lotes
            ),
            receta_sugerida=receta,
            precio_original_total=precio_original,
            precio_combo_descuento=round(
                precio_original * (1 - descuento / 100), 2
            ),
            porcentaje_descuento=descuento,
            origen="Motor Determinista (Reglas)",
        )


class CalculadoraImpactoODS:
    FACTORES_CO2 = {
        "Carnes": 6.0,
        "Lacteos": 3.2,
        "Panaderia": 1.2,
        "Frutas_Verduras": 1.2,
    }
    FACTOR_CO2_POR_DEFECTO = 2.5

    def calcular_resumen_impacto(self, lotes: list[LoteAlimento]) -> dict:
        lotes_rescatados = [
            lote for lote in lotes if lote.estado in {"DONADO", "DESCUENTO"}
        ]
        kg_rescatados = sum(lote.peso_total_kg for lote in lotes_rescatados)
        co2_evitado_kg = round(
            sum(
                lote.peso_total_kg
                * self.FACTORES_CO2.get(
                    lote.categoria, self.FACTOR_CO2_POR_DEFECTO
                )
                for lote in lotes_rescatados
            ),
            2,
        )

        return {
            "kg_rescatados": kg_rescatados,
            "co2_evitado_kg": co2_evitado_kg,
            "raciones_equivalentes": int(round(kg_rescatados / 0.4)),
            "lotes_donados": sum(
                lote.estado == "DONADO" for lote in lotes_rescatados
            ),
            "lotes_descuento": sum(
                lote.estado == "DESCUENTO" for lote in lotes_rescatados
            ),
        }


if __name__ == "__main__":
    db = InventarioDB()
    db.inicializar_esquema()
    cantidad = db.cargar_semilla_csv("seed_inventory.csv")
    print(f"Lotes cargados: {cantidad}")

    agente = AgenteDeteccion(ClienteOllamaResiliente())
    orquestador = OrquestadorRescate(db, [agente])
    trazas, combos = orquestador.procesar_inventario_en_riesgo(
        dias_umbral=5,
        max_lotes=2,
        modo_ejecucion="fallback_determinista",
    )
    print(f"Trazas generadas: {len(trazas)}")
    print(f"Combos generados: {len(combos)}")

    if trazas:
        primera = trazas[0]
        origen = "fallback" if primera.fue_fallback else "Ollama"
        print(f"Primera traza - Lote: {primera.id_lote}")
        print(f"Acción sugerida: {primera.accion_sugerida}")
        print(f"Origen: {origen}")
        print(f"Pensamiento: {primera.pensamiento}")

from dataclasses import dataclass, field
from datetime import date, datetime, time


class LoteAlimento:
    ESTADOS_VALIDOS = {"ACTIVO", "DONADO", "DESCUENTO", "DESCARTADO"}

    def __init__(
        self,
        id_lote: str,
        producto: str,
        categoria: str,
        unidades: int,
        peso_unitario_kg: float,
        precio_base: float,
        fecha_caducidad: date,
        estado: str = "ACTIVO",
    ) -> None:
        self._id_lote = id_lote
        self._producto = producto
        self._categoria = categoria
        self._unidades = unidades
        self._peso_unitario_kg = peso_unitario_kg
        self._precio_base = precio_base
        self._fecha_caducidad = fecha_caducidad
        self._estado = estado
        self.validar_invariantes()

    @property
    def id_lote(self) -> str:
        return self._id_lote

    @property
    def producto(self) -> str:
        return self._producto

    @property
    def categoria(self) -> str:
        return self._categoria

    @property
    def unidades(self) -> int:
        return self._unidades

    @property
    def peso_unitario_kg(self) -> float:
        return self._peso_unitario_kg

    @property
    def precio_base(self) -> float:
        return self._precio_base

    @property
    def fecha_caducidad(self) -> date:
        return self._fecha_caducidad

    @property
    def estado(self) -> str:
        return self._estado

    @property
    def dias_para_caducar(self) -> int:
        return self.dias_restantes_Respecto_a()

    def dias_restantes_Respecto_a(
        self, fecha_referencia: date | None = None
    ) -> int:
        referencia = fecha_referencia or date.today()
        return (self._fecha_caducidad - referencia).days

    def tiempo_restante_formateado(
        self, ahora: datetime | None = None
    ) -> str:
        instante = ahora or datetime.now()
        vence = datetime.combine(self._fecha_caducidad, time(23, 59, 59))
        segundos_restantes = int((vence - instante).total_seconds())
        if segundos_restantes <= 0:
            return "CADUCADO (00d 00h 00m 00s)"

        dias, segundos_restantes = divmod(segundos_restantes, 24 * 60 * 60)
        horas, segundos_restantes = divmod(segundos_restantes, 60 * 60)
        minutos, segundos = divmod(segundos_restantes, 60)
        return f"{dias:02d}d {horas:02d}h {minutos:02d}m {segundos:02d}s"

    @property
    def peso_total_kg(self) -> float:
        return round(self._unidades * self._peso_unitario_kg, 2)

    @property
    def esta_caducado(self) -> bool:
        return self.dias_para_caducar < 0

    def validar_invariantes(self) -> None:
        if not self._id_lote or not self._id_lote.strip():
            raise ValueError("El id_lote no puede estar vacío.")
        if not self._producto or not self._producto.strip():
            raise ValueError("El producto no puede estar vacío.")
        if self._unidades <= 0:
            raise ValueError("Las unidades deben ser mayores que cero.")
        if self._peso_unitario_kg <= 0:
            raise ValueError("El peso_unitario_kg debe ser mayor que cero.")
        if self._precio_base <= 0:
            raise ValueError("El precio_base debe ser mayor que cero.")
        if self._estado not in self.ESTADOS_VALIDOS:
            raise ValueError(
                f"Estado no válido: {self._estado!r}. "
                f"Debe ser uno de {sorted(self.ESTADOS_VALIDOS)}."
            )


@dataclass
class TrazaDecisionReAct:
    id_traza: str
    id_lote: str
    nombre_agente: str
    pensamiento: str
    accion_sugerida: str
    detalle_accion: str
    fue_fallback: bool
    autorizada: bool = False
    fecha_registro: datetime = field(default_factory=datetime.now)


@dataclass
class BancoAlimentos:
    id_banco: str
    nombre: str
    distancia_km: float
    capacidad_disponible_kg: float
    tiene_cadena_frio: bool
    prioridad_nutricional: str

    def puede_recibir(self, lote: LoteAlimento) -> bool:
        if lote.peso_total_kg > self.capacidad_disponible_kg:
            return False
        if (
            lote.categoria in {"Carnes", "Lacteos"}
            and not self.tiene_cadena_frio
        ):
            return False
        return True


@dataclass
class ComboRescate:
    id_combo: str
    nombre_combo: str
    lotes_incluidos: list[str]
    productos_descripcion: str
    receta_sugerida: str
    precio_original_total: float
    precio_combo_descuento: float
    porcentaje_descuento: int
    origen: str

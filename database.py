import csv
from datetime import date, timedelta

import duckdb

from models import LoteAlimento, TrazaDecisionReAct


class InventarioDB:
    def __init__(self, db_path: str = "inventario_rescate.db") -> None:
        self.db_path = db_path

    def inicializar_esquema(self) -> None:
        with duckdb.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lotes_inventario (
                    id_lote VARCHAR PRIMARY KEY,
                    producto VARCHAR NOT NULL,
                    categoria VARCHAR NOT NULL,
                    unidades INTEGER NOT NULL,
                    peso_unitario_kg DOUBLE NOT NULL,
                    precio_base DOUBLE NOT NULL,
                    fecha_caducidad DATE NOT NULL,
                    estado VARCHAR NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historial_trazas (
                    id_traza VARCHAR PRIMARY KEY,
                    id_lote VARCHAR NOT NULL,
                    nombre_agente VARCHAR NOT NULL,
                    pensamiento VARCHAR NOT NULL,
                    accion_sugerida VARCHAR NOT NULL,
                    detalle_accion VARCHAR NOT NULL,
                    fue_fallback BOOLEAN NOT NULL,
                    autorizada BOOLEAN NOT NULL,
                    fecha_registro TIMESTAMP NOT NULL
                )
                """
            )

    def cargar_semilla_csv(self, ruta_csv: str = "seed_inventory.csv") -> int:
        lotes: list[LoteAlimento] = []
        fecha_base = date.today()

        with open(ruta_csv, newline="", encoding="utf-8") as archivo:
            lector = csv.DictReader(archivo)
            columnas_comunes = {
                "id_lote",
                "producto",
                "categoria",
                "unidades",
                "peso_unitario_kg",
                "precio_base",
                "estado",
            }
            columnas_csv = set(lector.fieldnames or [])
            usa_fecha_real = "fecha_caducidad" in columnas_csv
            if (
                lector.fieldnames is None
                or not columnas_comunes.issubset(columnas_csv)
                or not usa_fecha_real and "dias_vigencia" not in columnas_csv
            ):
                faltantes = columnas_comunes - columnas_csv
                if not usa_fecha_real and "dias_vigencia" not in columnas_csv:
                    faltantes.add("fecha_caducidad o dias_vigencia")
                raise ValueError(
                    f"El CSV no contiene todas las columnas requeridas: "
                    f"{sorted(faltantes)}."
                )

            for numero_fila, registro in enumerate(lector, start=2):
                try:
                    lote = LoteAlimento(
                        id_lote=registro["id_lote"],
                        producto=registro["producto"],
                        categoria=registro["categoria"],
                        unidades=int(registro["unidades"]),
                        peso_unitario_kg=float(registro["peso_unitario_kg"]),
                        precio_base=float(registro["precio_base"]),
                        fecha_caducidad=(
                            date.fromisoformat(registro["fecha_caducidad"])
                            if usa_fecha_real
                            else fecha_base
                            + timedelta(days=int(registro["dias_vigencia"]))
                        ),
                        estado=registro["estado"],
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Registro inválido en la fila {numero_fila}: {error}"
                    ) from error
                lotes.append(lote)

        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM lotes_inventario")
            conn.execute("DELETE FROM historial_trazas")
            for lote in lotes:
                self._insertar_lote_en_conexion(conn, lote)

        return len(lotes)

    def insertar_lote(self, lote: LoteAlimento) -> None:
        lote.validar_invariantes()
        with duckdb.connect(self.db_path) as conn:
            self._insertar_lote_en_conexion(conn, lote)

    def obtener_lotes_activos(self) -> list[LoteAlimento]:
        with duckdb.connect(self.db_path) as conn:
            filas = conn.execute(
                """
                SELECT id_lote, producto, categoria, unidades, peso_unitario_kg,
                       precio_base, fecha_caducidad, estado
                FROM lotes_inventario
                WHERE estado = 'ACTIVO'
                ORDER BY fecha_caducidad ASC
                """
            ).fetchall()
        return [self._fila_a_lote(fila) for fila in filas]

    def obtener_todos_los_lotes(self) -> list[LoteAlimento]:
        with duckdb.connect(self.db_path) as conn:
            filas = conn.execute(
                """
                SELECT id_lote, producto, categoria, unidades, peso_unitario_kg,
                       precio_base, fecha_caducidad, estado
                FROM lotes_inventario
                """
            ).fetchall()
        return [self._fila_a_lote(fila) for fila in filas]

    def limpiar_historial_trazas(self) -> None:
        with duckdb.connect(self.db_path) as conn:
            conn.execute("DELETE FROM historial_trazas")

    def guardar_traza(self, traza: TrazaDecisionReAct) -> None:
        self.guardar_trazas([traza])

    def guardar_trazas(self, trazas: list[TrazaDecisionReAct]) -> None:
        if not trazas:
            return

        with duckdb.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for traza in trazas:
                    conn.execute(
                        """
                        DELETE FROM historial_trazas
                        WHERE id_lote = ? AND autorizada = FALSE
                        """,
                        [traza.id_lote],
                    )
                    self._insertar_traza_en_conexion(conn, traza)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def autorizar_accion(
        self, id_traza: str, id_lote: str, nuevo_estado: str
    ) -> None:
        if nuevo_estado not in LoteAlimento.ESTADOS_VALIDOS:
            raise ValueError(
                f"Estado no válido: {nuevo_estado!r}. "
                f"Debe ser uno de {sorted(LoteAlimento.ESTADOS_VALIDOS)}."
            )

        with duckdb.connect(self.db_path) as conn:
            traza = conn.execute(
                """
                SELECT id_lote
                FROM historial_trazas
                WHERE id_traza = ?
                """,
                [id_traza],
            ).fetchone()
            if traza is None:
                raise ValueError(f"No existe la traza {id_traza!r}.")
            if traza[0] != id_lote:
                raise ValueError(
                    f"La traza {id_traza!r} no corresponde al lote {id_lote!r}."
                )

            lote = conn.execute(
                "SELECT 1 FROM lotes_inventario WHERE id_lote = ?", [id_lote]
            ).fetchone()
            if lote is None:
                raise ValueError(f"No existe el lote {id_lote!r}.")

            conn.execute(
                "UPDATE historial_trazas SET autorizada = TRUE WHERE id_traza = ?",
                [id_traza],
            )
            conn.execute(
                "UPDATE lotes_inventario SET estado = ? WHERE id_lote = ?",
                [nuevo_estado, id_lote],
            )

    def revertir_autorizacion(self, id_traza: str, id_lote: str) -> None:
        with duckdb.connect(self.db_path) as conn:
            traza = conn.execute(
                "SELECT id_lote FROM historial_trazas WHERE id_traza = ?",
                [id_traza],
            ).fetchone()
            if traza is None:
                raise ValueError(f"No existe la traza {id_traza!r}.")
            if traza[0] != id_lote:
                raise ValueError(
                    f"La traza {id_traza!r} no corresponde al lote {id_lote!r}."
                )

            lote = conn.execute(
                "SELECT 1 FROM lotes_inventario WHERE id_lote = ?", [id_lote]
            ).fetchone()
            if lote is None:
                raise ValueError(f"No existe el lote {id_lote!r}.")

            conn.execute(
                "UPDATE historial_trazas SET autorizada = FALSE WHERE id_traza = ?",
                [id_traza],
            )
            conn.execute(
                "UPDATE lotes_inventario SET estado = 'ACTIVO' WHERE id_lote = ?",
                [id_lote],
            )

    def autorizar_combo(self, trazas: list[TrazaDecisionReAct]) -> None:
        if not trazas:
            raise ValueError("El combo no contiene lotes para autorizar.")
        if any(traza.accion_sugerida != "DESCUENTO" for traza in trazas):
            raise ValueError("Todas las trazas del combo deben sugerir DESCUENTO.")
        if len({traza.id_lote for traza in trazas}) != len(trazas):
            raise ValueError("El combo contiene IDs de lote duplicados.")

        with duckdb.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for traza in trazas:
                    lote = conn.execute(
                        "SELECT estado FROM lotes_inventario WHERE id_lote = ?",
                        [traza.id_lote],
                    ).fetchone()
                    if lote is None:
                        raise ValueError(f"No existe el lote {traza.id_lote!r}.")
                    if lote[0] != "ACTIVO":
                        raise ValueError(
                            f"El lote {traza.id_lote!r} ya no está activo."
                        )

                    conn.execute(
                        """
                        DELETE FROM historial_trazas
                        WHERE id_lote = ? AND autorizada = FALSE
                        """,
                        [traza.id_lote],
                    )
                    self._insertar_traza_en_conexion(conn, traza)
                    conn.execute(
                        "UPDATE historial_trazas SET autorizada = TRUE "
                        "WHERE id_traza = ?",
                        [traza.id_traza],
                    )
                    conn.execute(
                        "UPDATE lotes_inventario SET estado = 'DESCUENTO' "
                        "WHERE id_lote = ?",
                        [traza.id_lote],
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def obtener_historial_trazas(self) -> list[TrazaDecisionReAct]:
        with duckdb.connect(self.db_path) as conn:
            filas = conn.execute(
                """
                SELECT id_traza, id_lote, nombre_agente, pensamiento,
                       accion_sugerida, detalle_accion, fue_fallback,
                       autorizada, fecha_registro
                FROM historial_trazas
                ORDER BY fecha_registro DESC
                """
            ).fetchall()
        return [
            TrazaDecisionReAct(
                id_traza=fila[0],
                id_lote=fila[1],
                nombre_agente=fila[2],
                pensamiento=fila[3],
                accion_sugerida=fila[4],
                detalle_accion=fila[5],
                fue_fallback=fila[6],
                autorizada=fila[7],
                fecha_registro=fila[8],
            )
            for fila in filas
        ]

    @staticmethod
    def _insertar_lote_en_conexion(
        conn: duckdb.DuckDBPyConnection, lote: LoteAlimento
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO lotes_inventario (
                id_lote, producto, categoria, unidades, peso_unitario_kg,
                precio_base, fecha_caducidad, estado
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                lote.id_lote,
                lote.producto,
                lote.categoria,
                lote.unidades,
                lote.peso_unitario_kg,
                lote.precio_base,
                lote.fecha_caducidad,
                lote.estado,
            ],
        )

    @staticmethod
    def _insertar_traza_en_conexion(
        conn: duckdb.DuckDBPyConnection, traza: TrazaDecisionReAct
    ) -> None:
        conn.execute(
            """
            INSERT INTO historial_trazas (
                id_traza, id_lote, nombre_agente, pensamiento,
                accion_sugerida, detalle_accion, fue_fallback,
                autorizada, fecha_registro
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                traza.id_traza,
                traza.id_lote,
                traza.nombre_agente,
                traza.pensamiento,
                traza.accion_sugerida,
                traza.detalle_accion,
                traza.fue_fallback,
                traza.autorizada,
                traza.fecha_registro,
            ],
        )

    @staticmethod
    def _fila_a_lote(fila: tuple) -> LoteAlimento:
        return LoteAlimento(
            id_lote=fila[0],
            producto=fila[1],
            categoria=fila[2],
            unidades=fila[3],
            peso_unitario_kg=fila[4],
            precio_base=fila[5],
            fecha_caducidad=fila[6],
            estado=fila[7],
        )


if __name__ == "__main__":
    db = InventarioDB()
    db.inicializar_esquema()
    cantidad = db.cargar_semilla_csv()
    print(f"Lotes cargados: {cantidad}")

    try:
        LoteAlimento(
            id_lote="PRUEBA-NEGATIVA",
            producto="Producto de prueba",
            categoria="Prueba",
            unidades=-1,
            peso_unitario_kg=1.0,
            precio_base=1.0,
            fecha_caducidad=date.today(),
        )
    except ValueError as error:
        print(f"Validación correcta: {error}")
    else:
        raise AssertionError("Se esperaba un ValueError para unidades negativas.")

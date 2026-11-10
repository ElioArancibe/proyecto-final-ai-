from pydantic import BaseModel, Field

from .jobs import EstadoJob


class NuevaTarea(BaseModel):
    consulta: str = Field(min_length=5, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=100, pattern=r"^[\w-]+$")


class TareaEncolada(BaseModel):
    job_id: str
    thread_id: str
    estado: EstadoJob


class Aprobacion(BaseModel):
    aprobado: bool
    comentario: str | None = Field(default=None, max_length=500)


class Orden(BaseModel):
    proveedor: str
    producto: str
    cantidad: int
    subtotal: int
    descuento_pct: int
    descuento: int
    envio: int
    total: int
    costo_por_unidad: int


class OrdenCompra(Orden):
    orden_id: str
    fecha: str
    aprobacion: dict


class Job(TareaEncolada):
    consulta: str
    creado: str
    actualizado: str
    resultado: str | None = None
    error: str | None = None
    pendiente: dict | None = None
    decision: dict | None = None
    orden: Orden | dict | None = None
    orden_id: str | None = None

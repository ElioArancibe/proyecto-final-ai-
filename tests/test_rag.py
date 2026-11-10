from app.config import get_settings
from app.rag import BaseConocimiento, _tokens, cargar_fragmentos


def test_fragmentos_con_id_unico_y_titulo():
    fragmentos = cargar_fragmentos(get_settings())
    ids = [f.chunk_id for f in fragmentos]
    assert len(ids) == len(set(ids))
    assert all(f.texto.startswith(("#", "[")) for f in fragmentos)


def test_tokens_sin_tildes_ni_stopwords():
    assert _tokens("¿Qué política de aprobación tiene ImportPC?") == ["politica", "aprobacion", "tiene", "importpc"]


async def test_busqueda_hibrida_fusiona_bm25_y_vectorial(monkeypatch):
    base = BaseConocimiento(get_settings())

    async def vectorial(consulta, n):
        return ["logistica::c00", "politica_compras::c00"]

    monkeypatch.setattr(base, "_buscar_vectorial", vectorial)
    resultado = await base.buscar("niveles de aprobación de la gerencia comercial", k=3)
    ids = [f.chunk_id for f in resultado]

    assert ids[0] == "politica_compras::c00"
    assert "logistica::c00" in ids
    await base.cerrar()


async def test_si_falla_la_vectorial_queda_bm25(monkeypatch):
    base = BaseConocimiento(get_settings())

    async def caida(consulta, n):
        raise RuntimeError("503 UNAVAILABLE")

    async def sin_espera(segundos):
        pass

    monkeypatch.setattr(base, "_buscar_vectorial", caida)
    monkeypatch.setattr("app.rag.asyncio.sleep", sin_espera)
    resultado = await base.buscar("penalidad por demora de ImportPC", k=2)

    assert resultado[0].doc_id == "proveedores_homologados"
    await base.cerrar()

import asyncio
import logging
import re
import unicodedata

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import AsyncPinecone, ServerlessSpec
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from .config import Settings

log = logging.getLogger("rag")

STOPWORDS = set(
    "a al como con cual de del donde el en es esta este hay la las lo los mas me no o para por que se si sin su sus un una y".split()
)


class Fragmento(BaseModel):
    chunk_id: str
    doc_id: str
    titulo: str
    texto: str


def _tokens(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS and len(t) > 1]


def cargar_fragmentos(settings: Settings) -> list[Fragmento]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    fragmentos = []
    for path in sorted(settings.docs_dir.glob("*.md")):
        texto = path.read_text(encoding="utf-8")
        titulo = texto.splitlines()[0].lstrip("# ").strip()
        for i, parte in enumerate(splitter.split_text(texto)):
            # el título va adelante para que un fragmento suelto no pierda de qué documento sale
            contenido = parte if i == 0 else f"[{titulo}]\n{parte}"
            fragmentos.append(Fragmento(chunk_id=f"{path.stem}::c{i:02d}", doc_id=path.stem, titulo=titulo, texto=contenido))
    return fragmentos


class BaseConocimiento:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fragmentos = cargar_fragmentos(settings)
        self.por_id = {f.chunk_id: f for f in self.fragmentos}
        self.bm25 = BM25Okapi([_tokens(f.texto) for f in self.fragmentos])
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key,
            output_dimensionality=settings.embedding_dimension,
        )
        self.pc = AsyncPinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self.indice = None

    async def preparar(self) -> None:
        nombre = self.settings.pinecone_index
        if not await self.pc.has_index(nombre):
            log.info("creando índice %s", nombre)
            await self.pc.create_index(
                name=nombre,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region),
            )
        self.indice = await self.pc.index(name=nombre)
        if await self._cantidad_vectores() < len(self.fragmentos):
            await self.ingestar()

    async def _cantidad_vectores(self) -> int:
        stats = await self.indice.describe_index_stats()
        ns = (stats.namespaces or {}).get(self.settings.pinecone_namespace)
        return ns.vector_count if ns else 0

    async def ingestar(self) -> int:
        log.info("ingestando %d fragmentos en %s", len(self.fragmentos), self.settings.pinecone_namespace)
        vectores = await self.embeddings.aembed_documents([f.texto for f in self.fragmentos])
        await self.indice.upsert(
            vectors=[
                {"id": f.chunk_id, "values": v, "metadata": {"doc_id": f.doc_id, "titulo": f.titulo}}
                for f, v in zip(self.fragmentos, vectores, strict=True)
            ],
            namespace=self.settings.pinecone_namespace,
            show_progress=False,
        )
        return len(vectores)

    async def _buscar_vectorial(self, consulta: str, n: int) -> list[str]:
        vector = await self.embeddings.aembed_query(consulta)
        respuesta = await self.indice.query(top_k=n, vector=vector, namespace=self.settings.pinecone_namespace)
        return [m.id for m in respuesta.matches if m.id in self.por_id]

    def _buscar_bm25(self, consulta: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(_tokens(consulta))
        orden = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.fragmentos[i].chunk_id for i in orden[:n] if scores[i] > 0]

    async def buscar(self, consulta: str, k: int | None = None) -> list[Fragmento]:
        k = k or self.settings.top_k
        n = k * 3
        vectorial, lexico = await asyncio.gather(
            self._buscar_vectorial(consulta, n),
            asyncio.to_thread(self._buscar_bm25, consulta, n),
        )
        # reciprocal rank fusion: se combinan posiciones, no los puntajes de cada buscador
        puntaje: dict[str, float] = {}
        for ids, peso in ((lexico, self.settings.bm25_peso), (vectorial, 1 - self.settings.bm25_peso)):
            for pos, chunk_id in enumerate(ids):
                puntaje[chunk_id] = puntaje.get(chunk_id, 0) + peso / (60 + pos)
        mejores = sorted(puntaje, key=puntaje.get, reverse=True)[:k]
        return [self.por_id[i] for i in mejores]

    async def cerrar(self) -> None:
        await self.pc.close()


async def main() -> None:
    from .config import get_settings

    base = BaseConocimiento(get_settings())
    try:
        await base.preparar()
        print(f"{await base._cantidad_vectores()} vectores en el índice")
        for f in await base.buscar("¿Qué aprobación necesita una orden de 3 millones?"):
            print(f"- {f.chunk_id}: {f.texto[:90]!r}")
    finally:
        await base.cerrar()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

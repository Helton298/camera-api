"""
API de ingestao - recebe eventos (fotos de rosto) e roda reconhecimento
facial contra o banco de rostos do cliente, gravando o evento no Supabase.

Dois jeitos de um evento chegar aqui:

1. POST /events (multipart) - usado pelo agente de captura customizado
   (agent/capture_agent.py) quando NAO estamos usando o Viseron.

2. POST /webhooks/viseron (JSON) - usado quando o Viseron esta rodando
   na borda e configurado com o componente `webhook` (ver
   viseron/config/config.yaml). O Viseron manda um JSON com a URL do
   snapshot dentro da propria rede docker, e a API busca a imagem.

Os dois caminhos convergem na mesma funcao `process_event_image()`:
1. Sobe a imagem para o Supabase Storage.
2. Gera o embedding facial (face_recognition / dlib).
3. Busca o rosto mais proximo no banco (pgvector) dentro do MESMO client_id
   da camera (nunca compara entre clientes diferentes).
4. Se achou match dentro do limiar -> evento "face_recognized".
   Se nao achou -> evento "face_unknown" (fica disponivel pra nomear no dashboard).
5. Grava o evento na tabela `events`.

Nota de setup: a biblioteca `face_recognition` depende do dlib, que precisa
de cmake/build tools no ambiente pra compilar. Ver README para instrucoes.

Nota de infra: essa API fala com o Postgres so por HTTPS, via PostgREST
(cliente supabase-py) - nunca abre conexao direta na porta 5432. Isso e
proposital: em Supabase self-hosted (e na maioria dos hosts gerenciados),
o Postgres nao fica acessivel publicamente, so a API REST fica. As duas
consultas que precisariam de SQL cru (join camera->site->client, e a busca
por similaridade no pgvector) viram funcoes SQL expostas via RPC - ver
002_rpc_functions.sql.
"""

import os
import uuid
from typing import Optional

import face_recognition
import numpy as np
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="DVR IA SaaS - API de eventos")

# CORS liberado geral - o dashboard (Artifact, roda no navegador) precisa
# chamar essa API de outro dominio. Ja existe autenticacao por API_KEY em
# cada rota, entao liberar origem nao abre a API pra quem nao tem a chave.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EXPECTED_API_KEY = os.environ.get("API_KEY", "")

# limiar de distancia para considerar "mesma pessoa" (face_recognition usa distancia euclidiana;
# quanto menor, mais parecido - 0.6 e o valor padrao recomendado pela lib)
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.6"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def check_auth(authorization: Optional[str]):
    token = (authorization or "").replace("Bearer ", "")
    if not EXPECTED_API_KEY or token != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API invalida")


def get_client_id_for_camera(camera_id: str) -> str:
    """Chama a funcao SQL get_client_id_for_camera via RPC (ver 002_rpc_functions.sql)."""
    result = supabase.rpc("get_client_id_for_camera", {"p_camera_id": camera_id}).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Camera nao encontrada")
    return result.data


def find_best_match(client_id: str, embedding: np.ndarray) -> Optional[dict]:
    """Busca o rosto mais proximo dentro do mesmo client_id, via RPC (pgvector)."""
    result = supabase.rpc(
        "match_face_embedding",
        {"query_embedding": embedding.tolist(), "match_client_id": client_id},
    ).execute()
    return result.data[0] if result.data else None


def get_or_create_person(client_id: str, name: str) -> str:
    """Reaproveita a pessoa se ja existe alguem com esse nome (mesmo client_id) -
    evita criar um registro duplicado toda vez que se adiciona mais uma foto de
    referencia da mesma pessoa (ex: nomeando varios eventos antigos dela)."""
    existing = (
        supabase.table("camera_people")
        .select("id")
        .eq("client_id", client_id)
        .ilike("name", name)
        .execute()
        .data
    )
    if existing:
        return existing[0]["id"]

    person_result = supabase.table("camera_people").insert({"client_id": client_id, "name": name}).execute()
    return person_result.data[0]["id"]


def process_event_image(camera_id: str, occurred_at: str, image_bytes: bytes) -> dict:
    """Logica central compartilhada pelos dois caminhos de ingestao (agent proprio e Viseron)."""
    client_id = get_client_id_for_camera(camera_id)

    # 1. sobe a imagem pro Storage
    storage_path = f"{client_id}/{camera_id}/{uuid.uuid4()}.jpg"
    supabase.storage.from_("event-photos").upload(storage_path, image_bytes, {"content-type": "image/jpeg"})
    image_url = supabase.storage.from_("event-photos").get_public_url(storage_path)

    # 2. extrai o embedding facial
    np_image = _bytes_to_face_image(image_bytes)
    encodings = face_recognition.face_encodings(np_image)

    if not encodings:
        # nenhum rosto detectavel na imagem recortada -> registra como pessoa_detectada generico
        event_type = "person_detected"
        person_id = None
        confidence = None
    else:
        embedding = encodings[0]
        match = find_best_match(client_id, embedding)
        if match and match["distance"] <= MATCH_THRESHOLD:
            event_type = "face_recognized"
            person_id = match["person_id"]
            confidence = float(match["distance"])
        else:
            event_type = "face_unknown"
            person_id = None
            confidence = None

    result = (
        supabase.table("camera_events")
        .insert(
            {
                "camera_id": camera_id,
                "event_type": event_type,
                "image_url": image_url,
                "person_id": person_id,
                "match_confidence": confidence,
                "occurred_at": occurred_at,
            }
        )
        .execute()
    )
    event_id = result.data[0]["id"]

    return {"event_id": event_id, "event_type": event_type, "person_id": person_id}


@app.post("/events")
async def create_event(
    camera_id: str = Form(...),
    occurred_at: str = Form(...),
    image: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Usado pelo agente de captura customizado (agent/capture_agent.py)."""
    check_auth(authorization)
    image_bytes = await image.read()
    return process_event_image(camera_id, occurred_at, image_bytes)


class ViseronWebhookPayload(BaseModel):
    camera_id: str
    occurred_at: str
    snapshot_url: str


@app.post("/webhooks/viseron")
async def viseron_webhook(
    payload: ViseronWebhookPayload,
    authorization: Optional[str] = Header(None),
):
    """
    Recebe o POST disparado pelo componente `webhook` do Viseron.

    IMPORTANTE: o payload exato que o Viseron envia depende de como voce
    templatizar o `webhook` no config.yaml (ver viseron/config/config.yaml).
    O formato usado aqui (camera_id, occurred_at, snapshot_url) e o que
    configuramos no template - ajuste os dois lados juntos se mudar um.

    `snapshot_url` deve ser uma URL acessivel pela API dentro da rede
    (ex: http://viseron:8888/... quando os dois rodam no mesmo docker-compose).
    """
    check_auth(authorization)

    try:
        resp = requests.get(payload.snapshot_url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao buscar snapshot do Viseron: {exc}")

    return process_event_image(payload.camera_id, payload.occurred_at, resp.content)


@app.post("/people")
async def register_person(
    client_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Cadastra uma pessoa nova a partir de uma foto de referencia (ex: a partir
    de um evento 'face_unknown' que o usuario decidiu nomear no dashboard)."""
    check_auth(authorization)

    image_bytes = await image.read()
    np_image = _bytes_to_face_image(image_bytes)
    encodings = face_recognition.face_encodings(np_image)
    if not encodings:
        raise HTTPException(status_code=400, detail="Nenhum rosto detectado na foto enviada")

    person_id = get_or_create_person(client_id, name)

    # guarda a propria foto de referencia no Storage, pra dar pra ver depois no dashboard
    photo_path = f"{client_id}/people/{person_id}/{uuid.uuid4()}.jpg"
    supabase.storage.from_("event-photos").upload(photo_path, image_bytes, {"content-type": "image/jpeg"})
    photo_url = supabase.storage.from_("event-photos").get_public_url(photo_path)

    supabase.table("camera_face_embeddings").insert(
        {"person_id": person_id, "embedding": encodings[0].tolist(), "photo_url": photo_url}
    ).execute()

    return {"person_id": person_id}


@app.post("/people/from-event")
async def register_person_from_event(
    event_id: str = Form(...),
    name: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    """Nomeia uma pessoa usando a foto de um evento ja existente (ex: um
    'face_unknown' visto no dashboard) - sem precisar tirar uma foto nova.
    Tambem atualiza esse mesmo evento pra 'face_recognized'."""
    check_auth(authorization)

    event_result = supabase.table("camera_events").select("*").eq("id", event_id).execute()
    if not event_result.data:
        raise HTTPException(status_code=404, detail="Evento nao encontrado")
    event = event_result.data[0]

    client_id = get_client_id_for_camera(event["camera_id"])

    resp = requests.get(event["image_url"], timeout=10)
    resp.raise_for_status()
    image_bytes = resp.content

    np_image = _bytes_to_face_image(image_bytes)
    encodings = face_recognition.face_encodings(np_image)
    if not encodings:
        raise HTTPException(status_code=400, detail="Nenhum rosto detectado na foto desse evento")

    person_id = get_or_create_person(client_id, name)

    # a foto ja existe no Storage (e o proprio evento) - so reaproveita a URL
    supabase.table("camera_face_embeddings").insert(
        {
            "person_id": person_id,
            "embedding": encodings[0].tolist(),
            "source_event_id": event_id,
            "photo_url": event["image_url"],
        }
    ).execute()

    supabase.table("camera_events").update(
        {"event_type": "face_recognized", "person_id": person_id, "match_confidence": 0.0}
    ).eq("id", event_id).execute()

    return {"person_id": person_id}


@app.get("/people")
def list_people(
    client_id: str,
    name: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """Lista/busca pessoas cadastradas de um cliente. `name` faz busca parcial (ex: 'fu' acha 'Fulano').
    Cada pessoa vem com `camera_face_embeddings` embutido (so id/photo_url/created_at) - as fotos
    de referencia usadas pra cadastra-la, pra dar pra ver no dashboard."""
    check_auth(authorization)
    query = (
        supabase.table("camera_people")
        .select("*, camera_face_embeddings(id, photo_url, created_at)")
        .eq("client_id", client_id)
        .order("name")
    )
    if name:
        query = query.ilike("name", f"%{name}%")
    return query.execute().data


@app.get("/events")
def list_events(
    camera_id: Optional[str] = None,
    person_id: Optional[str] = None,
    person_name: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
):
    """`person_name` e um atalho: busca pessoas com esse nome e ja filtra os eventos delas,
    pra nao precisar chamar /people antes so pra descobrir o person_id."""
    check_auth(authorization)

    query = supabase.table("camera_events").select("*").order("occurred_at", desc=True).limit(limit)
    if camera_id:
        query = query.eq("camera_id", camera_id)
    if person_id:
        query = query.eq("person_id", person_id)
    if person_name:
        people = supabase.table("camera_people").select("id").ilike("name", f"%{person_name}%").execute().data
        person_ids = [p["id"] for p in people]
        if not person_ids:
            return []
        query = query.in_("person_id", person_ids)
    return query.execute().data


def _bytes_to_face_image(image_bytes: bytes) -> np.ndarray:
    import io
    from PIL import Image

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)

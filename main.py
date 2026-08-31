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
"""

import os
import uuid
from typing import Optional

import face_recognition
import numpy as np
import psycopg2
import psycopg2.extras
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="DVR IA SaaS - API de eventos")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]  # connection string direta do Postgres (para pgvector)
EXPECTED_API_KEY = os.environ.get("API_KEY", "")

# limiar de distancia para considerar "mesma pessoa" (face_recognition usa distancia euclidiana;
# quanto menor, mais parecido - 0.6 e o valor padrao recomendado pela lib)
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.6"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def check_auth(authorization: Optional[str]):
    token = (authorization or "").replace("Bearer ", "")
    if not EXPECTED_API_KEY or token != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API invalida")


def get_client_id_for_camera(conn, camera_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            """
            select c.id
            from camera_cameras cam
            join camera_sites s on s.id = cam.site_id
            join camera_clients c on c.id = s.client_id
            where cam.id = %s
            """,
            (camera_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Camera nao encontrada")
        return row[0]


def find_best_match(conn, client_id: str, embedding: np.ndarray):
    """Busca o rosto mais proximo dentro do mesmo client_id usando pgvector."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select fe.person_id, fe.embedding <-> %s::vector as distance
            from camera_face_embeddings fe
            join camera_people p on p.id = fe.person_id
            where p.client_id = %s
            order by fe.embedding <-> %s::vector
            limit 1
            """,
            (embedding.tolist(), client_id, embedding.tolist()),
        )
        return cur.fetchone()


def process_event_image(conn, camera_id: str, occurred_at: str, image_bytes: bytes) -> dict:
    """Logica central compartilhada pelos dois caminhos de ingestao (agent proprio e Viseron)."""
    client_id = get_client_id_for_camera(conn, camera_id)

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
        match = find_best_match(conn, client_id, embedding)
        if match and match["distance"] <= MATCH_THRESHOLD:
            event_type = "face_recognized"
            person_id = match["person_id"]
            confidence = float(match["distance"])
        else:
            event_type = "face_unknown"
            person_id = None
            confidence = None

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into camera_events (camera_id, event_type, image_url, person_id, match_confidence, occurred_at)
            values (%s, %s, %s, %s, %s, %s)
            returning id
            """,
            (camera_id, event_type, image_url, person_id, confidence, occurred_at),
        )
        event_id = cur.fetchone()[0]
        conn.commit()

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

    conn = get_db_connection()
    try:
        return process_event_image(conn, camera_id, occurred_at, image_bytes)
    finally:
        conn.close()


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

    conn = get_db_connection()
    try:
        return process_event_image(conn, payload.camera_id, payload.occurred_at, resp.content)
    finally:
        conn.close()


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

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "insert into camera_people (client_id, name) values (%s, %s) returning id",
                (client_id, name),
            )
            person_id = cur.fetchone()[0]
            cur.execute(
                "insert into camera_face_embeddings (person_id, embedding) values (%s, %s::vector)",
                (person_id, encodings[0].tolist()),
            )
            conn.commit()
        return {"person_id": person_id}
    finally:
        conn.close()


@app.get("/events")
def list_events(
    camera_id: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
):
    check_auth(authorization)
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if camera_id:
                cur.execute(
                    "select * from camera_events where camera_id = %s order by occurred_at desc limit %s",
                    (camera_id, limit),
                )
            else:
                cur.execute("select * from camera_events order by occurred_at desc limit %s", (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def _bytes_to_face_image(image_bytes: bytes) -> np.ndarray:
    import io
    from PIL import Image

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)

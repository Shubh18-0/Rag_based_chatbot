import redis
import hashlib ,json

redis_server = redis.Redis.from_url(
    "redis://default:ASLdAAImcDE5OTAxMjk5MjYzOGI0NTIwOWNlOWQ4YTVhNTQ3MzAyZnAxODkyNQ@massive-goat-8925.upstash.io:6379",
    decode_responses=True)

def make_hash_key(text):
    return hashlib.sha256(text.encode()).hexdigest()

def getquery_cache(query, session_id):
    return f'{session_id}:query:{make_hash_key(query)}'

def cache_answer(query, answer, session_id):
    redis_server.set(getquery_cache(query, session_id), answer)

def get_cached_answer(query, session_id):
    cached = redis_server.get(getquery_cache(query, session_id))
    return cached if cached else None

def retrieval_cache(query,session_id,documents):
    key=f"{session_id}:retrieval:{make_hash_key(query)}"
    redis_server.set(key,json.dumps(documents),ex=3600)

def get_cached_retrieval(query,session_id):
    key=f'{session_id}:retrieval:{make_hash_key(query)}'
    cached=redis_server.get(key)
    return json.loads(cached) if cached else None
